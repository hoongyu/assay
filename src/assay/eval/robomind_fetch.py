"""只读关节地下 RoboMIND 公开的失败示范（Hugging Face 上 failure_data/，1,678 条、440.9 GB，大头是图像）。

按字节范围只读每条 trajectory.hdf5 里不是图像的那几块（关节、指令、仿真时间），每条拉 1 到 1.5 MB；
存成不带图像的 hdf5，图像只留同名的空数据集（读取器靠它认相机名），目录结构和官方一样。
数据集要先在 Hugging Face 上同意条款并登录（uv sync --extra hub，再 hf auth login）。
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = "x-humanoid-robomind/RoboMIND"
PREFIX = "failure_data/"
CAP_BYTES = 5 * 2**20  # 每条读过这么多还没读完关节，就跳过（多半是布局不认得、读到了图像）


class TooBig(Exception):
    pass


class CappedReader:
    """包住一个可 seek 的二进制文件对象，数被读走了多少字节，超过上限就抛 TooBig。"""

    def __init__(self, f, cap: int = CAP_BYTES):
        self.f, self.cap, self.n = f, cap, 0

    def _count(self, k: int) -> None:
        self.n += k
        if self.n > self.cap:
            raise TooBig(f"读过 {self.cap / 2**20:.0f} MB 还没读完关节")

    def readinto(self, b) -> int:
        k = self.f.readinto(b)
        self._count(k or 0)
        return k

    def read(self, size: int = -1) -> bytes:
        data = self.f.read(size)
        self._count(len(data))
        return data

    def seek(self, *args) -> int:
        return self.f.seek(*args)

    def tell(self) -> int:
        return self.f.tell()

    def __getattr__(self, name):
        return getattr(self.f, name)


def copy_joints(f, dst: Path, source: str, source_bytes: int) -> dict:
    """从 f（h5py 能打开的文件对象）里拷出不是图像的数据集和根属性，写成 dst；observations/ 下的图像留同名空数据集，
    记下原来的形状。先写 .part，写完再改名，中途出错不留半截文件。"""
    import h5py
    import numpy as np

    arrays: dict = {}
    images: list[tuple[str, tuple]] = []
    with h5py.File(f, "r") as h:
        attrs = dict(h.attrs.items())

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name.startswith("observations/"):
                images.append((name, obj.shape))
            else:
                arrays[name] = obj[()]

        h.visititems(visit)
    tmp = dst.with_suffix(".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp, "w") as g:
        for k, v in attrs.items():
            g.attrs[k] = v
        g.attrs["assay_joints_only"] = True
        g.attrs["assay_source"] = source
        g.attrs["assay_source_bytes"] = source_bytes
        for name, v in arrays.items():
            g.create_dataset(name, data=v)
        for name, shape in images:
            g.create_dataset(name, data=np.zeros(0, dtype=np.uint8))
            g[name].attrs["original_shape"] = shape
    tmp.replace(dst)
    frames = max((len(v) for v in arrays.values() if getattr(v, "ndim", 0) == 2), default=None)
    return {"frames": frames, "keys": {k: list(getattr(v, "shape", ())) for k, v in arrays.items()}}


def _fetch_one(path: str, size: int, out: str) -> dict:
    """一条：按字节范围读，网络和限流出错歇一会儿再试，最多四回。在子进程里跑。"""
    from huggingface_hub import HfFileSystem

    dst = Path(out) / path.removeprefix(PREFIX)
    if dst.exists():
        return {"path": path, "status": "已有"}
    fs = HfFileSystem()
    why = ""
    for attempt in range(4):
        try:
            with fs.open(f"datasets/{REPO}/{path}", "rb", block_size=256 * 1024, cache_type="blockcache") as raw:
                f = CappedReader(raw)
                info = copy_joints(f, dst, source=f"hf://datasets/{REPO}/{path}", source_bytes=size)
            return {"path": path, "status": "好", "read_bytes": f.n, **info}
        except TooBig as exc:
            return {"path": path, "status": "太大跳过", "why": str(exc)}
        except Exception as exc:  # 网络抖、限流
            why = f"{type(exc).__name__}: {exc}"
            time.sleep(15 * (attempt + 1))
    return {"path": path, "status": "失败", "why": why}


def fetch_robomind_failure(data_root: str | Path, workers: int = 4) -> dict[str, int]:
    """把 failure_data/ 下每条 trajectory.hdf5 的关节部分存到 data_root/robomind_failure/，写 manifest.json。
    已经有的跳过，可以断点续下。返回各状态的条数。"""
    from huggingface_hub import HfApi

    out = Path(data_root) / "robomind_failure"
    out.mkdir(parents=True, exist_ok=True)
    rows = [(e.path, e.size) for e in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo=PREFIX.rstrip("/"),
                                                             recursive=True)
            if getattr(e, "size", None) is not None and e.path.endswith(".hdf5")]
    manifest = out / "manifest.json"
    done = {r["path"]: r for r in json.loads(manifest.read_text(encoding="utf-8"))} if manifest.exists() else {}
    todo = [(p, s) for p, s in rows if done.get(p, {}).get("status") not in ("好", "已有")]
    print(f"共 {len(rows)} 条，要下 {len(todo)} 条", flush=True)
    t0, read = time.time(), 0
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_fetch_one, p, s, str(out)) for p, s in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done[r["path"]] = r
            read += r.get("read_bytes", 0)
            if r["status"] not in ("好", "已有"):
                print(f"  {r['status']} {r['path']}：{r.get('why')}", flush=True)
            if i % 50 == 0 or i == len(futs):
                print(f"{i}/{len(futs)}，读了 {read / 2**30:.2f} GB，{time.time() - t0:.0f} 秒", flush=True)
                manifest.write_text(json.dumps(list(done.values()), ensure_ascii=False), encoding="utf-8")
    stat: dict[str, int] = {}
    for r in done.values():
        stat[r["status"]] = stat.get(r["status"], 0) + 1
    return stat
