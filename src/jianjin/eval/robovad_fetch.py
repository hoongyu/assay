"""从 Zenodo 上的 RoboVAD.zip 里按需取文件。压缩包 6.86 GB，关节数据只有 23 MB，用 HTTP Range 只读要的部分。

RoboVAD 授权是 CC BY-NC-ND 4.0：只拿来评测，不改编、不再分发数据。
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

ROBOVAD_URL = "https://zenodo.org/records/22754659/files/RoboVAD.zip?download=1"
ROOT = "RoboVAD/"
SMALL_PREFIXES = (
    "RoboVAD/README.md",
    "RoboVAD/license.txt",
    "RoboVAD/annotations.json",
    "RoboVAD/splits/",
    "RoboVAD/evaluation/",
    "RoboVAD/proprioception/",
)
VIDEO_PREFIX = "RoboVAD/videos/"


class HttpRangeFile(io.RawIOBase):
    """把远程文件当成可 seek 的本地文件读，每次 read 发一个 Range 请求。"""

    def __init__(self, url: str):
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.url = head.url  # 跟随重定向后的真实地址
        self.size = int(head.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(buf), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        buf[: len(data)] = data
        self.pos += len(data)
        return len(data)


def extract(z: zipfile.ZipFile, dest: Path, prefixes: tuple[str, ...]) -> int:
    """解出以 prefixes 开头的成员（去掉 RoboVAD/ 前缀）；本地已有且大小相同的跳过。返回新解出的个数。"""
    n = 0
    for info in z.infolist():
        name = info.filename
        if name.endswith("/") or name.startswith("__MACOSX") or name.endswith(".DS_Store"):
            continue
        if not name.startswith(prefixes):
            continue
        target = Path(dest) / name[len(ROOT):]
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(name))
        n += 1
    return n


def fetch_robovad(data_root: str | Path, videos: bool = False) -> Path:
    dest = Path(data_root) / "robovad"
    prefixes = SMALL_PREFIXES + ((VIDEO_PREFIX,) if videos else ())
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ROBOVAD_URL), buffer_size=1 << 20))
    extract(z, dest, prefixes)
    return dest
