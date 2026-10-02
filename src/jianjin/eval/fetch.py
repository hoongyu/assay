"""按考卷配置下载公开数据集，只要 meta 和 data（关节数据），不下视频。"""

from __future__ import annotations

from pathlib import Path

from jianjin.eval.exam_a import DatasetSpec, local_dir


def fetch(specs: list[DatasetSpec], data_root: str | Path, with_videos: bool = False) -> list[Path]:
    from huggingface_hub import snapshot_download  # 可选依赖：uv sync --extra hub

    patterns = ["meta/*", "data/*"] + (["videos/*"] if with_videos else [])
    out = []
    for s in specs:
        target = local_dir(data_root, s.repo)
        snapshot_download(repo_id=s.repo, repo_type="dataset", local_dir=target, allow_patterns=patterns)
        out.append(target)
    return out
