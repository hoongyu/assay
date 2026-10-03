"""按考卷配置下载公开数据集。默认只要 meta 和 data（关节数据）；要视频时只下配置里写的那一路相机。"""

from __future__ import annotations

from pathlib import Path

from jianjin.eval.exam_a import DatasetSpec, local_dir


def patterns_for(s: DatasetSpec, with_videos: bool) -> list[str]:
    """要下哪些文件。视频只下配置里写的那一路相机；没写相机的（只用关节数据的双臂开发集）不下视频。"""
    patterns = ["meta/*", "data/*"]
    if with_videos and s.camera:
        patterns.append(f"videos/{s.camera}/*")
    return patterns


def fetch(specs: list[DatasetSpec], data_root: str | Path, with_videos: bool = False) -> list[Path]:
    from huggingface_hub import snapshot_download  # 可选依赖：uv sync --extra hub

    out = []
    for s in specs:
        target = local_dir(data_root, s.repo)
        snapshot_download(repo_id=s.repo, repo_type="dataset", local_dir=target,
                          allow_patterns=patterns_for(s, with_videos))
        out.append(target)
    return out
