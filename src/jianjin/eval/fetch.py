"""按考卷配置下载公开数据集。默认只要 meta 和 data（关节数据）；要视频时只下配置里那一路相机。"""

from __future__ import annotations

from pathlib import Path

from jianjin.eval.exam_a import DatasetSpec, local_dir


def fetch(specs: list[DatasetSpec], data_root: str | Path, with_videos: bool = False) -> list[Path]:
    from huggingface_hub import snapshot_download  # 可选依赖：uv sync --extra hub

    out = []
    for s in specs:
        patterns = ["meta/*", "data/*"]
        if with_videos:
            patterns.append(f"videos/{s.camera}/*" if s.camera else "videos/*")
        target = local_dir(data_root, s.repo)
        snapshot_download(repo_id=s.repo, repo_type="dataset", local_dir=target, allow_patterns=patterns)
        out.append(target)
    return out
