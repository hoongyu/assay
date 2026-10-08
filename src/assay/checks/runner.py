"""按顺序跑全部关节规则，把查不了的记进 not_checked。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from assay.checks.batch import empty_recording, end_pose, extra_motion, jerky, regrasp
from assay.checks.config import CheckConfig, NotApplicable, reason
from assay.checks.integrity import missing_values, timestamp_gaps, too_short, video_length
from assay.checks.motion import long_pause
from assay.types import Episode, Finding

EpisodeCheck = Callable[[Episode, CheckConfig], list[Finding]]
BatchCheck = Callable[..., tuple[dict[str, list[Finding]], dict[str, str]]]

EPISODE_CHECKS: dict[str, EpisodeCheck] = {
    "missing_values": missing_values,
    "too_short": too_short,
    "timestamp_gaps": timestamp_gaps,
    "video_length": video_length,
    "long_pause": long_pause,
}

BATCH_CHECKS: dict[str, BatchCheck] = {
    "extra_motion": extra_motion,
    "end_pose": end_pose,
    "empty_recording": empty_recording,
    "regrasp": regrasp,
    "jerky": jerky,
}


@dataclass
class CheckResult:
    episode_id: str
    findings: list[Finding] = field(default_factory=list)
    not_checked: dict[str, str] = field(default_factory=dict)


def run_checks(
    episodes: list[Episode],
    cfg: CheckConfig | None = None,
    reference: list[Episode] | None = None,
) -> list[CheckResult]:
    """reference：同任务比较时拿来当参照的示范，默认就是这一批自己。"""
    cfg = cfg or CheckConfig()
    results = {ep.id: CheckResult(ep.id) for ep in episodes}
    for ep in episodes:
        r = results[ep.id]
        for name, fn in EPISODE_CHECKS.items():
            try:
                r.findings.extend(fn(ep, cfg))
            except NotApplicable as exc:
                r.not_checked[name] = reason(exc)
    for name, fn in BATCH_CHECKS.items():
        found, skipped = fn(episodes, cfg, reference=reference)
        for eid, fs in found.items():
            results[eid].findings.extend(fs)
        for eid, why in skipped.items():
            results[eid].not_checked[name] = why
    return [results[ep.id] for ep in episodes]
