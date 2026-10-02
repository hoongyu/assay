"""按示范计的检出率、定位率、原样示范上的报出率，都带分母和 95% Wilson 区间。"""

from __future__ import annotations

import math

from jianjin.checks import CheckResult
from jianjin.eval.inject import Defect


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def score_kind(
    kind: str, labels: list[Defect], injected: list[CheckResult], clean: list[CheckResult]
) -> dict:
    cause = labels[0].cause if labels else None
    detected = localized = 0
    for d, r in zip(labels, injected, strict=True):
        hits = [f for f in r.findings if f.cause == d.cause]
        detected += bool(hits)
        localized += any(overlaps((f.t0, f.t1), (d.t0, d.t1)) for f in hits)
    flagged = sum(any(f.cause == cause for f in r.findings) for r in clean)
    n, m = len(labels), len(clean)
    return {
        "kind": kind,
        "cause": cause.value if cause else None,
        "n_injected": n,
        "detected": detected,
        "localized": localized,
        "n_clean": m,
        "clean_flagged": flagged,
        "detect_rate": detected / n if n else math.nan,
        "detect_ci": wilson(detected, n),
        "localize_rate": localized / n if n else math.nan,
        "localize_ci": wilson(localized, n),
        "clean_flag_rate": flagged / m if m else math.nan,
        "clean_flag_ci": wilson(flagged, m),
    }
