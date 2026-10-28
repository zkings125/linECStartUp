"""无外部副作用的时间波形插值。"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def validate_points(points: Sequence[Tuple[float, float]]) -> list[tuple[float, float]]:
    normalized = [(float(t), float(v)) for t, v in points]
    if any(not (math.isfinite(t) and math.isfinite(v)) for t, v in normalized):
        raise ValueError("波形时间和值必须为有限数")
    if any(normalized[i + 1][0] <= normalized[i][0] for i in range(len(normalized) - 1)):
        raise ValueError("波形时间点必须严格递增")
    return normalized


def linear_value(points: Sequence[Tuple[float, float]] | None, t_s: float, default: float) -> float:
    """分段线性插值；区间外保持端点值。"""

    if not points:
        return float(default)
    pts = validate_points(points)
    t = float(t_s)
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            return v0 + alpha * (v1 - v0)
    return pts[-1][1]
