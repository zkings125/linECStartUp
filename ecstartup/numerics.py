"""Small numerical compatibility helpers shared across physics modules."""

from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, dx: float = 1.0, axis: int = -1):
    """Integrate with the NumPy 2 API while retaining NumPy 1.x support."""

    implementation = getattr(np, "trapezoid", None)
    if implementation is None:
        implementation = np.trapz
    return implementation(y, x=x, dx=dx, axis=axis)
