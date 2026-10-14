"""Per-user settings measured at onboarding (Python twin of Runtime/Onboarding.cs)."""

from __future__ import annotations

import numpy as np

REST_MARGIN, REST_MIN, REST_MAX = 0.25, -1.9, -1.2


def rest_height(resting_wrist_y, fallback: float = -1.5) -> float:
    """Median resting wrist height + 0.25, clamped to [-1.9, -1.2] shoulder widths; fallback with < 10 samples."""
    y = np.asarray(resting_wrist_y, float)
    if y.size < 10:
        return fallback
    return float(np.clip(np.median(y) + REST_MARGIN, REST_MIN, REST_MAX))
