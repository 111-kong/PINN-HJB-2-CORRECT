# -*- coding: utf-8 -*-
"""
失效检测（可接张力）、推力幅值与速率限制；可选与 PINN 输出串联。

滚动时域 MPC / 二次规划（QP）可作为外层安全壳：将 NN 输出作为参考力，
在推力与变化率约束下做一步重投影；此处未实现数值 QP，仅保留串联接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MooringFailureDetector:
    """张力阈值或外部布尔；可与 AQWA 张力读数对接。"""

    tension_threshold_n: Optional[float] = None
    _latched_failed: bool = field(default=False, repr=False)

    def reset(self) -> None:
        self._latched_failed = False

    def update(
        self,
        time_s: float,
        mooring_failure_time_s: Optional[float] = None,
        line_tensions_n: Optional[Sequence[float]] = None,
    ) -> bool:
        """
        返回 True 表示判定系泊失效（之后可切换价值网络/代理）。
        """
        if self._latched_failed:
            return True
        if mooring_failure_time_s is not None and time_s >= mooring_failure_time_s:
            self._latched_failed = True
            return True
        if self.tension_threshold_n is not None and line_tensions_n is not None:
            if len(line_tensions_n) == 0:
                return False
            if min(line_tensions_n) < self.tension_threshold_n:
                self._latched_failed = True
                return True
        return False


@dataclass
class ThrustSlewLimiter:
    """每步最大推力变化（N / N·m per dt）。"""

    max_delta_surge: float = 5e5
    max_delta_sway: float = 5e5
    max_delta_yaw: float = 5e7
    _prev: Optional[np.ndarray] = field(default=None, repr=False)

    def reset(self) -> None:
        self._prev = None

    def apply(self, u_target: np.ndarray, dt: float) -> np.ndarray:
        u = np.asarray(u_target, dtype=np.float64).reshape(3)
        if self._prev is None:
            self._prev = u.copy()
            return u
        max_d = np.array([self.max_delta_surge, self.max_delta_sway, self.max_delta_yaw], dtype=np.float64)
        delta_max = max_d * max(dt, 1e-6)
        du = u - self._prev
        du_clipped = np.clip(du, -delta_max, delta_max)
        u_out = self._prev + du_clipped
        self._prev = u_out.copy()
        return u_out


def clip_thrust(
    u: np.ndarray,
    clip_ranges: Sequence[Tuple[float, float]],
) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    lo = np.array([r[0] for r in clip_ranges], dtype=np.float64)
    hi = np.array([r[1] for r in clip_ranges], dtype=np.float64)
    return np.clip(u, lo, hi)


def apply_safety_pipeline(
    u_raw: np.ndarray,
    clip_ranges: Sequence[Tuple[float, float]],
    slew_limiter: Optional[ThrustSlewLimiter] = None,
    dt: float = 0.1,
) -> np.ndarray:
    u = np.asarray(u_raw, dtype=np.float64).reshape(3)
    if slew_limiter is not None:
        u = slew_limiter.apply(u, dt)
    return clip_thrust(u, clip_ranges)
