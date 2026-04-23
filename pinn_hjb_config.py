# -*- coding: utf-8 -*-
"""
半潜平台系泊失效后推进器控制 — 集中配置（Q、R、dt、约束、工况枚举）。

在线脚本与 PINNTrainer 应从此模块读取 r_diag / clip，与 HJB 残差及控制律保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


def _default_u_physics_scale() -> Tuple[float, float, float]:
    """
    PINNController 已将代理的 G 从「归一化 Δx / 归一化 u」换算为 ∂Δx_phys/∂u_phys，
    控制律输出的 u 即为牛顿/牛米量级；此处默认不再额外放大。
    若需经验微调，可在 PINNHJBConfig 中显式覆盖 u_physics_scale。
    """
    return (1.0, 1.0, 1.0)


class MooringRegime(IntEnum):
    INTACT = 0
    FAILED = 1


@dataclass(frozen=True)
class ControlBounds:
    # 与 surrogate 训练 CSV 一致：历史 u_surge 均值为正（约 MN），故保持单向非负。
    # 若 AQWA 最优律需负 surge，应改为对称界并与代理重训数据对齐。
    surge: Tuple[float, float] = (0.0, 8.0e6)
    sway: Tuple[float, float] = (-8.0e6, 8.0e6)
    yaw: Tuple[float, float] = (-1.0e9, 2.0e9)


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    description: str
    regime: MooringRegime


AQWA_SCENARIOS: Tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "INTACT_RANDOM_THRUST",
        "完好系泊 + 随机推力激励（采集代理主数据）",
        MooringRegime.INTACT,
    ),
    ScenarioSpec(
        "FAILED_SINGLE_LINE",
        "单根系泊失效（指定断缆时刻）",
        MooringRegime.FAILED,
    ),
    ScenarioSpec(
        "FAILED_THRUSTER_ONLY",
        "失效后仅靠推进器限制漂移（长时段）",
        MooringRegime.FAILED,
    ),
)


@dataclass
class PINNHJBConfig:
    dt: float = 0.1
    state_dim: int = 6
    control_dim: int = 3
    q_diag: Tuple[float, ...] = (
        0.01,
        0.01,
        0.02,
        0.001,
        0.001,
        0.005,
    )
    # HJB 中 term2 ∝ g^2/r：r 过小会导致残差爆炸、训练无法收敛。故 r 保持可训练量级。
    r_diag: Tuple[float, float, float] = (1e-2, 1e-2, 1e-1)
    # 施加到 AQWA / 写入日志的物理推力 = clip( u_hjb * u_physics_scale )；
    # u_hjb 为 -0.5 R^{-1} G^T∇J（G 已按 scaler 换到物理 u）；默认 scale=1，可用 calibrate_r_for_u_scale.py 调 R 或微调 scale。
    u_physics_scale: Tuple[float, float, float] = field(default_factory=_default_u_physics_scale)
    control_bounds: ControlBounds = field(default_factory=ControlBounds)
    control_start_time_s: float = 2096.3
    use_regime_in_state: bool = False
    extended_state_dim: int = 7
    # 在线 pinn_hjb_aqwa_online：PINN 仅在 t ≥ 本时刻才加载并运行（与失效判据一致）。
    # 若本值严格大于 control_start_time_s，则 [施控启动, 失效) 内 user_force 直接返回空白力、
    # 不写 CSV、不加载 torch，以减少 Python 侧与磁盘开销；AQWA 侧仍按耦合设置交换状态。
    mooring_failure_time_s: Optional[float] = 2096.3
    """AQWA 断缆/失效时刻；在线在此刻起才初始化并调用 PINN；None 则与 control_start_time_s 相同。"""

    def clip_ranges(self) -> List[Tuple[float, float]]:
        b = self.control_bounds
        return [b.surge, b.sway, b.yaw]

    def validate(self) -> None:
        assert len(self.q_diag) == self.state_dim
        assert len(self.r_diag) == self.control_dim
        for r in self.r_diag:
            assert r > 0
        assert len(self.u_physics_scale) == self.control_dim
        for s in self.u_physics_scale:
            assert s > 0

    def effective_state_dim(self) -> int:
        return self.extended_state_dim if self.use_regime_in_state else self.state_dim

    def q_diag_for_network(self) -> Tuple[float, ...]:
        if not self.use_regime_in_state:
            return self.q_diag
        return tuple(self.q_diag) + (1e-6,)


DEFAULT_CONFIG = PINNHJBConfig()
DEFAULT_CONFIG.validate()


def append_regime_to_states(x: np.ndarray, regime: MooringRegime) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    col = np.full((x.shape[0], 1), float(regime), dtype=np.float64)
    return np.hstack([x, col])


def strip_regime_from_states(x_ext: np.ndarray) -> np.ndarray:
    x_ext = np.asarray(x_ext, dtype=np.float64)
    return x_ext[:, :6]
