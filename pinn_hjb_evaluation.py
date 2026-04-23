# -*- coding: utf-8 -*-
"""
闭环记录指标：水平偏移、RMS、推力能量；OOD z-score；消融矩阵模板。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def horizontal_rms(x_hist: np.ndarray) -> float:
    """x_hist: (T,6) -> sqrt(mean(η_s^2+η_y^2))"""
    x_hist = np.asarray(x_hist, dtype=np.float64)
    if x_hist.size == 0:
        return 0.0
    p = x_hist[:, :2]
    return float(np.sqrt(np.mean(np.sum(p * p, axis=1))))


def max_horizontal_offset(x_hist: np.ndarray) -> float:
    x_hist = np.asarray(x_hist, dtype=np.float64)
    if x_hist.size == 0:
        return 0.0
    d = np.sqrt(x_hist[:, 0] ** 2 + x_hist[:, 1] ** 2)
    return float(np.max(d))


def mechanical_power_mean(
    u_hist: np.ndarray,
    vel_hist: np.ndarray,
    t: Optional[np.ndarray] = None,
) -> float:
    """
    近似机械功率 u·ν（surge/sway 线速度，yaw 用角速度），时间平均。
    u_hist: (T,3), vel_hist: (T,6) 取 nu_surge, nu_sway, nu_yaw。
    """
    u_hist = np.asarray(u_hist, dtype=np.float64)
    vel_hist = np.asarray(vel_hist, dtype=np.float64)
    if u_hist.shape[0] == 0:
        return 0.0
    v = vel_hist[:, [3, 4, 5]]
    p = np.sum(u_hist * v, axis=1)
    return float(np.mean(p))


def summarize_closed_loop(history: Dict[str, Any]) -> Dict[str, float]:
    """history 含 'time','x','u' 为 numpy 数组（与 PINNHJBClosedLoop 一致）。"""
    x = np.asarray(history.get("x", []))
    u = np.asarray(history.get("u", []))
    t = np.asarray(history.get("time", []))
    out: Dict[str, float] = {}
    if x.size == 0:
        return out
    out["rms_horizontal_m"] = horizontal_rms(x)
    out["max_horizontal_offset_m"] = max_horizontal_offset(x)
    if u.size and x.shape[0] == u.shape[0]:
        out["mean_mech_power_W"] = mechanical_power_mean(u, x, t)
    if t.size:
        out["duration_s"] = float(t[-1] - t[0])
    return out


def ood_max_abs_zscore(
    x6: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    每个样本各状态分量最大 |z|，用于代理训练包络外推诊断。
    x6: (N,6), x_mean/x_std: (6,)
    """
    x6 = np.asarray(x6, dtype=np.float64)
    m = np.asarray(x_mean, dtype=np.float64).reshape(1, -1)
    s = np.asarray(x_std, dtype=np.float64).reshape(1, -1)
    s = np.where(np.abs(s) < eps, eps, s)
    z = np.abs((x6 - m) / s)
    return np.max(z, axis=1)


ABLATION_EXPERIMENT_ROWS: List[Dict[str, str]] = [
    {"id": "A0", "surrogate": "single_6dof", "value": "single", "hjb": "paper_eq15", "note": "基线"},
    {"id": "A1", "surrogate": "conditional_7dof", "value": "single_7dof", "hjb": "paper_eq15", "note": "含 regime 输入"},
    {"id": "A2", "surrogate": "dual_checkpoint", "value": "per_regime", "hjb": "paper_eq15", "note": "分模态两套权重"},
    {"id": "A3", "surrogate": "single_6dof", "value": "single", "hjb": "mismatched_legacy", "note": "对照旧 HJB 残差（不应再用）"},
]
