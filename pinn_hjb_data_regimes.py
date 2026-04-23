# -*- coding: utf-8 -*-
"""
分工况状态采样与 AQWA 数据列约定（x_k, u_k, x_{k+1}, regime）。
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pinn_hjb_config import MooringRegime, ScenarioSpec, AQWA_SCENARIOS

# AQWA「Parameters vs Time」导出：前 8 行为说明，第 9 行为列名，第 10 行起为数据
AQWA_PARAMETERS_VS_TIME_SKIP_ROWS = 8

DEFAULT_FAILED_STATE_BOUNDS: List[Tuple[float, float]] = [
    (-80.0, 80.0),
    (-80.0, 80.0),
    (-0.5, 0.5),
    (-3.0, 3.0),
    (-3.0, 3.0),
    (-0.2, 0.2),
]

DEFAULT_CONTROL_BOUNDS_SATURATED: List[Tuple[float, float]] = [
    (0.0, 8.0e6),
    (-8.0e6, 8.0e6),
    (-1.0e9, 2.0e9),
]


def aqwa_transition_csv_header(include_regime: bool = True) -> List[str]:
    base = [
        "time_s",
        "eta_surge_m",
        "eta_sway_m",
        "eta_yaw_rad",
        "nu_surge_ms",
        "nu_sway_ms",
        "nu_yaw_rads",
        "u_surge_N",
        "u_sway_N",
        "u_yaw_Nm",
        "eta_surge_next_m",
        "eta_sway_next_m",
        "eta_yaw_next_rad",
        "nu_surge_next_ms",
        "nu_sway_next_ms",
        "nu_yaw_next_rads",
    ]
    if include_regime:
        base.append("mooring_regime")
    return base


def aqwa_online_log_csv_header() -> List[str]:
    """
    与 `pinn_hjb_aqwa_online.initialize_log` 写入的表头一致（单时刻轨迹，无 x_{k+1} 列）。
    """
    return [
        "Time(s)",
        "η_surge(m)",
        "η_sway(m)",
        "η_yaw(rad)",
        "ν_surge(m/s)",
        "ν_sway(m/s)",
        "ν_yaw(rad/s)",
        "u_surge(N)",
        "u_sway(N)",
        "u_yaw(N·m)",
    ]


def aqwa_online_log_state_control_columns() -> Tuple[List[str], List[str]]:
    """返回 (6 维状态列名, 3 维控制列名)，与在线日志 CSV 一致。"""
    h = aqwa_online_log_csv_header()
    return h[1:7], h[7:10]


def scenario_by_id(sid: str) -> ScenarioSpec:
    for s in AQWA_SCENARIOS:
        if s.id == sid:
            return s
    raise KeyError(f"未知工况 id={sid}，可选: {[s.id for s in AQWA_SCENARIOS]}")


def latin_hypercube(bounds: Sequence[Tuple[float, float]], n: int) -> np.ndarray:
    bounds = list(bounds)
    d = len(bounds)
    samples = np.zeros((n, d))
    for di in range(d):
        intervals = np.linspace(0.0, 1.0, n + 1)
        samples[:, di] = np.random.uniform(intervals[:-1], intervals[1:])
    for di in range(d):
        np.random.shuffle(samples[:, di])
    out = np.zeros_like(samples)
    for di, (lo, hi) in enumerate(bounds):
        out[:, di] = lo + (hi - lo) * samples[:, di]
    return out


def sample_states_for_regime(
    regime: MooringRegime,
    n: int,
    intact_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    failed_bounds: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    if regime == MooringRegime.FAILED:
        b = failed_bounds if failed_bounds is not None else DEFAULT_FAILED_STATE_BOUNDS
    else:
        if intact_bounds is None:
            raise ValueError("完好系泊采样需提供 intact_bounds（如 scaler mean±3σ）")
        b = intact_bounds
    return latin_hypercube(b, n)


def mix_regime_samples(
    intact_x: np.ndarray,
    failed_x: np.ndarray,
    intact_weight: float = 0.5,
    n_total: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    intact_x = np.asarray(intact_x, dtype=np.float64)
    failed_x = np.asarray(failed_x, dtype=np.float64)
    ni, nf = intact_x.shape[0], failed_x.shape[0]
    if ni == 0 and nf == 0:
        return np.zeros((0, 6)), np.zeros((0,), dtype=np.int64)

    wi = max(0.0, min(1.0, intact_weight))
    if n_total is None:
        n_total = max(ni, nf, ni + nf)
    n_i = int(round(wi * n_total)) if ni > 0 else 0
    n_f = n_total - n_i if nf > 0 else 0
    if ni == 0:
        n_i, n_f = 0, n_total
    if nf == 0:
        n_i, n_f = n_total, 0

    parts: List[np.ndarray] = []
    regs: List[np.ndarray] = []
    if n_i > 0 and ni > 0:
        idx_i = np.random.choice(ni, size=n_i, replace=ni < n_i)
        parts.append(intact_x[idx_i])
        regs.append(np.full(n_i, int(MooringRegime.INTACT), dtype=np.int64))
    if n_f > 0 and nf > 0:
        idx_f = np.random.choice(nf, size=n_f, replace=nf < n_f)
        parts.append(failed_x[idx_f])
        regs.append(np.full(n_f, int(MooringRegime.FAILED), dtype=np.int64))
    if not parts:
        return np.zeros((0, 6)), np.zeros((0,), dtype=np.int64)
    x_mixed = np.vstack(parts)
    regime_ids = np.concatenate(regs)
    perm = np.random.permutation(len(x_mixed))
    return x_mixed[perm], regime_ids[perm]


def attach_regime_for_conditional_model(x6: np.ndarray, regime_ids: np.ndarray) -> np.ndarray:
    x6 = np.asarray(x6, dtype=np.float64)
    regime_ids = np.asarray(regime_ids, dtype=np.float64).reshape(-1, 1)
    if x6.ndim == 1:
        x6 = x6.reshape(1, -1)
    if x6.shape[0] != regime_ids.shape[0]:
        raise ValueError("x6 与 regime_ids 行数不一致")
    return np.hstack([x6, regime_ids.astype(np.float64)])


def describe_collection_plan() -> Dict[str, str]:
    return {s.id: s.description for s in AQWA_SCENARIOS}


_CSV_ENCODINGS_TRY: Tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936")


def _read_text_lines(path: str) -> Tuple[List[str], str]:
    with open(path, "rb") as bf:
        raw = bf.read()
    last_err: Optional[UnicodeDecodeError] = None
    for enc in _CSV_ENCODINGS_TRY:
        try:
            text = raw.decode(enc)
            return text.splitlines(), enc
        except UnicodeDecodeError as e:
            last_err = e
    # 混合编码/个别非法字节：替换后仍能解析数值列
    text = raw.decode("utf-8", errors="replace")
    return text.splitlines(), "utf-8-replace"


def load_aqwa_parameters_vs_time_states(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 AQWA 导出的「Parameters vs Time」CSV（位移/速度，角度为 ° 与 °/s）。

    列约定（与 AQWA Line A–F 一致，水平面内与控制器状态对齐）:
      - Line A: Global X → η_surge (m)
      - Line B: Global Y → η_sway (m)
      - Line C: Global RZ → η_yaw (rad)，由度转弧度
      - Line D/E: Global X/Y 速度 → ν_surge, ν_sway (m/s)
      - Line F: Global RZ 角速度 → ν_yaw (rad/s)，由 °/s 转 rad/s

    返回:
      times: (N,) 时间 (s)
      states: (N, 6) 按 [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw]
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    lines, _enc = _read_text_lines(path)
    if len(lines) <= AQWA_PARAMETERS_VS_TIME_SKIP_ROWS:
        raise ValueError(f"CSV 行数不足: {path}")

    data_lines = lines[AQWA_PARAMETERS_VS_TIME_SKIP_ROWS + 1 :]
    times_list: List[float] = []
    states_list: List[List[float]] = []

    for line in data_lines:
        if not line.strip():
            continue
        row = next(csv.reader([line]))
        if len(row) < 7:
            continue
        try:
            t = float(row[0])
            a, b, c, d, e, f = (float(row[i]) for i in range(1, 7))
        except ValueError:
            continue
        times_list.append(t)
        states_list.append(
            [
                a,
                b,
                math.radians(c),
                d,
                e,
                math.radians(f),
            ]
        )

    if not times_list:
        raise ValueError(f"未解析到有效数据行: {path}")

    times = np.asarray(times_list, dtype=np.float64)
    states = np.asarray(states_list, dtype=np.float64)
    return times, states


def build_state_samples_from_aqwa_export_csvs(
    path_intact: str,
    path_failed: str,
    *,
    failed_t_start: float = 2096.3,
    intact_t_start: Optional[float] = None,
    stride: int = 1,
    intact_weight: float = 0.5,
    n_total: int = 10000,
    seed: Optional[int] = None,
    lhs_frac: float = 0.0,
    intact_bounds_lhs: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    从两份 AQWA「Parameters vs Time」导出构造 PINN 配点 (N, 6)。

    - 失效文件：默认只取 Time >= failed_t_start（与控制启动对齐，默认 2096.3 s）。
    - 完好文件：可选 Time >= intact_t_start（None 表示整段轨迹）。
    - stride: 每隔 stride 行取一点。
    - lhs_frac: (0,1) 时混入拉丁超立方配点，需 intact_bounds_lhs。
    """
    if seed is not None:
        np.random.seed(seed)
    if stride < 1:
        raise ValueError("stride 须 >= 1")
    if not (0.0 <= lhs_frac < 1.0):
        raise ValueError("lhs_frac 须在 [0, 1)")

    t_i, x_i = load_aqwa_parameters_vs_time_states(path_intact)
    t_f, x_f = load_aqwa_parameters_vs_time_states(path_failed)

    if intact_t_start is not None:
        m_i = t_i >= intact_t_start
        t_i, x_i = t_i[m_i], x_i[m_i]
    m_f = t_f >= failed_t_start
    t_f, x_f = t_f[m_f], x_f[m_f]

    if x_i.shape[0] == 0:
        raise ValueError("完好 CSV 在时间与阈值筛选后无有效行")
    if x_f.shape[0] == 0:
        raise ValueError(
            f"失效 CSV 在 Time>={failed_t_start} 后无有效行，请检查时刻或文件"
        )

    if stride > 1:
        x_i = x_i[::stride]
        x_f = x_f[::stride]

    n_lhs = int(round(lhs_frac * n_total)) if lhs_frac > 0 else 0
    n_csv = n_total - n_lhs

    parts: List[np.ndarray] = []
    if n_csv > 0:
        x_csv, _ = mix_regime_samples(
            x_i, x_f, intact_weight=intact_weight, n_total=n_csv
        )
        parts.append(x_csv)

    if n_lhs > 0:
        if intact_bounds_lhs is None:
            raise ValueError("lhs_frac>0 时必须提供 intact_bounds_lhs（如 scaler mean±3σ）")
        ni = max(1, int(round(intact_weight * n_lhs)))
        nf = max(1, n_lhs - ni)
        intact_lhs = sample_states_for_regime(
            MooringRegime.INTACT, ni, intact_bounds=intact_bounds_lhs
        )
        failed_lhs = sample_states_for_regime(MooringRegime.FAILED, nf)
        x_lhs, _ = mix_regime_samples(
            intact_lhs, failed_lhs, intact_weight=intact_weight, n_total=n_lhs
        )
        parts.append(x_lhs)

    if not parts:
        raise ValueError("未生成任何配点，请检查 n_total / lhs_frac")

    out = np.vstack(parts)
    perm = np.random.permutation(len(out))
    return out[perm]


def default_aqwa_export_csv_paths() -> Tuple[str, str]:
    """默认千年一遇导出路径（可用环境变量覆盖）。"""
    root = os.environ.get("PINN_HJB_AQWA_DATA_DIR", r"f:\aqwa数据")
    intact = os.environ.get(
        "PINN_HJB_AQWA_INTACT_CSV",
        os.path.join(root, "千年一遇未失效位移加速度.csv"),
    )
    failed = os.environ.get(
        "PINN_HJB_AQWA_FAILED_CSV",
        os.path.join(root, "千年一遇失效位移加速度.csv"),
    )
    return intact, failed
