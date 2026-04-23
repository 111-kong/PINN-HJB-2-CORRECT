# -*- coding: utf-8 -*-
"""
PINN-HJB × AQWA 在线控制集成
==============================
将训练好的PINN-HJB控制器集成到AQWA仿真中，
实现实时最优控制（类似UF5的结构）

核心功能：
1. 读取AQWA实时状态 (位置、速度)
2. 使用PINN-HJB计算最优控制力
3. 输出控制力到AQWA
4. 记录仿真过程

PINN 仅在 t ≥ 失效时刻 `MOORING_FAILURE_TIME_S`（环境变量优先）才加载并运行。
此前 user_force 直接返回空白力（不组装状态、不调 PINN、不写 CSV），以降低 Python
与磁盘负载；AQWA 仍按 User Force 耦合收发状态，该部分由 AQWA 侧设置决定。
若 `mooring_failure_time_s` 严格大于 `control_start_time_s`，则 [施控启动, 失效) 为上述空窗。

Author: Buddy (Proactive Agent)
Date: 2026-04-10
"""

import math
import numpy as np
import torch
import json
import csv
import os
import time
from typing import Optional
from AqwaServerMgr import *
from pinn_hjb_controller import PINNController, value_network_from_state_dict
from pinn_hjb_aqwa_integration import AQWASimulator
from pinn_hjb_config import DEFAULT_CONFIG, MooringRegime
from pinn_hjb_safety import MooringFailureDetector, ThrustSlewLimiter, apply_safety_pipeline

# ============================================================
# 全局配置和状态变量
# ============================================================

# PINN-HJB模型配置（默认同本文件所在 Analysis 目录；可用 PINN_HJB_ANALYSIS_DIR 覆盖）
BASE_DIR = os.environ.get(
    "PINN_HJB_ANALYSIS_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
SURROGATE_MODEL_PATH = os.path.join(BASE_DIR, "surrogate_model", "surrogate_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "surrogate_model", "scaler.json")
VALUE_NETWORK_PATH = os.path.join(BASE_DIR, "surrogate_model", "value_network.pt")
VALUE_NETWORK_FAILED_PATH = os.path.join(
    BASE_DIR, "surrogate_model", "value_network_failed.pt"
)

# 控制参数（与 pinn_hjb_config 一致）
CONTROL_START_TIME = float(DEFAULT_CONFIG.control_start_time_s)
DT = float(DEFAULT_CONFIG.dt)

# 系泊失效时刻（秒）：优先环境变量，其次 DEFAULT_CONFIG；均未指定时退化为 CONTROL_START_TIME
#（进入施控窗口即视为已达失效时刻，与旧版「一进施控就算 PINN」一致）
_moor_t = os.environ.get("MOORING_FAILURE_TIME_S", "").strip()
if _moor_t:
    MOORING_FAILURE_TIME_S = float(_moor_t)
elif DEFAULT_CONFIG.mooring_failure_time_s is not None:
    MOORING_FAILURE_TIME_S = float(DEFAULT_CONFIG.mooring_failure_time_s)
else:
    MOORING_FAILURE_TIME_S = float(CONTROL_START_TIME)

# t ≥ 该时刻才加载模型并执行 PINN（与 MooringFailureDetector 判失效时刻一致）
PINN_ENABLE_TIME_S = float(MOORING_FAILURE_TIME_S)
# 施控已开但尚未到 PINN 启用：空窗（仅当失效时刻严格晚于施控启动）
_PINN_IDLE_BEFORE_ENABLE = PINN_ENABLE_TIME_S > CONTROL_START_TIME + 1e-9

# 控制器状态
pinn_controller = None  # 兼容旧调试代码：带 .controller / .simulator 的轻量对象
controller_intact = None
controller_failed = None
value_net = None
value_net_failed = None
simulator = None
mooring_detector = MooringFailureDetector()
thrust_slew_limiter = ThrustSlewLimiter()
control_initialized = False
model_loaded = False

# 日志记录
log_file_path = None
control_log_data = []
# 已写入 CSV 的最后仿真时刻（严格落在 DT 网格上）；None 表示本日志尚未写过行
last_logged_grid_time: Optional[float] = None

# 性能统计
control_stats = {
    'steps': 0,
    'last_state': None,
    'last_control': None,
    'state_history': [],
    'control_history': []
}

# ============================================================
# PINN-HJB 初始化函数
# ============================================================

def initialize_pinn_controller():
    """
    初始化PINN-HJB控制器
    加载训练好的模型和价值网络
    """
    global pinn_controller, controller_intact, controller_failed
    global value_net, value_net_failed, simulator, model_loaded, control_initialized

    if model_loaded:
        print("[OK] PINN-HJB模型已加载")
        return True

    try:
        print("\n" + "=" * 60)
        print("  [BRAIN] 初始化PINN-HJB控制器")
        print("=" * 60)

        # 检查文件
        print("\n[FILE] 检查模型文件...")
        files = {
            "代理模型": SURROGATE_MODEL_PATH,
            "标准化参数": SCALER_PATH,
            "价值网络": VALUE_NETWORK_PATH
        }

        for name, path in files.items():
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"  [OK] {name}: {os.path.basename(path)} ({size_kb:.1f} KB)")
            else:
                print(f"  [ERROR] {name}: 不存在")
                return False

        # 加载代理模型和标准化参数
        print("\n[ROBOT] 加载代理模型...")
        simulator = AQWASimulator(
            model_path=SURROGATE_MODEL_PATH,
            scaler_path=SCALER_PATH,
            device='cpu'
        )

        # 加载价值网络
        print("\n[BRAIN] 加载价值网络...")
        checkpoint = torch.load(VALUE_NETWORK_PATH, map_location="cpu")
        vsd = checkpoint["value_state_dict"]
        value_net = value_network_from_state_dict(vsd)
        value_net.load_state_dict(vsd)
        value_net.eval()

        loss_history = checkpoint.get('loss_history', [])
        if loss_history:
            print(f"  [OK] 价值网络加载成功")
            print(f"     训练Loss范围: {min(loss_history):.6f} - {max(loss_history):.6f}")
            print(f"     最终Loss: {loss_history[-1]:.6f}")

        print("\n[LOOP] 初始化 PINNController（R 与 DEFAULT_CONFIG 一致）...")
        # PINNController 内部：x/dx 为 6 维状态，u 为 3 维控制（与 scaler.json / surrogate 一致）
        def _to_np1d(t, n: int, name: str) -> np.ndarray:
            a = np.asarray(t.detach().cpu().numpy().reshape(-1), dtype=np.float64)
            if a.size != n:
                raise ValueError(f"{name} 长度 {a.size}，期望 {n}")
            return a

        xm = _to_np1d(simulator.x_mean, 6, "x_mean")
        xs = _to_np1d(simulator.x_std, 6, "x_std")
        us = _to_np1d(simulator.u_std, 3, "u_std")
        ds = _to_np1d(simulator.dx_std, 6, "dx_std")

        controller_intact = PINNController(
            surrogate_model=simulator.model,
            value_network=value_net,
            pinn_cfg=DEFAULT_CONFIG,
            device="cpu",
            surrogate_x_mean=xm,
            surrogate_x_std=xs,
            surrogate_u_std=us,
            surrogate_dx_std=ds,
        )

        value_net_failed = None
        controller_failed = controller_intact
        if os.path.exists(VALUE_NETWORK_FAILED_PATH):
            print(f"\n[BRAIN] 加载系泊失效专用价值网络: {VALUE_NETWORK_FAILED_PATH}")
            ckf = torch.load(VALUE_NETWORK_FAILED_PATH, map_location="cpu")
            vsdf = ckf["value_state_dict"]
            value_net_failed = value_network_from_state_dict(vsdf)
            value_net_failed.load_state_dict(vsdf)
            value_net_failed.eval()
            controller_failed = PINNController(
                surrogate_model=simulator.model,
                value_network=value_net_failed,
                pinn_cfg=DEFAULT_CONFIG,
                device="cpu",
                surrogate_x_mean=xm,
                surrogate_x_std=xs,
                surrogate_u_std=us,
                surrogate_dx_std=ds,
            )
        else:
            print("\n[INFO] 未找到 value_network_failed.pt，失效后与完好共用同一价值网络。")

        class _LoopShim:
            pass

        shim = _LoopShim()
        shim.controller = controller_intact
        shim.simulator = simulator
        pinn_controller = shim

        mooring_detector.reset()
        thrust_slew_limiter.reset()

        model_loaded = True
        control_initialized = True

        if _PINN_IDLE_BEFORE_ENABLE:
            print(
                f"\n[INFO] PINN 仅在 t ≥ {PINN_ENABLE_TIME_S:g}s 运行；"
                f"[{CONTROL_START_TIME:g}, {PINN_ENABLE_TIME_S:g}) 内不加载本脚本模型、不写控制 CSV。"
            )

        print("\n[OK] PINN-HJB控制器初始化完成！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n[ERROR] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# 控制计算函数
# ============================================================

def compute_optimal_control(time_step, pos, vel):
    """
    计算最优控制力

    参数:
      time_step: 当前时间步
      pos: 位置状态 [η_surge, η_sway, η_yaw]
      vel: 速度状态 [ν_surge, ν_sway, ν_yaw]

    返回:
      control_force: [u_surge, u_sway, u_yaw]
    """
    global pinn_controller, controller_intact, controller_failed
    global mooring_detector, thrust_slew_limiter, control_stats

    if not model_loaded or controller_intact is None:
        print("[WARN] 控制器未初始化，返回0控制力")
        return [0.0, 0.0, 0.0]

    try:
        # 组合状态向量 (6维)
        x = np.array([
            pos[0] if len(pos) > 0 else 0.0,  # η_surge
            pos[1] if len(pos) > 1 else 0.0,  # η_sway
            pos[2] if len(pos) > 2 else 0.0,  # η_yaw
            vel[0] if len(vel) > 0 else 0.0,  # ν_surge
            vel[1] if len(vel) > 1 else 0.0,  # ν_sway
            vel[2] if len(vel) > 2 else 0.0   # ν_yaw
        ], dtype=np.float64)

        failed = mooring_detector.update(
            float(time_step),
            mooring_failure_time_s=MOORING_FAILURE_TIME_S,
        )

        pin = controller_failed if failed else controller_intact
        mooring_regime = float(MooringRegime.FAILED if failed else MooringRegime.INTACT)

        # 调试：打印状态
        if control_stats['steps'] % 500 == 0:
            print(f"[DEBUG] 状态: {x} | mooring_failed={failed}")

        clips = DEFAULT_CONFIG.clip_ranges()
        u_raw = pin.compute_control(
            x,
            clip_ranges=clips,
            mooring_regime=mooring_regime,
        )
        control = apply_safety_pipeline(
            u_raw,
            clip_ranges=clips,
            slew_limiter=thrust_slew_limiter,
            dt=DT,
        )

        # #region agent log
        _h4n = getattr(compute_optimal_control, "_bce714_h4", 0)
        if _h4n < 30:
            setattr(compute_optimal_control, "_bce714_h4", _h4n + 1)
            try:
                import json
                import os
                import time as _t

                _lp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-bce714.log")
                with open(_lp, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        json.dumps(
                            {
                                "sessionId": "bce714",
                                "hypothesisId": "H4_safety_pipeline",
                                "location": "pinn_hjb_aqwa_online.py:compute_optimal_control",
                                "message": "u_raw vs control after apply_safety_pipeline",
                                "timestamp": int(_t.time() * 1000),
                                "data": {
                                    "n": _h4n,
                                    "u_raw": np.asarray(u_raw, dtype=np.float64).reshape(-1).tolist(),
                                    "control_out": np.asarray(control, dtype=np.float64).reshape(-1).tolist(),
                                    "dt": float(DT),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

        # 调试：检查控制力是否为0
        for i, c in enumerate(control):
            if abs(c) < 1e-6:
                dim_names = ['surge', 'sway', 'yaw']
                print(f"[DEBUG] {dim_names[i]}控制力为0: u_{dim_names[i]}={c:.2e}, 状态: {x}")
                # 进一步检查内部计算
                try:
                    from pinn_hjb_controller import ConditionalSurrogateModel

                    xa = np.asarray(x, dtype=np.float64).reshape(-1)
                    x_phys = torch.tensor(xa[:6], dtype=torch.float32, device=pin.device).unsqueeze(0)
                    if isinstance(pin.surrogate_model, ConditionalSurrogateModel):
                        if xa.size < 7:
                            reg = torch.tensor(
                                [[float(mooring_regime)]],
                                dtype=torch.float32,
                                device=pin.device,
                            )
                            x_val = torch.cat([x_phys, reg], dim=1)
                        else:
                            x_val = torch.tensor(xa[:7], dtype=torch.float32, device=pin.device).unsqueeze(0)
                    else:
                        x_val = x_phys
                    x_val.requires_grad_(True)
                    dJ_dx = pin.value_network.compute_gradient(x_val)
                    print(f"[DEBUG] dJ_dx: {dJ_dx.detach().cpu().numpy()}")
                    print(f"[DEBUG] R_inv: {pin.R_inv.detach().cpu().numpy()}")
                except Exception as debug_e:
                    print(f"[DEBUG] 内部检查失败: {debug_e}")

        # 统计信息
        control_stats['steps'] += 1
        control_stats['last_state'] = x.copy()
        control_stats['last_control'] = control.copy()

        if len(control_stats['state_history']) >= 1000:
            control_stats['state_history'].pop(0)
            control_stats['control_history'].pop(0)
        control_stats['state_history'].append(x.copy())
        control_stats['control_history'].append(control.copy())

        return control.tolist()

    except Exception as e:
        print(f"[WARN] 控制计算出错: {e}")
        import traceback
        traceback.print_exc()
        return [0.0, 0.0, 0.0]


# ============================================================
# 日志记录函数
# ============================================================

def initialize_log():
    """
    初始化日志文件
    """
    global log_file_path, last_logged_grid_time

    log_file_path = os.path.join(BASE_DIR, "pinn_hjb_aqwa_online_log.csv")
    last_logged_grid_time = None

    try:
        with open(log_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Time(s)',
                'η_surge(m)', 'η_sway(m)', 'η_yaw(rad)',
                'ν_surge(m/s)', 'ν_sway(m/s)', 'ν_yaw(rad/s)',
                'u_surge(N)', 'u_sway(N)', 'u_yaw(N·m)'
            ])
        print(f"[OK] 日志文件创建: {log_file_path}")
    except Exception as e:
        print(f"[ERROR] 创建日志文件失败: {e}")


def log_control_data(time_step, pos, vel, control):
    """
    按仿真时间严格以 DT 为间隔写入 CSV。

    AQWA 回调间隔通常不等于 DT；本函数在每次回调时，把 (last_grid, Time] 内
    所有未写的 DT 网格点一次性补写。网格时刻之间缺失的中间状态/控制无法从
    回调获知，故对中间行使用本次回调的 pos/vel/control（零阶保持），与力在
    步内常值假设一致。若 AQWA 内部步长已等于 DT，则每格一行且与瞬时量一致。
    """
    global log_file_path, last_logged_grid_time

    if log_file_path is None:
        return

    dt = DT
    eps = 1e-9
    try:
        t_now = float(time_step)
        if last_logged_grid_time is None:
            # 锚到不大于当前时刻的最近网格点之下，使首行时间为 floor(t_now/dt)*dt
            last_logged_grid_time = math.floor(t_now / dt + eps) * dt - dt

        rows = []
        t_next = last_logged_grid_time + dt
        while t_next <= t_now + eps:
            rows.append(t_next)
            t_next += dt

        if not rows:
            return

        with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for tg in rows:
                writer.writerow([
                    f"{tg:.1f}",
                    f"{pos[0]:.4f}" if len(pos) > 0 else "0",
                    f"{pos[1]:.4f}" if len(pos) > 1 else "0",
                    f"{pos[2]:.6f}" if len(pos) > 2 else "0",
                    f"{vel[0]:.4f}" if len(vel) > 0 else "0",
                    f"{vel[1]:.4f}" if len(vel) > 1 else "0",
                    f"{vel[2]:.6f}" if len(vel) > 2 else "0",
                    f"{control[0]:.2e}",
                    f"{control[1]:.2e}",
                    f"{control[2]:.2e}",
                ])
        last_logged_grid_time = rows[-1]
    except Exception as e:
        print(f"[WARN] 日志记录失败: {e}")


# ============================================================
# AQWA回调函数 (与UF5类似的结构)
# ============================================================

def user_force(Analysis, Mode, Stage, Time, TimeStep, Pos, Vel):
    """
    AQWA用户自定义力函数 (socket接口回调)

    参数:
      Analysis: AQWA分析对象
      Mode:     模式
      Stage:    阶段
      Time:     当前时间
      TimeStep: 时间步长
      Pos:      位置 (从AQWA读取)
      Vel:      速度 (从AQWA读取)

    返回:
      Force:    控制力 (BlankForce对象)
      AddMass:  附加质量 (BlankAddedMass对象)
      ErrorFlag: 错误标志 (0=成功)
    """
    global control_initialized, log_file_path

    # 初始化Force和AddMass对象
    Force = BlankForce(Analysis.NOfStruct)
    AddMass = BlankAddedMass(Analysis.NOfStruct)
    ErrorFlag = 0

    try:
        # 阶段1: 控制启动前 (Time < CONTROL_START_TIME)
        if Time < CONTROL_START_TIME:
            # 未到 AQWA 施控窗口：空白力；不预加载 PINN（避免失效前加载 torch/代理）
            return Force, AddMass, ErrorFlag

        # 施控窗口内、但未到 PINN 启用（失效）时刻：不加载模型、不写日志、不调 compute_optimal_control
        if Time < PINN_ENABLE_TIME_S:
            return Force, AddMass, ErrorFlag

        # t ≥ PINN_ENABLE_TIME_S：首次在此加载并运行 PINN
        if not control_initialized:
            if not initialize_pinn_controller():
                # 初始化失败时不写日志，避免在模型未就绪时刷屏；AQWA 仍收到零力
                return Force, AddMass, ErrorFlag
            initialize_log()
            print(f"[GAME] {Time:.1f}s: 已达失效/启用时刻 {PINN_ENABLE_TIME_S:g}s，PINN-HJB 启动")

        # 提取状态 (假设NOfStruct=1)
        pos = [Pos[0][0], Pos[0][1], Pos[0][5]] if len(Pos) > 0 and len(Pos[0]) >= 3 else [0, 0, 0]
        vel = [Vel[0][0], Vel[0][1], Vel[0][5]] if len(Vel) > 0 and len(Vel[0]) >= 3 else [0, 0, 0]

        # 计算控制力
        control = compute_optimal_control(Time, pos, vel)

        # 应用控制力 (仅控制surge, sway, yaw DOF)
        Force[0][0] = control[0]  # surge力 (DOF 0)
        Force[0][1] = control[1]  # sway力 (DOF 1)
        Force[0][5] = control[2]  # yaw力矩 (DOF 2)

        # 记录
        if control_stats['steps'] % 100 == 0:
            print(f"[TIMER]  Time: {Time:.1f}s | "
                  f"位置: η={pos[0]:+.2f}m, η={pos[1]:+.2f}m, η={pos[2]:+.4f}rad | "
                  f"控制: u={control[0]:.2e}N, u={control[1]:.2e}N, τ={control[2]:.2e}N·m")

        log_control_data(Time, pos, vel, control)

        return Force, AddMass, ErrorFlag

    except Exception as e:
        print(f"[ERROR] user_force异常: {e}")
        import traceback
        traceback.print_exc()
        ErrorFlag = 1
        return Force, AddMass, ErrorFlag


# ============================================================
# 统计和报告函数
# ============================================================

def print_control_statistics():
    """
    打印控制统计信息
    """
    global control_stats

    from pinn_hjb_evaluation import horizontal_rms, max_horizontal_offset

    print("\n" + "=" * 60)
    print("  [STATS] PINN-HJB控制统计")
    print("=" * 60)

    print(f"\n控制步数: {control_stats['steps']}")

    if len(control_stats['state_history']) > 0:
        state_array = np.array(control_stats['state_history'])
        control_array = np.array(control_stats['control_history'])

        print(f"\n最后状态:")
        print(f"  η_surge: {control_stats['last_state'][0]:+.4f} m")
        print(f"  η_sway:  {control_stats['last_state'][1]:+.4f} m")
        print(f"  η_yaw:   {control_stats['last_state'][2]:+.6f} rad")
        print(f"  ν_surge: {control_stats['last_state'][3]:+.4f} m/s")
        print(f"  ν_sway:  {control_stats['last_state'][4]:+.4f} m/s")
        print(f"  ν_yaw:   {control_stats['last_state'][5]:+.6f} rad/s")

        print(f"\n平均控制力:")
        print(f"  u_surge: {control_array[:, 0].mean():.2e} N (范围: {control_array[:, 0].min():.2e} ~ {control_array[:, 0].max():.2e})")
        print(f"  u_sway:  {control_array[:, 1].mean():.2e} N (范围: {control_array[:, 1].min():.2e} ~ {control_array[:, 1].max():.2e})")
        print(f"  u_yaw:   {control_array[:, 2].mean():.2e} N·m (范围: {control_array[:, 2].min():.2e} ~ {control_array[:, 2].max():.2e})")

        print(f"\n水平面指标（窗口内）:")
        print(f"  RMS(√(η_s^2+η_y^2)): {horizontal_rms(state_array):.4f} m")
        print(f"  最大水平偏移: {max_horizontal_offset(state_array):.4f} m")

    print(f"\n日志文件: {log_file_path}")
    print("=" * 60)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  [START] PINN-HJB × AQWA 在线控制集成")
    print("=" * 60)

    print("\n[INFO] 使用说明:")
    print("  1. 启动AQWA服务器")
    print("  2. 配置AQWA以使用此脚本的 user_force 函数")
    print("  3. 运行AQWA仿真")
    print(f"  4. 仿真在 Time = {CONTROL_START_TIME}s 进入施控窗口（可由 pinn_hjb_config 修改）")
    print("  5. 控制数据会记录到 pinn_hjb_aqwa_online_log.csv")
    print(
        "  6. 失效/启用时刻 MOORING_FAILURE_TIME_S 或 config.mooring_failure_time_s："
        "此前不加载 PINN、不写 CSV；达到该时刻后再初始化并计算控制力"
    )
    print("  环境变量: PINN_HJB_ANALYSIS_DIR, MOORING_FAILURE_TIME_S（秒，优先于 config）")

    print("\n[CONFIG]  配置信息:")
    print(f"  代理模型: {os.path.basename(SURROGATE_MODEL_PATH)}")
    print(f"  价值网络: {os.path.basename(VALUE_NETWORK_PATH)}")
    print(f"  失效价值网络(可选): {os.path.basename(VALUE_NETWORK_FAILED_PATH)}")
    print(f"  R_diag: {DEFAULT_CONFIG.r_diag}")
    print(f"  控制启动时间: {CONTROL_START_TIME} s")
    print(f"  系泊失效时刻(用于施控/切换): {MOORING_FAILURE_TIME_S} s")
    _idle_msg = (
        f"是 [{CONTROL_START_TIME:g}, {PINN_ENABLE_TIME_S:g}) s"
        if _PINN_IDLE_BEFORE_ENABLE
        else "否（与施控同时启用）"
    )
    print(f"  失效前 PINN 空窗: {_idle_msg}")
    print(f"  时间步长: {DT} s")
    Server = AqwaUserForceServer()
    print('现在使用PINN-HJB控制器进行在线控制')
    print('开始模拟...')
    Server.Run(user_force)

    print("\n" + "=" * 60)

