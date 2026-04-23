# -*- coding: utf-8 -*-
"""
PINN-HJB × AQWA 集成层
=======================
将训练好的 AQWA 代理模型 (F_φ + G_φ) 整合进 PINN-HJB 控制器，
实现基于真实数据的闭环仿真与在线控制。
1
Author: Buddy (Proactive Agent)
Date: 2026-04-08
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os
from typing import Optional

from pinn_hjb_config import DEFAULT_CONFIG, MooringRegime
from pinn_hjb_data_regimes import (
    build_state_samples_from_aqwa_export_csvs,
    default_aqwa_export_csv_paths,
    mix_regime_samples,
    sample_states_for_regime,
)
from pinn_hjb_evaluation import summarize_closed_loop

# 与 surrogate_trainer 一致：默认同本文件所在 Analysis 目录；可用 PINN_HJB_ANALYSIS_DIR 覆盖
_ANALYSIS_DIR = os.environ.get(
    "PINN_HJB_ANALYSIS_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

# ============================================================
# 第一节: AQWA 仿真器 (基于训练好的代理模型)
# ============================================================

class AQWASimulator:
    """
    基于 AQWA 实测数据训练的代理模型，实现平台运动仿真。

    功能:
      1. 加载训练好的 SurrogateModel 权重
      2. 加载标准化参数 (scaler.json)
      3. 给定当前状态 x_k 和控制输入 u_k，预测下一状态 x_{k+1}

    状态方程:
      x_{k+1} = x_k + Δt * (F_φ(x_k) + G_φ(x_k) · u_k)

    其中:
      x = [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw]  (6维)
      u = [u_surge, u_sway, u_yaw]                          (3维)

    使用方法:
      sim = AQWASimulator(
          model_path="surrogate_model/surrogate_model.pt",
          scaler_path="surrogate_model/scaler.json"
      )
      x_next = sim.step(x_current, u_current, dt=0.1)
    """

    def __init__(self, model_path, scaler_path, device=None):
        """
        初始化 AQWA 仿真器

        参数:
          model_path:   surrogate_model.pt 文件路径
          scaler_path:  scaler.json 文件路径
          device:       计算设备 (默认 auto)
        """
        self.device = device or torch.device("cpu")

        # ---- 加载标准化参数 ----
        with open(scaler_path, 'r', encoding='utf-8') as f:
            s = json.load(f)
        self.x_mean = torch.tensor(s['x_mean'], dtype=torch.float32)
        self.x_std  = torch.tensor(s['x_std'],  dtype=torch.float32)
        self.u_mean = torch.tensor(s['u_mean'], dtype=torch.float32)
        self.u_std  = torch.tensor(s['u_std'],  dtype=torch.float32)
        self.dx_mean = torch.tensor(s['dx_mean'], dtype=torch.float32)
        self.dx_std  = torch.tensor(s['dx_std'],  dtype=torch.float32)

        # ---- 加载 SurrogateModel ----
        # 从 pinn_hjb_controller 导入 (同一文件内使用时直接引用)
        from pinn_hjb_controller import SurrogateModel
        self.model = SurrogateModel().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        # 从 checkpoint 读取训练信息
        self.train_info = ckpt.get('info', {})

        print(f"[AQWASimulator] 初始化完成")
        print(f"  模型: {model_path}")
        print(f"  设备: {self.device}")
        print(f"  训练信息: {self.train_info}")

    # ----------------------------------------------------------
    def predict_delta(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        预测状态增量 Δx = F_φ(x) + G_φ(x)·u

        参数:
          x: 当前状态 (6,) 或 (batch, 6)
          u: 控制输入 (3,) 或 (batch, 3)

        返回:
          dx: 状态增量 (6,) 或 (batch, 6)，已反归一化
        """
        x_t = torch.FloatTensor(x).to(self.device)
        u_t = torch.FloatTensor(u).to(self.device)

        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            u_t = u_t.unsqueeze(0)

        with torch.no_grad():
            x_n = (x_t - self.x_mean) / self.x_std
            u_n = (u_t - self.u_mean) / self.u_std
            delta_norm = self.model.predict_delta(x_n, u_n)
            dx = delta_norm * self.dx_std + self.dx_mean

        result = dx.cpu().numpy()
        return result[0] if x.ndim == 1 else result

    # ----------------------------------------------------------
    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        仿真一步: x_{k+1} = x_k + Δx，Δx = F_φ(x_k) + G_φ(x_k)·u_k

        注意:
          predict_delta 已输出每步的状态增量（m 或 m/s），
          已经是 0.1s 时间步内的变化，无需再乘 dt。

        参数:
          x: 当前状态 (6,)
          u: 控制输入 (3,)

        返回:
          x_next: 下一状态 (6,)
        """
        dx = self.predict_delta(x, u)
        x_next = x + dx
        return x_next

    # ----------------------------------------------------------
    def get_dynamics(self, x: np.ndarray) -> tuple:
        """
        获取当前状态的内在动力学 F_φ(x) 和控制效应矩阵 G_φ(x)

        返回:
          F_phi: (6,) 内在动力学 (已反归一化)
          G_phi: (6, 3) 控制效应矩阵 (已反归一化)
        """
        x_t = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        x_n = (x_t - self.x_mean) / self.x_std

        with torch.no_grad():
            F_n, G_n_flat = self.model(x_n)
            G_n = G_n_flat.view(6, 3)
            F_phi = F_n * self.dx_std
            G_phi = G_n * self.dx_std.unsqueeze(-1)

        return F_phi.cpu().numpy()[0], G_phi.cpu().numpy()


# ============================================================
# 第二节: PINN-HJB 闭环仿真器
# ============================================================

class PINNHJBClosedLoop:
    """
    PINN-HJB 控制器 + AQWA 代理模型 完整闭环仿真器2

    仿真流程:
      for each timestep k:
          1. 获取当前状态 x_k
          2. 计算最优控制 u_k = -1/2 R^{-1} G_φ(x_k)^T ∇J_θ(x_k)
          3. 代理模型预测 x_{k+1} = x_k + dt * (F_φ(x_k) + G_φ(x_k)·u_k)
          4. 记录数据

    集成方式:
      - PINNController: 从 pinn_hjb_controller 导入
      - AQWASimulator:   基于 AQWA 实测数据训练的代理模型
    """

    def __init__(
        self,
        surrogate_path: str,
        scaler_path: str,
        value_net,
        R_diag=None,
        device=None
    ):
        """
        初始化闭环仿真器

        参数:
          surrogate_path: surrogate_model.pt 路径
          scaler_path:   scaler.json 路径
          value_net:      训练好的 ValueNetwork (PINN-HJB 价值函数)
          R_diag:         控制权重对角线，默认 [1e-4, 1e-4, 1e-2]
          device:         计算设备
        """
        from pinn_hjb_controller import PINNController

        self.device = device or torch.device("cpu")
        self.dt = float(DEFAULT_CONFIG.dt)

        # AQWA 代理仿真器
        self.simulator = AQWASimulator(surrogate_path, scaler_path, self.device)

        if R_diag is None:
            R_diag = list(DEFAULT_CONFIG.r_diag)

        def _to_np1d(t, n: int, name: str) -> np.ndarray:
            a = np.asarray(t.detach().cpu().numpy().reshape(-1), dtype=np.float64)
            if a.size != n:
                raise ValueError(f"{name} length {a.size}, expected {n}")
            return a

        xm = _to_np1d(self.simulator.x_mean, 6, "x_mean")
        xs = _to_np1d(self.simulator.x_std, 6, "x_std")
        us = _to_np1d(self.simulator.u_std, 3, "u_std")
        ds = _to_np1d(self.simulator.dx_std, 6, "dx_std")

        # PINN-HJB 控制器（R 与 HJB 训练、配置文件一致）
        self.controller = PINNController(
            surrogate_model=self.simulator.model,
            value_network=value_net,
            R_diag=R_diag,
            device=self.device,
            pinn_cfg=DEFAULT_CONFIG,
            surrogate_x_mean=xm,
            surrogate_x_std=xs,
            surrogate_u_std=us,
            surrogate_dx_std=ds,
        )

        # 控制器需要共享仿真器的模型参数（共享同一个 SurrogateModel）
        # PINNController 在初始化时接收的是 AQWASimulator.model (已加载权重)
        # 所以 G_φ 来自 AQWA 实测数据，F_φ 也来自 AQWA 实测数据

        # 仿真记录
        self.history = {
            'time': [],
            'x': [],
            'u': [],
            'F_phi': [],
            'G_phi_diag': []
        }

        print(f"[PINNHJBClosedLoop] 初始化完成")
        print(f"  代理模型: {surrogate_path}")
        print(f"  控制权重 R: {R_diag}")
        print(f"  仿真步长 dt: {self.dt}s")

    # ----------------------------------------------------------
    def reset(self):
        """清空仿真记录"""
        self.history = {k: [] for k in self.history}

    # ----------------------------------------------------------
    def simulate(
        self,
        x0: np.ndarray,
        duration: float,
        reference=None,
        clip_ranges=None,
        verbose=True
    ) -> dict:
        """
        运行闭环仿真

        参数:
          x0:          初始状态 (6,)
          duration:    仿真时长 (秒)
          reference:   参考轨迹 (可选, 格式同 x0，若为 None 则镇定到0)
          clip_ranges: 控制输出限制 [(min,max), (min,max), (min,max)]
          verbose:     是否打印进度

        返回:
          history: 仿真记录 (time, x, u, F_phi, G_phi_diag)
        """
        from pinn_hjb_controller import PINNController

        self.reset()
        n_steps = int(duration / self.dt)

        if verbose:
            print(f"\n[闭环仿真] 初始状态: {x0}")
            print(f"  仿真时长: {duration}s, 步数: {n_steps}")

        x_current = np.array(x0, dtype=np.float64)

        for k in range(n_steps):
            t = k * self.dt

            # ---- 1. 计算最优控制 ----
            if reference is not None:
                x_ref = np.array(reference) if hasattr(reference, '__len__') else reference
                x_error = x_current - x_ref
            else:
                x_error = x_current.copy()

            if clip_ranges is None:
                clip_ranges = DEFAULT_CONFIG.clip_ranges()
            u = self.controller.compute_control(x_error, clip_ranges=clip_ranges)

            # ---- 2. 代理模型预测下一步 ----
            x_next = self.simulator.step(x_current, u)

            # ---- 3. 获取动力学信息（用于记录）----2
            F_phi, G_phi = self.simulator.get_dynamics(x_current)
            G_diag = np.diag(G_phi)

            # ---- 4. 记录 ----
            self.history['time'].append(t)
            self.history['x'].append(x_current.copy())
            self.history['u'].append(u.copy())
            self.history['F_phi'].append(F_phi.copy())
            self.history['G_phi_diag'].append(G_diag.copy())

            # ---- 5. 状态更新 ----
            x_current = x_next

            # 打印进度
            if verbose and (k + 1) % 1000 == 0:
                print(f"  进度: {k+1}/{n_steps} 步 ({t:.1f}s) | "
                      f"η_surge={x_current[0]:+.3f}m | "
                      f"η_yaw={x_current[2]:+.4f}rad")

        if verbose:
            print(f"  仿真完成 | 最终状态: η_surge={x_current[0]:+.3f}m, "
                  f"η_sway={x_current[1]:+.3f}m, η_yaw={x_current[2]:+.4f}rad")

        # 转为 numpy 数组
        self.history['time'] = np.array(self.history['time'])
        self.history['x']    = np.array(self.history['x'])
        self.history['u']    = np.array(self.history['u'])
        self.history['F_phi'] = np.array(self.history['F_phi'])
        self.history['G_phi_diag'] = np.array(self.history['G_phi_diag'])

        summ = summarize_closed_loop(self.history)
        if verbose and summ:
            print(f"  [指标] {summ}")

        return self.history

    # ----------------------------------------------------------
    def save_log(self, filepath: str):
        """保存仿真记录到 CSV"""
        import csv
        names = ['η_surge', 'η_sway', 'η_yaw', 'ν_surge', 'ν_sway', 'ν_yaw']
        ctrl  = ['u_surge(N)', 'u_sway(N)', 'u_yaw(N·m)']

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Time(s)'] +
                       [f'{n}_actual' for n in names] +
                       [f'{c}_actual' for c in ctrl])
            for i in range(len(self.history['time'])):
                row = [self.history['time'][i]]
                row.extend(self.history['x'][i].tolist())
                row.extend(self.history['u'][i].tolist())
                w.writerow(row)

        print(f"[保存] 仿真记录已保存: {filepath}")


def build_mixed_training_state_samples(
    intact_bounds,
    n_intact: int = 5000,
    n_failed: int = 5000,
    intact_weight: float = 0.5,
    n_total: int = 10000,
    seed: Optional[int] = None,
):
    """
    混合「完好系泊状态云 + 系泊失效大偏移包络」用于 PINN 价值训练采样。
    intact_bounds: list of (lo,hi) 六维，通常取 scaler mean±3σ。
    """
    if seed is not None:
        np.random.seed(seed)
    intact_x = sample_states_for_regime(
        MooringRegime.INTACT, n_intact, intact_bounds=intact_bounds
    )
    failed_x = sample_states_for_regime(MooringRegime.FAILED, n_failed)
    x_mix, _reg = mix_regime_samples(
        intact_x, failed_x, intact_weight=intact_weight, n_total=n_total
    )
    return x_mix


# ============================================================
# 第三节: 快速使用入口
# ============================================================

def quick_demo():
    """
    快速演示: 加载模型 → 运行闭环仿真

    使用方法:
      from pinn_hjb_aqwa_integration import quick_demo
      history = quick_demo()
    """
    import matplotlib.pyplot as plt
    from pinn_hjb_controller import ValueNetwork, load_pinn_models

    print("=" * 60)
    print("  PINN-HJB × AQWA 闭环仿真演示")
    print("=" * 60)

    # ---- 路径配置 ----
    base = _ANALYSIS_DIR
    surrogate_path = os.path.join(base, "surrogate_model", "surrogate_model.pt")
    scaler_path    = os.path.join(base, "surrogate_model", "scaler.json")

    # ---- 加载价值网络 (如果已训练) ----
    # 若无训练好的价值网络，使用随机初始化的网络演示控制器响应
    value_net = ValueNetwork(state_dim=6, hidden_dim=1024)
    value_net.eval()

    # ---- 初始化闭环仿真器 ----
    closed_loop = PINNHJBClosedLoop(
        surrogate_path=surrogate_path,
        scaler_path=scaler_path,
        value_net=value_net,
        R_diag=None,
    )

    # ---- 初始状态 (偏离平衡位置的典型工况) ----
    x0 = np.array([
        -5.0,    # η_surge: 纵荡偏移 -5m
         3.0,    # η_sway:  横荡偏移 +3m
        -0.05,   # η_yaw:   艏摇角 -0.05rad
         0.0,    # ν_surge: 初始速度 0
         0.0,    # ν_sway:  初始速度 0
         0.0     # ν_yaw:   初始角速度 0
    ])

    # ---- 运行仿真 (150秒, 对应控制启动后真实数据段) ----
    history = closed_loop.simulate(
        x0=x0,
        duration=150.0,   # 仿真150秒
        clip_ranges=DEFAULT_CONFIG.clip_ranges(),
        verbose=True
    )

    # ---- 绘图 ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    t = history['time']

    # 位置
    axes[0].plot(t, history['x'][:, 0], label='η_surge (m)', color='tab:blue')
    axes[0].plot(t, history['x'][:, 1], label='η_sway (m)',  color='tab:orange')
    axes[0].plot(t, history['x'][:, 2], label='η_yaw (rad)', color='tab:green')
    axes[0].set_ylabel('Position / Angle')
    axes[0].legend(loc='upper right')
    axes[0].set_title('PINN-HJB 闭环仿真: 状态响应')
    axes[0].grid(True, alpha=0.3)

    # 速度
    axes[1].plot(t, history['x'][:, 3], label='ν_surge (m/s)', color='tab:blue')
    axes[1].plot(t, history['x'][:, 4], label='ν_sway (m/s)',  color='tab:orange')
    axes[1].plot(t, history['x'][:, 5], label='ν_yaw (rad/s)', color='tab:green')
    axes[1].set_ylabel('Velocity')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # 控制
    axes[2].plot(t, history['u'][:, 0] / 1e6, label='u_surge (MN)', color='tab:blue')
    axes[2].plot(t, history['u'][:, 1] / 1e6, label='u_sway (MN)',  color='tab:orange')
    axes[2].plot(t, history['u'][:, 2] / 1e6, label='u_yaw (MN·m)', color='tab:green')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Control (MN / MN·m)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(base, "surrogate_model", "closed_loop_simulation.png")
    fig.savefig(save_path, dpi=150)
    print(f"[绘图] 闭环仿真曲线已保存: {save_path}")

    return history


# ============================================================
# 第四节: 主程序 — 演示 + 价值网络训练入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PINN-HJB × AQWA 集成演示")
    print("=" * 60)
    print()
    print("  选项:")
    print("    1 - 快速演示闭环仿真 (使用随机价值网络)")
    print("    2 - 训练价值网络 (PINN-HJB 在线学习)")
    print()
    choice = input("请选择 [1/2]: ").strip()

    if choice == '1':
        # 演示模式: 加载代理模型 + 随机价值网络 → 闭环仿真
        quick_demo()

    elif choice == '2':
        # 训练模式: 使用 AQWA 实测数据的状态分布训练价值网络
        from pinn_hjb_controller import PINNTrainer, DataGenerator, SurrogateModel

        base = r"e:/ADRC control model/ADRC control_files/dp0/AQW-1/AQW/AQ/Analysis"
        surrogate_path = os.path.join(base, "surrogate_model", "surrogate_model.pt")
        scaler_path    = os.path.join(base, "surrogate_model", "scaler.json")

        # 加载代理模型
        sim = AQWASimulator(surrogate_path, scaler_path)

        # 使用 AQWA 实测数据的状态范围 (从 scaler 推断)
        x_mean = sim.x_mean.numpy()
        x_std  = sim.x_std.numpy()
        state_bounds = list(zip(x_mean - 3*x_std, x_mean + 3*x_std))

        print("\n[价值网络训练]")
        print("  状态范围 (3σ覆盖):")
        names = ['η_surge', 'η_sway', 'η_yaw', 'ν_surge', 'ν_sway', 'ν_yaw']
        for i, n in enumerate(names):
            print(f"    {n}: [{state_bounds[i][0]:+.3f}, {state_bounds[i][1]:+.3f}]")

        # 配点：优先 AQWA「Parameters vs Time」千年一遇 CSV；失效侧默认 Time>=control_start_time_s(2096.3s)
        path_intact, path_failed = default_aqwa_export_csv_paths()
        _t_ctrl = float(DEFAULT_CONFIG.control_start_time_s)
        _lhs_frac = float(os.environ.get("PINN_HJB_PINN_LHS_FRAC", "0"))
        if os.path.isfile(path_intact) and os.path.isfile(path_failed):
            print(f"\n[配点] 尝试 AQWA 导出 CSV（失效段 Time>={_t_ctrl:g} s）")
            try:
                state_samples = build_state_samples_from_aqwa_export_csvs(
                    path_intact,
                    path_failed,
                    failed_t_start=_t_ctrl,
                    intact_t_start=None,
                    stride=1,
                    intact_weight=0.5,
                    n_total=10000,
                    seed=42,
                    lhs_frac=_lhs_frac,
                    intact_bounds_lhs=state_bounds if _lhs_frac > 0 else None,
                )
                print(f"  完好: {path_intact}")
                print(f"  失效: {path_failed}")
                if _lhs_frac > 0:
                    print(f"  混入 LHS 比例: {_lhs_frac}（PINN_HJB_PINN_LHS_FRAC）")
                print(f"  配点数量: {state_samples.shape[0]}")
            except Exception as exc:
                print(f"  ⚠️ CSV 配点失败: {exc}")
                print("  回退: LHS 混合配点（3σ 盒 + 失效大盒）")
                state_samples = build_mixed_training_state_samples(
                    intact_bounds=state_bounds,
                    n_intact=5000,
                    n_failed=5000,
                    intact_weight=0.5,
                    n_total=10000,
                    seed=42,
                )
                print(f"  配点数量: {state_samples.shape[0]}")
        else:
            print("\n[配点] 未找到 AQWA 导出 CSV，使用 LHS 混合配点")
            print(f"  期望路径: {path_intact} / {path_failed}")
            state_samples = build_mixed_training_state_samples(
                intact_bounds=state_bounds,
                n_intact=5000,
                n_failed=5000,
                intact_weight=0.5,
                n_total=10000,
                seed=42,
            )
            print(f"  配点数量: {state_samples.shape[0]}")

        # HJB 残差与 DEFAULT_CONFIG.r_diag / q_diag 一致
        # 学习率 / 调度：较原 1e-4 略提高以加快收敛；StepLR 在 2000 epoch 内实际生效（原 5e4 步等价于不衰减）
        _lr = 3e-4
        _epochs = 2000
        _batch = 64
        # 与 surrogate_trainer / AQWASimulator 一致：代理在归一化状态上训练；价值网络对物理 x 做 (x-μ)/σ 再入 MLP，利于降低 HJB MSE
        _xm = sim.x_mean.detach().cpu().numpy()
        _xs = sim.x_std.detach().cpu().numpy()
        _um = sim.u_mean.detach().cpu().numpy()
        _us = sim.u_std.detach().cpu().numpy()
        _dxm = sim.dx_mean.detach().cpu().numpy()
        _dxs = sim.dx_std.detach().cpu().numpy()
        trainer = PINNTrainer(
            state_dim=6,
            control_dim=3,
            surrogate_hidden=128,
            value_hidden=1024,
            lr=_lr,
            decay_factor=0.5,
            decay_interval=800,
            device='cpu',
            pinn_cfg=DEFAULT_CONFIG,
            surrogate_input_mean=_xm,
            surrogate_input_std=_xs,
            surrogate_u_mean=_um,
            surrogate_u_std=_us,
            surrogate_dx_mean=_dxm,
            surrogate_dx_std=_dxs,
            value_input_mean=_xm,
            value_input_std=_xs,
        )

        # 将代理模型加载到训练器
        trainer.surrogate_model = sim.model
        trainer.surrogate_model.eval()  # 冻结代理模型参数

        print("\n[PINN价值网络训练] 参数配置:")
        print(f"  学习率: {_lr} (前 50 个 epoch 线性预热到此值)")
        print(f"  Batch Size: {_batch}")
        print(f"  Epochs: {_epochs}；StepLR 每 800 epoch 衰减 ×0.5")
        print(
            "  代理: 归一化 x 前向；HJB: F_eff、G_phys 使用 scaler.json 的 u/dx 标量（与在线 PINNController 一致）"
        )
        
        loss_history = trainer.train(
            state_samples,
            num_epochs=_epochs,
            batch_size=_batch,
            print_interval=100
        )

        # 保存
        save_path = os.path.join(base, "surrogate_model", "value_network.pt")
        torch.save({
            'value_state_dict': trainer.value_network.state_dict(),
            'loss_history': loss_history
        }, save_path)
        print(f"\n[保存] 价值网络已保存: {save_path}")

    else:
        print("无效选项")
        
