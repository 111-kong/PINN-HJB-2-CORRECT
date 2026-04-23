# -*- coding: utf-8 -*-
"""
PINN-HJB 在线训练控制器 (三自由度版本)
基于论文: "Load mitigation in floating wind turbines via active tuned mass damper 
          using a physics-informed neural network based controller"
          (Ocean Engineering 351 (2026) 123914)

适配说明:
- 原论文: 6维状态 [x_tmd, ẋ_tmd, x_plfm, ẋ_plfm, x_tt, ẋ_tt] (风力机ATMD系统)
- 本适配: 深海一号半潜式平台三自由度控制
  - 状态: [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw] (6维)
  - 控制: [u_surge, u_sway, u_yaw] (3维)
  - DOF: 纵荡(X) + 横荡(Y) + 艏摇(ψ)

注意: 本代码是OFFLINE训练的PINN框架设计
      在线推理时直接使用训练好的价值网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import csv
import time
import os
from typing import Optional, Sequence, Tuple, Union

from pinn_hjb_config import DEFAULT_CONFIG, PINNHJBConfig

# ============================================================
# 第一节: 代理模型 (Surrogate Model) - 三自由度版本
# ============================================================
# 
# 三自由度系统方程:
#   x = [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw] (6维)
#   ẋ = F_φ(x) + G_φ(x) · u
#   
# 其中:
#   F_φ(x) ∈ R^6: 内在动力学 (环境力驱动)
#   G_φ(x) ∈ R^{6×3}: 控制效应矩阵
#   u ∈ R^3: 控制输入 [u_surge, u_sway, u_yaw]

class SurrogateModel(nn.Module):
    """
    三自由度代理模型
    
    输入: x = [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw] (6维)
    输出: 
        - F_φ(x): 内在动力学 (6维)
        - G_φ(x): 控制效应矩阵 (6×3 → 重塑为18维向量)
    
    论文原文架构:
    "Both subnetworks share the same input layer and employ a single hidden 
     layer with m=128 neurons and a tanh activation function."
    """
    
    def __init__(self, state_dim=6, control_dim=3, hidden_dim=128):
        super(SurrogateModel, self).__init__()
        self.state_dim = state_dim
        self.physics_dim = state_dim
        self.control_dim = control_dim
        
        # 共享输入层
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.Tanh()
        
        # F_φ 子网络 - 近似内在动力学 (6维 → 6维)
        # 对于位置自由度，ẋ = ν，所以F_φ的前3维应该接近速度
        self.F_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)  # 输出6维内在动力学
        )
        
        # G_φ 子网络 - 近似控制效应矩阵 (6维 → 6×3=18维)
        # 重塑为(batch, 6, 3)的控制效应矩阵
        self.G_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim * control_dim)  # 输出18维
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 系统状态 (batch_size, 6)
        
        返回:
        - F_phi: 内在动力学 (batch_size, 6)
        - G_phi: 控制效应矩阵 (batch_size, 18) → 可重塑为(batch, 6, 3)
        """
        # 共享隐藏层
        h = self.activation(self.input_layer(x))
        
        # F_φ: 内在动力学
        F_phi = self.F_net(h)  # (batch, 6)
        
        # G_φ: 控制效应矩阵 (展平)
        G_phi_flat = self.G_net(h)  # (batch, 18)
        
        return F_phi, G_phi_flat
    
    def get_G_matrix(self, G_phi_flat):
        """
        将展平的G_phi重塑为矩阵形式
        
        参数:
        - G_phi_flat: (batch, 18) 展平的控制效应
        
        返回:
        - G_matrix: (batch, 6, 3) 控制效应矩阵
        """
        batch_size = G_phi_flat.shape[0]
        G_matrix = G_phi_flat.view(batch_size, self.state_dim, self.control_dim)
        return G_matrix
    
    def predict_delta(self, x, u):
        """
        预测状态增量
        
        参数:
        - x: 当前状态 (batch, 6)
        - u: 控制输入 (batch, 3)
        
        返回:
        - delta: 状态增量 (batch, 6)
          Δx = F_φ(x) + G_φ(x) · u
        """
        F_phi, G_phi_flat = self.forward(x)
        G_matrix = self.get_G_matrix(G_phi_flat)  # (batch, 6, 3)
        
        # 矩阵乘法: (batch, 6, 3) @ (batch, 3, 1) → (batch, 6, 1)
        # 或 (batch, 6, 3) @ (batch, 3,) → (batch, 6)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        delta_dynamics = torch.bmm(G_matrix, u.unsqueeze(-1)).squeeze(-1)  # (batch, 6)
        
        delta = F_phi + delta_dynamics
        return delta


class ConditionalSurrogateModel(nn.Module):
    """
    条件代理：输入 x_ext = [η,ν 共6维, mooring_regime 标量]，输出 F∈R^6、G∈R^{6×3}。
    动力学增量仍只作用在物理 6 维：Δx_6 = F + G u。
    state_dim 属性为 7，供 ValueNetwork 与 PINNController 对扩展状态求 ∇J。
    """

    def __init__(
        self,
        physics_dim: int = 6,
        regime_dim: int = 1,
        hidden_dim: int = 128,
        control_dim: int = 3,
    ):
        super().__init__()
        self.physics_dim = physics_dim
        self.regime_dim = regime_dim
        self.control_dim = control_dim
        self.input_dim = physics_dim + regime_dim
        self.state_dim = self.input_dim

        self.input_layer = nn.Linear(self.input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.F_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, physics_dim),
        )
        self.G_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, physics_dim * control_dim),
        )

    def forward(self, x_ext: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.activation(self.input_layer(x_ext))
        return self.F_net(h), self.G_net(h)

    def get_G_matrix(self, G_phi_flat: torch.Tensor) -> torch.Tensor:
        b = G_phi_flat.shape[0]
        return G_phi_flat.view(b, self.physics_dim, self.control_dim)

    def predict_delta(self, x_ext: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        F_phi, G_flat = self.forward(x_ext)
        Gm = self.get_G_matrix(G_flat)
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        gu = torch.bmm(Gm, u.unsqueeze(-1)).squeeze(-1)
        return F_phi + gu


# ============================================================
# 第二节: 价值网络 (Value Network) - 对应论文的 J_θ(x)
# ============================================================
class ValueNetwork(nn.Module):
    """
    价值网络 - 对应论文的 J_θ(x)，用于近似最优值函数
    
    论文原文 (Eq.8):
    "J_θ(x) = k_J2 σ(k_J1 x + b_J1) + b_J2"
    
    架构: 输入层 → 1024隐藏神经元(Tanh) → 1输出(标量值函数)
    
    重要: 输出层的梯度 ∇J_θ(x) 用于计算最优控制律
    
    三自由度版本:
    - 输入: 6维状态 [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw]
    - 输出: 标量值函数 J_θ(x) ∈ R
    - 梯度: ∇J_θ(x) ∈ R^6
    """
    
    def __init__(
        self,
        state_dim=6,
        hidden_dim=1024,
        input_mean: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        input_std: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
    ):
        super(ValueNetwork, self).__init__()
        self.state_dim = state_dim
        self._use_input_norm = input_mean is not None and input_std is not None
        if self._use_input_norm:
            m = torch.as_tensor(np.asarray(input_mean, dtype=np.float64), dtype=torch.float32).reshape(
                1, state_dim
            )
            s = torch.as_tensor(np.asarray(input_std, dtype=np.float64), dtype=torch.float32).reshape(
                1, state_dim
            )
            s = torch.clamp(s, min=1e-8)
            self.register_buffer("input_mean", m)
            self.register_buffer("input_std", s)

        # 价值网络结构 (论文建议: 1024隐藏层神经元)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # ✅ 新增: Xavier初始化，确保初期梯度合理
        for layer in [self.fc1, self.fc2, self.fc3]:
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        
        # ✅ 新增: 输出层缩放，使初始价值函数接近0
        with torch.no_grad():
            self.fc3.weight.mul_(0.01)  # 缩小输出权重
            if self.fc3.bias is not None:
                self.fc3.bias.mul_(0.01)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 系统状态 (batch_size, 6)
        
        返回:
        - J: 标量值函数 (batch_size, 1)
        """
        z = x
        if self._use_input_norm:
            z = (x - self.input_mean) / self.input_std
        h = self.activation(self.fc1(z))
        h = self.activation(self.fc2(h))
        J = self.fc3(h)
        return J
    
    def compute_gradient(self, x):
        """
        计算值函数关于状态的梯度 ∇J_θ(x)
        这是PINN-HJB求解的关键步骤
        
        参数:
        - x: 系统状态 (batch_size, 6)
        
        返回:
        - dJ_dx: ∇J_θ(x) (batch_size, 6)
        """
        x.requires_grad_(True)
        J = self.forward(x)
        
        # 使用自动微分计算梯度
        dJ_dx = grad(
            outputs=J,
            inputs=x,
            grad_outputs=torch.ones_like(J),
            create_graph=True,
            retain_graph=True
        )[0]
        
        return dJ_dx


# ============================================================
# 第三节: HJB损失函数 (三自由度版本) - 对应论文 Eq.15
# ============================================================
class HJBLoss(nn.Module):
    """
    HJB 残差 — 与论文 Eq.(7)/(15) 及 PINNController 控制律一致（对角 R）。

    连续时间 HJB: 0 = x^T Q x + 1/4 (G^T∇J)^T R^{-1} (G^T∇J) + (∇J_{1:6})^T F
    其中 F∈R^6、G∈R^{6×3}；若状态维为 7（含 regime），仅前 6 维与 F、G 耦合。
    代理输出为每步增量 Δx≈F+Gu 时，仍将 (F,G) 视为该离散化下的等效漂移/输入矩阵。
    """

    def __init__(
        self,
        Q: Optional[torch.Tensor] = None,
        R_diag: Optional[Tuple[float, float, float]] = None,
        state_dim: int = 6,
        control_dim: int = 3,
        physics_dim: int = 6,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.physics_dim = physics_dim

        if Q is None:
            q_list = list(DEFAULT_CONFIG.q_diag)[:state_dim]
            while len(q_list) < state_dim:
                q_list.append(1e-6)
            q_list = q_list[:state_dim]
            Q = torch.diag(torch.tensor(q_list, dtype=torch.float32))
        if Q.dim() == 1:
            Q = torch.diag(Q)
        self.register_buffer("Q", Q)

        if R_diag is None:
            R_diag = tuple(float(x) for x in DEFAULT_CONFIG.r_diag)
        r = torch.tensor(R_diag, dtype=torch.float32)
        if torch.any(r <= 0):
            raise ValueError("R 对角元必须为正")
        self.register_buffer("R_diag", r)

    def compute_hjb_residual(
        self,
        x: torch.Tensor,
        F_phi: torch.Tensor,
        G_phi_flat: torch.Tensor,
        dJ_dx: torch.Tensor,
    ) -> torch.Tensor:
        x = x.view(-1, self.state_dim)
        dJ_dx = dJ_dx.view(-1, self.state_dim)
        F_phi = F_phi.view(-1, F_phi.shape[-1])
        batch_size = x.shape[0]

        # term1: x^T Q x（支持满秩 Q）
        term1 = torch.einsum("bi,ij,bj->b", x, self.Q, x)

        phys = min(self.physics_dim, F_phi.shape[1], dJ_dx.shape[1])
        dJ_phys = dJ_dx[:, :phys]
        F_phi = F_phi[:, :phys]
        G_matrix = G_phi_flat.view(batch_size, phys, self.control_dim)
        g = torch.bmm(G_matrix.transpose(1, 2), dJ_phys.unsqueeze(-1)).squeeze(-1)
        inv_r = 1.0 / self.R_diag.clamp(min=1e-12)
        term2 = 0.25 * torch.sum(g * g * inv_r.unsqueeze(0), dim=1)
        term3 = torch.sum(dJ_phys * F_phi, dim=1)

        hjb_residual = term1 + term2 + term3
        if torch.isnan(hjb_residual).any() or torch.isinf(hjb_residual).any():
            print("⚠️ HJB 残差含 NaN/Inf")
        mr = hjb_residual.abs().max().item()
        if mr > 1e4:
            print(f"⚠️ HJB 残差过大: max={mr:.2e}")
        return hjb_residual


# ============================================================
# 第四节: PINN训练器 (三自由度版本) - 对应论文 Algorithm 1
# ============================================================
class PINNTrainer:
    """
    三自由度 PINN训练器 - 实现论文 Algorithm 1 的训练流程

    与 `PINNController` / `AQWASimulator` 对齐：当提供 surrogate 的 x 归一化参数时，
    必须同时提供 `surrogate_u_mean/std` 与 `surrogate_dx_mean/std`（来自 scaler.json），
    HJB 残差使用物理控制下的等效漂移 F_eff 与雅可比 G_phys（见 `_surrogate_forward_hjb`）。
    
    论文 Algorithm 1 核心步骤:
    1. 从状态集合中采样一批 x
    2. 每10个epoch更新一次F_φ, G_φ, J_θ的前向传播
    3. 计算HJB损失
    4. 梯度下降更新J_θ参数
    5. 定期衰减学习率
    """
    
    def __init__(
        self,
        state_dim=6,
        control_dim=3,
        surrogate_hidden=128,
        value_hidden=1024,
        Q=None,
        R_diag=None,
        lr=0.001,
        decay_factor=0.5,
        decay_interval=50000,
        device='cpu',
        pinn_cfg: Optional[PINNHJBConfig] = None,
        surrogate_input_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        surrogate_input_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
        surrogate_u_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        surrogate_u_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
        surrogate_dx_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        surrogate_dx_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
        value_input_mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        value_input_std: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ):
        self.device = device
        self.control_dim = control_dim
        self.pinn_cfg = pinn_cfg

        if pinn_cfg is not None:
            state_dim = pinn_cfg.effective_state_dim()
            if R_diag is None:
                R_diag = tuple(float(x) for x in pinn_cfg.r_diag)
            if Q is None:
                q_list = list(pinn_cfg.q_diag_for_network())
                Q = torch.diag(torch.tensor(q_list, dtype=torch.float32))

        self.state_dim = state_dim

        if pinn_cfg is not None and pinn_cfg.use_regime_in_state:
            self.surrogate_model = ConditionalSurrogateModel(
                physics_dim=6,
                regime_dim=1,
                hidden_dim=surrogate_hidden,
                control_dim=control_dim,
            ).to(device)
        else:
            self.surrogate_model = SurrogateModel(
                state_dim=6,
                control_dim=control_dim,
                hidden_dim=surrogate_hidden,
            ).to(device)

        # 与 surrogate_trainer / AQWASimulator 一致：代理网络在**归一化状态**上训练，此处须先 (x-μ)/σ 再前向
        self._device_t = torch.device(device) if not isinstance(device, torch.device) else device
        self._surrogate_norm = False
        self._surrogate_x_mean: Optional[torch.Tensor] = None
        self._surrogate_x_std: Optional[torch.Tensor] = None
        self._phys_scaler: bool = False
        if surrogate_input_mean is not None and surrogate_input_std is not None:
            sm = torch.as_tensor(np.asarray(surrogate_input_mean, dtype=np.float64), dtype=torch.float32)
            ss = torch.as_tensor(np.asarray(surrogate_input_std, dtype=np.float64), dtype=torch.float32)
            sm = sm.flatten()
            ss = torch.clamp(ss.flatten(), min=1e-8)
            if sm.numel() != 6 or ss.numel() != 6:
                raise ValueError("surrogate_input_mean/std 须为长度 6，与物理状态 η/ν 一致")
            self._surrogate_x_mean = sm.view(1, 6).to(self._device_t)
            self._surrogate_x_std = ss.view(1, 6).to(self._device_t)
            self._surrogate_norm = True

            need_u_dx = (
                surrogate_u_mean is not None
                and surrogate_u_std is not None
                and surrogate_dx_mean is not None
                and surrogate_dx_std is not None
            )
            if not need_u_dx:
                raise ValueError(
                    "启用 surrogate 状态 (x) 归一化时，必须同时传入 surrogate_u_mean/std 与 "
                    "surrogate_dx_mean/std（与 surrogate_model/scaler.json 一致），"
                    "以便 HJB 中 F、G 与在线 PINNController 的 ∂Δx/∂u 量纲一致。"
                )
            um = torch.as_tensor(np.asarray(surrogate_u_mean, dtype=np.float64), dtype=torch.float32).reshape(3)
            us = torch.clamp(
                torch.as_tensor(np.asarray(surrogate_u_std, dtype=np.float64), dtype=torch.float32).reshape(3),
                min=1e-8,
            )
            dxm = torch.as_tensor(np.asarray(surrogate_dx_mean, dtype=np.float64), dtype=torch.float32).reshape(6)
            dxs = torch.clamp(
                torch.as_tensor(np.asarray(surrogate_dx_std, dtype=np.float64), dtype=torch.float32).reshape(6),
                min=1e-8,
            )
            self._hjb_u_mean = um.view(1, 1, 3).to(self._device_t)
            self._hjb_u_std = us.view(1, 1, 3).to(self._device_t)
            self._hjb_u_mean_over_std = (um / us).view(1, 1, 3).to(self._device_t)
            self._hjb_dx_mean = dxm.view(1, 6).to(self._device_t)
            self._hjb_dx_std = dxs.view(1, 6).to(self._device_t)
            self._hjb_dx_std_g = dxs.view(1, 6, 1).to(self._device_t)
            self._phys_scaler = True
        elif any(
            v is not None
            for v in (
                surrogate_u_mean,
                surrogate_u_std,
                surrogate_dx_mean,
                surrogate_dx_std,
            )
        ):
            raise ValueError("仅当提供 surrogate_input_mean/std 时才能传入 u/dx 标量。")

        self.value_network = ValueNetwork(
            state_dim,
            value_hidden,
            input_mean=value_input_mean,
            input_std=value_input_std,
        ).to(device)
        self.hjb_loss = HJBLoss(
            Q=Q,
            R_diag=R_diag,
            state_dim=state_dim,
            control_dim=control_dim,
            physics_dim=6,
        )
        
        # 优化器 (只优化价值网络参数)
        # 论文原文: "All neural network parameters are trained using the Adam 
        #            optimisation algorithm"
        # 但Algorithm 1显示只更新θ (value network参数)
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=decay_interval, 
            gamma=decay_factor
        )
        
        # 训练统计
        self.iteration = 0
        self.loss_history = []

    def _surrogate_forward_hjb(self, x_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        与 AQWASimulator / PINNController 一致：代理输入为归一化 x（及可选 regime）；
        若启用 _phys_scaler，将 (F_n, G_n) 换为 HJB 用的物理增量仿射系数：
          Δx = F_eff + G_phys @ u_phys，
        其中 G_phys[i,j] = G_n[i,j]*dx_std[i]/u_std[j]，
        F_eff = dx_std*F_n + dx_mean - dx_std * sum_j(G_n[i,j]*u_mean[j]/u_std[j])。
        """
        if self._surrogate_norm:
            x_n = (x_batch[:, :6] - self._surrogate_x_mean) / self._surrogate_x_std
            if isinstance(self.surrogate_model, ConditionalSurrogateModel):
                if x_batch.shape[1] >= 7:
                    reg = x_batch[:, 6:7].to(dtype=x_n.dtype)
                else:
                    reg = torch.zeros(
                        (x_batch.shape[0], 1),
                        device=self.device,
                        dtype=x_n.dtype,
                    )
                x_surr_in = torch.cat([x_n, reg], dim=1)
            else:
                x_surr_in = x_n
            F_n, G_flat = self.surrogate_model(x_surr_in)
        else:
            x_in = x_batch[:, :6] if x_batch.shape[1] > 6 else x_batch
            F_n, G_flat = self.surrogate_model(x_in)

        if not self._phys_scaler:
            return F_n, G_flat

        G = self.surrogate_model.get_G_matrix(G_flat)
        ustd = self._hjb_u_std
        G_phys = G * self._hjb_dx_std_g / ustd
        bias_n = (G * self._hjb_u_mean_over_std).sum(dim=2)
        dxs = self._hjb_dx_std
        F_eff = F_n * dxs + self._hjb_dx_mean - bias_n * dxs
        return F_eff, G_phys.reshape(G.size(0), -1)
        
    def train_step(self, x_batch):
        """
        单步训练 - 对应论文 Algorithm 1 的内层循环
        
        ✅ 改进: 添加梯度监控、Loss裁剪、异常检测
        
        参数:
        - x_batch: 状态样本 (batch_size, state_dim)
        """
        x_batch = torch.FloatTensor(x_batch).to(self.device)
        
        # 代理：(x_n[,regime]) → F,G；HJB 用物理 u 下的 F_eff、G_phys（与 PINNController 一致）
        F_phi, G_phi_flat = self._surrogate_forward_hjb(x_batch)
        dJ_dx = self.value_network.compute_gradient(x_batch)
        
        # ✅ 新增: 检查梯度是否异常
        if torch.isnan(dJ_dx).any() or torch.isinf(dJ_dx).any():
            print(f"⚠️ Epoch {len(self.loss_history)}: 梯度包含NaN/Inf，跳过此batch")
            return None
        
        # 计算HJB损失
        hjb_residual = self.hjb_loss.compute_hjb_residual(
            x_batch, F_phi, G_phi_flat, dJ_dx
        )
        loss = torch.mean(hjb_residual ** 2)  # MSE
        
        # ✅ 新增: Loss裁剪检查
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Epoch {len(self.loss_history)}: Loss为NaN/Inf，跳过此batch")
            return None
        
        if loss.item() > 10000:
            print(f"⚠️ Epoch {len(self.loss_history)}: Loss过大({loss.item():.2f})，跳过此batch")
            return None
        
        # 反向传播 (只更新价值网络)
        self.optimizer.zero_grad()
        loss.backward()
        
        # ✅ 新增: 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(
            self.value_network.parameters(),
            max_norm=10.0,
        )
        
        self.optimizer.step()
        
        # 更新统计
        self.iteration += 1
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def train(
        self, 
        state_samples, 
        num_epochs=1000,
        batch_size=64,
        print_interval=100
    ):
        """
        完整训练流程
        
        ✅ 改进: 学习率预热、更好的监控、异常检测
        
        参数:
        - state_samples: 状态采样点 (num_samples, state_dim)
        - num_epochs: 训练轮数
        - batch_size: 批大小
        - print_interval: 打印间隔
        """
        n_samples = len(state_samples)
        dataset_size = n_samples
        
        print(f"开始PINN训练 (PINN-HJB 价值网络)...")
        print(f"  状态维度: {self.state_dim}")
        print(f"  控制维度: {self.control_dim}")
        print(f"  样本数量: {n_samples}")
        print(f"  批大小: {batch_size}")
        print(f"  训练轮数: {num_epochs}")
        print(f"  设备: {self.device}")
        print("=" * 50)
        
        # ✅ 获取初始学习率；较短预热以便更快达到目标学习率
        initial_lr = self.optimizer.param_groups[0]['lr']
        warmup_epochs = 50
        
        # ✅ 新增: 统计失败的batch数
        failed_batches = 0
        last_valid_loss = None
        
        for epoch in range(num_epochs):
            # ✅ 新增: 学习率预热 (前100个epoch逐步增加学习率)
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                current_lr = initial_lr * warmup_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                x_batch = state_samples[batch_indices]
                
                # 训练步骤
                loss = self.train_step(x_batch)
                
                # ✅ 新增: 处理失败的batch
                if loss is None:
                    failed_batches += 1
                    continue
                
                epoch_loss += loss
                n_batches += 1
                last_valid_loss = loss
            
            # ✅ 改进: 处理所有batch都失败的情况
            if n_batches == 0:
                print(f"⚠️ Epoch {epoch}: 所有batch都失败，跳过此epoch")
                continue
            
            # 学习率衰减 (对应论文 line 16)
            self.scheduler.step()
            
            # 打印进度
            if (epoch + 1) % print_interval == 0 or (epoch < warmup_epochs and (epoch + 1) % 10 == 0):
                avg_loss = epoch_loss / n_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4d}/{num_epochs} | "
                      f"Loss: {avg_loss:.6f} | "
                      f"LR: {current_lr:.6e} | "
                      f"Failed: {failed_batches}")
                
                # HJB MSE 量级常远大于 100，仅在预热结束后且仍极高时提示
                if epoch + 1 > warmup_epochs and avg_loss > 5000:
                    print(f"  ⚠️ Loss 仍偏高，可考虑检查 Q/R 或学习率")
                if avg_loss < 1e-6:
                    print(f"  ✅ Loss已很小，可考虑提前停止")
        
        print("=" * 50)
        print(f"PINN训练完成! 失败batch数: {failed_batches}")
        if last_valid_loss is not None:
            print(f"最后有效loss: {last_valid_loss:.6f}")
        else:
            print("最后有效loss: N/A（本 epoch 无成功 batch）")
        return self.loss_history


# ============================================================
# 第五节: 在线控制器 (三自由度版本) - 对应论文 Eq.10
# ============================================================
class PINNController:
    """
    三自由度 PINN-HJB 在线控制器 - 对应论文的控制框架
    
    论文 Eq.10 (最优控制律):
    "u*(x) = -1/2 R^{-1} G_φ(x)^T ∇J*_θ(x)"
    
    三自由度版本 (MIMO):
    u* = -1/2 R^{-1} G_φ^T ∇J_θ
    
    其中:
    - R ∈ R^{3×3}: 控制权重矩阵
    - G_φ ∈ R^{6×3}: 控制效应矩阵
    - ∇J_θ ∈ R^6: 值函数梯度
    - u* ∈ R^3: 最优控制 [u_surge, u_sway, u_yaw]
    
    控制流程:
    1. 获取当前系统状态 x (6维)
    2. 计算 G_φ(x) (6×3矩阵) 和 ∇J_θ(x) (6维)
    3. 计算最优控制 u* = -1/2 R^{-1} G_φ^T ∇J_θ (3维)
    4. 应用控制到系统
    """
    
    def __init__(
        self,
        surrogate_model,
        value_network,
        R_diag=None,
        device='cpu',
        pinn_cfg: Optional[PINNHJBConfig] = None,
        clip_ranges: Optional[list] = None,
        surrogate_x_mean: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        surrogate_x_std: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        surrogate_u_std: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
        surrogate_dx_std: Optional[Union[np.ndarray, Sequence[float], torch.Tensor]] = None,
    ):
        self.surrogate_model = surrogate_model.to(device)
        self.value_network = value_network.to(device)
        self.device = device
        self.state_dim = surrogate_model.state_dim
        self.control_dim = surrogate_model.control_dim
        self.pinn_cfg = pinn_cfg
        self._default_clip = (
            list(clip_ranges)
            if clip_ranges is not None
            else DEFAULT_CONFIG.clip_ranges()
        )

        if R_diag is None:
            if pinn_cfg is not None:
                R_diag = tuple(float(x) for x in pinn_cfg.r_diag)
            else:
                R_diag = tuple(float(x) for x in DEFAULT_CONFIG.r_diag)
        R_values = torch.tensor(R_diag, dtype=torch.float32, device=device)
        self.R = torch.diag(R_values)
        self.R_inv = torch.inverse(self.R)
        self.physics_dim = getattr(surrogate_model, "physics_dim", 6)

        # 与 surrogate_trainer / AQWASimulator 一致：代理在 (x-μ)/σ 与 u_n 空间训练；
        # 在线控制律中的 G 须为 ∂Δx_phys/∂u_phys = G_n * diag(dx_std) @ diag(1/u_std)。
        self._surrogate_norm: bool = (
            surrogate_x_mean is not None
            and surrogate_x_std is not None
            and surrogate_u_std is not None
            and surrogate_dx_std is not None
        )
        if self._surrogate_norm:
            xm = torch.as_tensor(np.asarray(surrogate_x_mean, dtype=np.float64), dtype=torch.float32).reshape(1, 6)
            xs = torch.clamp(
                torch.as_tensor(np.asarray(surrogate_x_std, dtype=np.float64), dtype=torch.float32).reshape(1, 6),
                min=1e-8,
            )
            us = torch.clamp(
                torch.as_tensor(np.asarray(surrogate_u_std, dtype=np.float64), dtype=torch.float32).reshape(1, 1, 3),
                min=1e-8,
            )
            ds = torch.as_tensor(np.asarray(surrogate_dx_std, dtype=np.float64), dtype=torch.float32).reshape(1, 6, 1)
            self._sur_x_mean = xm.to(self.device)
            self._sur_x_std = xs.to(self.device)
            self._sur_u_std = us.to(self.device)
            self._sur_dx_std = ds.to(self.device)
        else:
            self._sur_x_mean = self._sur_x_std = self._sur_u_std = self._sur_dx_std = None  # type: ignore[assignment]
        
    def compute_control(self, x, clip_ranges=None, mooring_regime: float = 0.0):
        """
        计算三自由度最优控制
        
        参数:
        - x: 当前状态 (6,) 或 (batch, 6)；若代理为 ConditionalSurrogateModel 且仅传入 6 维，将自动拼接 mooring_regime
        - clip_ranges: 控制输出范围限制
            格式: [(min, max), (min, max), (min, max)] 对应 [surge, sway, yaw]
            默认: 取自 DEFAULT_CONFIG
        - mooring_regime: 0=完好, 1=失效（与 MooringRegime 一致）
        
        返回:
        - u: 最优控制 (3,) 或 (batch, 3)
        """
        x_tensor = torch.FloatTensor(x).to(self.device)
        if x_tensor.dim() == 1:
            x_tensor = x_tensor.unsqueeze(0)

        x_phys = x_tensor[:, :6].contiguous()

        if isinstance(self.surrogate_model, ConditionalSurrogateModel):
            if x_tensor.shape[1] == 6:
                reg = torch.full(
                    (x_tensor.shape[0], 1),
                    float(mooring_regime),
                    device=self.device,
                    dtype=x_tensor.dtype,
                )
                x_val = torch.cat([x_phys, reg], dim=1)
            else:
                x_val = x_tensor
        else:
            x_val = x_phys if x_tensor.shape[1] == 6 else x_tensor[:, :6].contiguous()

        # 需要 requires_grad 才能计算 ∇J（物理状态；条件代理时含 regime 列）
        x_val = x_val.clone()
        x_val.requires_grad_(True)

        # 获取 G：与训练时一致先送入归一化状态，再换算为 ∂Δx_phys/∂u_phys
        with torch.no_grad():
            if self._surrogate_norm:
                x_n = (x_phys.detach() - self._sur_x_mean) / self._sur_x_std
                if isinstance(self.surrogate_model, ConditionalSurrogateModel):
                    reg_col = x_val[:, 6:7].detach() if x_val.shape[1] > 6 else torch.full(
                        (x_val.shape[0], 1),
                        float(mooring_regime),
                        device=self.device,
                        dtype=torch.float32,
                    )
                    x_surr_in = torch.cat([x_n, reg_col.to(dtype=x_n.dtype)], dim=1)
                else:
                    x_surr_in = x_n
                _, G_phi_flat = self.surrogate_model(x_surr_in)
                G_matrix = self.surrogate_model.get_G_matrix(G_phi_flat)
                G_matrix = G_matrix * self._sur_dx_std / self._sur_u_std
            else:
                x_surr_in = x_val.detach()
                if not isinstance(self.surrogate_model, ConditionalSurrogateModel):
                    x_surr_in = x_surr_in[:, :6]
                _, G_phi_flat = self.surrogate_model(x_surr_in)
                G_matrix = self.surrogate_model.get_G_matrix(G_phi_flat)
        
        # 计算值函数梯度 (需要梯度追踪)
        dJ_dx = self.value_network.compute_gradient(x_val)
        dJ_phys = dJ_dx[:, : self.physics_dim]

        # 最优控制律: u* = -1/2 R^{-1} G_φ^T ∇J_θ
        GT_dJ = torch.bmm(
            G_matrix.transpose(1, 2), dJ_phys.unsqueeze(-1)
        ).squeeze(-1)  # (batch, 3)
        
        R_inv_GT_dJ = torch.mm(self.R_inv, GT_dJ.T).T
        u = -0.5 * R_inv_GT_dJ
        
        u = u.detach().cpu().numpy()
        
        if u.shape[0] == 1:
            u = u[0]

        u_pre_scale = np.asarray(u, dtype=np.float64).reshape(-1).copy()

        cfg = self.pinn_cfg
        u_physics_scale_list = None
        if cfg is not None and hasattr(cfg, "u_physics_scale"):
            sc = np.asarray(cfg.u_physics_scale, dtype=np.float64).reshape(3)
            u_physics_scale_list = sc.tolist()
            u = u * sc

        ranges = clip_ranges if clip_ranges is not None else self._default_clip
        u_pre_clip = np.asarray(u, dtype=np.float64).reshape(-1).copy()
        lo = np.array([r[0] for r in ranges], dtype=np.float64)
        hi = np.array([r[1] for r in ranges], dtype=np.float64)
        u = np.clip(u, lo, hi)

        # #region agent log
        _agent_dbg_n = getattr(PINNController.compute_control, "_bce714_n", 0)
        if _agent_dbg_n < 50:
            setattr(PINNController.compute_control, "_bce714_n", _agent_dbg_n + 1)
            try:
                import json
                import os
                import time as _agent_time

                _dj = dJ_phys.detach().float().cpu().numpy().reshape(-1)
                _gt = np.asarray(GT_dJ.detach().float().cpu().numpy()).reshape(-1)
                _g = G_matrix.detach().float().cpu().numpy().reshape(-1)
                _p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug-bce714.log")
                with open(_p, "a", encoding="utf-8") as _af:
                    _af.write(
                        json.dumps(
                            {
                                "sessionId": "bce714",
                                "hypothesisId": "H1_H2_H3",
                                "location": "pinn_hjb_controller.py:compute_control",
                                "message": "dJ vs GT_dJ vs u pipeline",
                                "timestamp": int(_agent_time.time() * 1000),
                                "data": {
                                    "n": _agent_dbg_n,
                                    "mooring_regime": float(mooring_regime),
                                    "norm_dJ_phys": float(np.linalg.norm(_dj)),
                                    "norm_GT_dJ": float(np.linalg.norm(_gt)),
                                    "norm_G_flat": float(np.linalg.norm(_g)),
                                    "dJ_phys": _dj.tolist(),
                                    "GT_dJ": _gt.tolist(),
                                    "u_after_neg_half_Rinv_GTdJ": u_pre_scale.tolist(),
                                    "u_physics_scale": u_physics_scale_list,
                                    "u_pre_clip": u_pre_clip.tolist(),
                                    "u_after_clip": np.asarray(u, dtype=np.float64).reshape(-1).tolist(),
                                    "clip_lo": lo.tolist(),
                                    "clip_hi": hi.tolist(),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

        return u
    
    def compute_control_simple(self, x):
        """
        简化版三自由度控制计算
        
        参数:
        - x: 当前状态 (6,)
        
        返回:
        - u: 最优控制 (3,)
        """
        return self.compute_control(x)


# ============================================================
# 第六节: 离线数据生成工具 (三自由度版本)
# ============================================================
class DataGenerator:
    """
    三自由度离线数据生成器
    
    用于生成代理模型的训练数据，包含状态转移对:
    (x_k, u_k) → x_{k+1}
    
    三自由度状态向量:
    x = [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw]
    
    控制向量:
    u = [u_surge, u_sway, u_yaw]
    """
    
    # 三自由度半潜平台的默认状态范围
    DEFAULT_STATE_BOUNDS = [
        (-20.0, 20.0),   # η_surge  纵荡位移 [m]
        (-20.0, 20.0),   # η_sway   横荡位移 [m]
        (-0.2, 0.2),     # η_yaw    艏摇角度 [rad] ≈ ±11.5°
        (-2.0, 2.0),     # ν_surge  纵荡速度 [m/s]
        (-2.0, 2.0),     # ν_sway   横荡速度 [m/s]
        (-0.1, 0.1),     # ν_yaw    艏摇角速度 [rad/s]
    ]
    
    # 控制力范围
    DEFAULT_CONTROL_BOUNDS = [
        (0, 8e6),        # u_surge  X向推力 [N]
        (0, 8e6),        # u_sway   Y向推力 [N]
        (-1e9, 1e9),     # u_yaw    艏摇力矩 [N·m]
    ]
    
    @staticmethod
    def latin_hypercube_sampling(bounds, n_samples):
        """
        Latin超立方采样
        
        参数:
        - bounds: list of (min, max) for each dimension
        - n_samples: 采样数量
        
        返回:
        - samples: (n_samples, n_dims)
        """
        n_dims = len(bounds)
        samples = np.zeros((n_samples, n_dims))
        
        for d in range(n_dims):
            # 将[0,1]区间分成n_samples份
            intervals = np.linspace(0, 1, n_samples + 1)
            # 每份中随机取一点
            samples[:, d] = np.random.uniform(intervals[:-1], intervals[1:])
        
        # 随机打乱每列
        for d in range(n_dims):
            np.random.shuffle(samples[:, d])
        
        # 映射到实际范围
        for d in range(n_dims):
            low, high = bounds[d]
            samples[:, d] = low + (high - low) * samples[:, d]
        
        return samples
    
    @staticmethod
    def generate_state_samples(n_samples=10000, state_bounds=None):
        """
        生成三自由度状态采样点
        
        参数:
        - n_samples: 采样数量
        - state_bounds: 状态范围，默认使用深海一号平台参数
        
        返回:
        - samples: (n_samples, 6) 状态采样点
        """
        if state_bounds is None:
            state_bounds = DataGenerator.DEFAULT_STATE_BOUNDS
        
        return DataGenerator.latin_hypercube_sampling(state_bounds, n_samples)
    
    @staticmethod
    def generate_control_samples(n_samples=1000, control_bounds=None):
        """
        生成三自由度控制采样点
        
        参数:
        - n_samples: 采样数量
        - control_bounds: 控制范围
        
        返回:
        - samples: (n_samples, 3) 控制采样点
        """
        if control_bounds is None:
            control_bounds = DataGenerator.DEFAULT_CONTROL_BOUNDS
        
        return DataGenerator.latin_hypercube_sampling(control_bounds, n_samples)
    
    @staticmethod
    def generate_training_data(
        state_bounds=None,
        control_bounds=None,
        n_initial_conditions=100,
        sim_duration=15.0,
        dt=0.01,
        control_sample_interval=0.1
    ):
        """
        生成三自由度代理模型训练数据
        
        参数:
        - state_bounds: 状态范围 (默认深海一号平台)
        - control_bounds: 控制力范围
        - n_initial_conditions: 初始条件数量
        - sim_duration: 每次仿真持续时间
        - dt: 仿真时间步
        - control_sample_interval: 控制力更新间隔
        
        返回:
        - X: (num_samples, 6) 当前状态
        - U: (num_samples, 3) 控制输入
        - X_next: (num_samples, 6) 下一状态
        
        注意: 此函数需要接入真实仿真器(AQWA)来获取状态转移
              目前只生成采样点框架
        """
        if state_bounds is None:
            state_bounds = DataGenerator.DEFAULT_STATE_BOUNDS
        if control_bounds is None:
            control_bounds = DataGenerator.DEFAULT_CONTROL_BOUNDS
        
        n_steps = int(sim_duration / dt)
        n_control_steps = int(control_sample_interval / dt)
        
        # 生成初始状态和控制输入
        X0 = DataGenerator.latin_hypercube_sampling(state_bounds, n_initial_conditions)
        U_samples = DataGenerator.latin_hypercube_sampling(control_bounds, n_initial_conditions)
        
        X_list = []
        U_list = []
        X_next_list = []
        
        print(f"生成三自由度训练数据...")
        print(f"  状态维度: {len(state_bounds)}")
        print(f"  控制维度: {len(control_bounds)}")
        print(f"  初始条件数量: {n_initial_conditions}")
        print(f"  每次仿真步数: {n_steps}")
        print(f"  采样间隔: {control_sample_interval}s")
        
        # 状态范围信息
        print("\n状态范围:")
        names = ['η_surge', 'η_sway', 'η_yaw', 'ν_surge', 'ν_sway', 'ν_yaw']
        for i, (name, bound) in enumerate(zip(names, state_bounds)):
            print(f"  {name}: [{bound[0]}, {bound[1]}]")
        
        print("\n控制范围:")
        ctrl_names = ['u_surge', 'u_sway', 'u_yaw']
        for name, bound in zip(ctrl_names, control_bounds):
            print(f"  {name}: [{bound[0]:.2e}, {bound[1]:.2e}]")
        
        # 需要实际系统模型来生成数据
        # 这里提供框架，实际使用时需要接入AQWA仿真
        
        return np.array(X_list), np.array(U_list), np.array(X_next_list)


# ============================================================
# 第七节: 模型保存与加载 (三自由度版本)
# ============================================================
def value_network_from_state_dict(
    state_dict: dict,
    state_dim: int = 6,
    value_hidden: int = 1024,
) -> "ValueNetwork":
    """
    从 state_dict 构建 ValueNetwork（兼容含/不含 input_mean、input_std 的权重）。
    """
    if "input_mean" in state_dict and "input_std" in state_dict:
        im = state_dict["input_mean"].detach().cpu().numpy().flatten()
        iv = state_dict["input_std"].detach().cpu().numpy().flatten()
        return ValueNetwork(
            state_dim, value_hidden, input_mean=im, input_std=iv
        )
    return ValueNetwork(state_dim, value_hidden)


def save_pinn_models(surrogate_model, value_network, filepath_prefix):
    """
    保存三自由度PINN模型
    
    参数:
    - surrogate_model: 代理模型
    - value_network: 价值网络
    - filepath_prefix: 文件路径前缀
    """
    torch.save({
        'surrogate_state_dict': surrogate_model.state_dict(),
        'value_state_dict': value_network.state_dict(),
        'state_dim': surrogate_model.state_dim,
        'control_dim': surrogate_model.control_dim,
    }, f"{filepath_prefix}_pinn_models_3dof.pth")
    print(f"模型已保存: {filepath_prefix}_pinn_models_3dof.pth")


def load_pinn_models(filepath, state_dim=6, control_dim=3, surrogate_hidden=128, value_hidden=1024):
    """
    加载三自由度PINN模型
    
    参数:
    - filepath: 模型文件路径
    - state_dim: 状态维度
    - control_dim: 控制维度
    - surrogate_hidden: 代理模型隐藏层维度
    - value_hidden: 价值网络隐藏层维度
    
    返回:
    - surrogate_model, value_network: 加载后的模型
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    surrogate_model = SurrogateModel(state_dim, control_dim, surrogate_hidden)
    vsd = checkpoint["value_state_dict"]
    value_network = value_network_from_state_dict(vsd, state_dim, value_hidden)
    surrogate_model.load_state_dict(checkpoint['surrogate_state_dict'])
    value_network.load_state_dict(vsd)
    
    surrogate_model.eval()
    value_network.eval()
    
    print(f"模型已加载: {filepath}")
    print(f"  状态维度: {checkpoint.get('state_dim', state_dim)}")
    print(f"  控制维度: {checkpoint.get('control_dim', control_dim)}")
    
    return surrogate_model, value_network


# ============================================================
# 使用示例与测试 (三自由度版本)
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🧪 PINN-HJB 三自由度控制器测试")
    print("   深海一号半潜式平台: 纵荡 + 横荡 + 艏摇")
    print("=" * 70)
    
    # 配置
    STATE_DIM = 6      # 三自由度状态
    CONTROL_DIM = 3    # 三自由度控制
    DEVICE = 'cpu'
    
    # 初始化网络
    print("\n📦 初始化网络...")
    surrogate = SurrogateModel(state_dim=STATE_DIM, control_dim=CONTROL_DIM, hidden_dim=128)
    value_net = ValueNetwork(state_dim=STATE_DIM, hidden_dim=1024)
    
    print("\n代理模型结构:")
    print(surrogate)
    print(f"参数数量: {sum(p.numel() for p in surrogate.parameters()):,}")
    
    print("\n价值网络结构:")
    print(value_net)
    print(f"参数数量: {sum(p.numel() for p in value_net.parameters()):,}")
    
    # 生成随机测试数据
    print("\n" + "-" * 50)
    print("测试前向传播...")
    test_x = torch.randn(10, STATE_DIM)  # 10个样本, 6维状态
    
    # 测试代理模型
    F, G_flat = surrogate(test_x)
    print(f"F_φ(x) 输出形状: {F.shape}  (内在动力学, 6维)")
    print(f"G_φ(x) 输出形状: {G_flat.shape}  (展平的控制效应矩阵, 18=6×3)")
    
    # 测试G矩阵重塑
    G_matrix = surrogate.get_G_matrix(G_flat)
    print(f"G_φ 重塑后: {G_matrix.shape}  (batch, 6, 3)")
    
    # 测试价值网络
    J = value_net(test_x)
    print(f"J_θ(x) 输出形状: {J.shape}  (标量值函数)")
    
    # 测试梯度计算
    dJ_dx = value_net.compute_gradient(test_x)
    print(f"∇J_θ(x) 输出形状: {dJ_dx.shape}  (值函数梯度, 6维)")
    
    # 测试控制器
    print("\n" + "-" * 50)
    print("测试三自由度PINN控制器...")
    controller = PINNController(
        surrogate_model=surrogate, 
        value_network=value_net, 
        R_diag=[1e-4, 1e-4, 1e-2],  # [surge, sway, yaw] 权重
        device=DEVICE
    )
    
    # 测试单状态输入
    test_state = np.array([0.5, 0.3, 0.01, 0.1, 0.05, 0.001])  # [η_surge, η_sway, η_yaw, ν_surge, ν_sway, ν_yaw]
    u = controller.compute_control_simple(test_state)
    print(f"\n测试状态 (6维):")
    print(f"  纵荡: {test_state[0]:.3f} m, 速度: {test_state[3]:.3f} m/s")
    print(f"  横荡: {test_state[1]:.3f} m, 速度: {test_state[4]:.3f} m/s")
    print(f"  艏摇: {test_state[2]:.3f} rad, 角速度: {test_state[5]:.3f} rad/s")
    print(f"\n计算得到控制量 (3维):")
    print(f"  u_surge: {u[0]:,.0f} N")
    print(f"  u_sway:  {u[1]:,.0f} N")
    print(f"  u_yaw:   {u[2]:,.0f} N·m")
    
    # 测试HJB损失
    print("\n" + "-" * 50)
    print("测试HJB损失...")
    hjb_loss = HJBLoss(R_diag=[1e-4, 1e-4, 1e-2])
    residual = hjb_loss.compute_hjb_residual(test_x, F, G_flat, dJ_dx)
    print(f"HJB残差形状: {residual.shape}")
    print(f"HJB残差均值: {residual.mean().item():.6f}")
    print(f"HJB残差范围: [{residual.min().item():.6f}, {residual.max().item():.6f}]")
    
    # 状态采样测试
    print("\n" + "-" * 50)
    print("测试状态采样...")
    state_samples = DataGenerator.generate_state_samples(n_samples=1000)
    print(f"生成状态样本: {state_samples.shape}")
    print(f"纵荡范围: [{state_samples[:,0].min():.2f}, {state_samples[:,0].max():.2f}] m")
    print(f"横荡范围: [{state_samples[:,1].min():.2f}, {state_samples[:,1].max():.2f}] m")
    print(f"艏摇范围: [{state_samples[:,2].min():.3f}, {state_samples[:,2].max():.3f}] rad")
    
    print("\n" + "=" * 70)
    print("✅ 三自由度PINN-HJB控制器测试完成!")
    print("=" * 70)
