# -*- coding: utf-8 -*-
"""
AQWA 过渡数据离线训练代理模型 SurrogateModel
============================================
从符合 `pinn_hjb_data_regimes.aqwa_online_log_csv_header()` 列名的 CSV 读取
（与 `pinn_hjb_aqwa_online_log.csv` / `initialize_log` 表头一致）；按相邻两行构造
(x_k, u_k, x_{k+1})，拟合 Δx = x_{k+1} - x_k，与 `pinn_hjb_aqwa_integration.AQWASimulator`
使用的归一化与 `predict_delta` 接口一致。

输出:
  - surrogate_model/surrogate_model.pt  (含 model_state_dict, info)
  - surrogate_model/scaler.json         (x/u/dx 的 mean/std，与集成层一致)

用法:
  在下方 TRANSITION_CSV_PATH 写入过渡数据 CSV 路径后执行:
  python surrogate_trainer.py

Author: generated for PINN-HJB × AQWA pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pinn_hjb_controller import SurrogateModel
from pinn_hjb_data_regimes import (
    aqwa_online_log_csv_header,
    aqwa_online_log_state_control_columns,
)


# 与 pinn_hjb_aqwa_online / integration 中默认根目录对齐（可用环境变量覆盖）
DEFAULT_ANALYSIS_DIR = os.environ.get(
    "PINN_HJB_ANALYSIS_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

# 过渡数据 CSV：默认使用在线日志；可改为任意绝对路径
TRANSITION_CSV_PATH = os.path.join(
    DEFAULT_ANALYSIS_DIR, "pinn_hjb_aqwa_online_log.csv"
)


@dataclass
class TransitionArrays:
    x: np.ndarray  # (N, 6)
    u: np.ndarray  # (N, 3)
    dx: np.ndarray  # (N, 6)  x_next - x


def _expected_headers() -> List[str]:
    return aqwa_online_log_csv_header()


def _csv_stripped_header_map(fieldnames: Optional[List[str]]) -> Dict[str, str]:
    """
    CSV 表头单元格常带前导/尾随空格（如 \" u_sway(N)\"），DictReader 的键含空格，
    与代码中的规范列名不一致会导致 KeyError。返回 strip 后的列名 -> 实际键。
    """
    out: Dict[str, str] = {}
    if not fieldnames:
        return out
    for h in fieldnames:
        if h is None:
            continue
        s = h.strip()
        if s not in out:
            out[s] = h
    return out


# 常见编码：Excel/记事本在中文 Windows 下另存常为 GBK 系，与 UTF-8 混用时会触发 UnicodeDecodeError
_CSV_ENCODING_FALLBACKS: Tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "cp936",
)


def _open_csv_read(path: str, preferred_encoding: str):
    """按顺序尝试编码并打开文本流，读取一段以尽早发现解码错误。"""
    tried: List[str] = []
    last_err: Optional[UnicodeDecodeError] = None
    for enc in (preferred_encoding,) + tuple(
        e for e in _CSV_ENCODING_FALLBACKS if e != preferred_encoding
    ):
        tried.append(enc)
        f = None
        try:
            f = open(path, "r", newline="", encoding=enc)
            f.read(262144)
            f.seek(0)
            return f, enc
        except UnicodeDecodeError as e:
            last_err = e
            if f is not None:
                f.close()
    msg = (
        f"无法用以下编码读取 CSV: {tried}。"
        "请用记事本/VS Code 将文件另存为 UTF-8，或指定正确编码。"
    )
    if last_err is not None:
        raise RuntimeError(msg) from last_err
    raise RuntimeError(msg)


def load_transition_csv(
    path: str,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
) -> TransitionArrays:
    """
    读取与在线日志约定一致的 CSV（含 Time(s)、η/ν、u 列）。
    无显式下一时刻状态列时，用相邻两行构造 x_k、u_k、x_{k+1}。
    首选 *encoding*（默认 utf-8-sig）；失败时自动尝试 gb18030/gbk 等常见中文编码。
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    x_cols, u_cols = aqwa_online_log_state_control_columns()
    need = set(x_cols + u_cols)

    f, _used_enc = _open_csv_read(path, encoding)
    with f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("CSV 无表头")
        header_map = _csv_stripped_header_map(reader.fieldnames)
        missing = need - set(header_map.keys())
        if missing:
            raise ValueError(
                f"CSV 缺少列: {sorted(missing)}。期望至少包含: {sorted(need)}"
            )

        rows_x: List[List[float]] = []
        rows_u: List[List[float]] = []
        for row in reader:
            try:
                xv = [float(row[header_map[c]]) for c in x_cols]
                uv = [float(row[header_map[c]]) for c in u_cols]
            except (KeyError, ValueError, TypeError):
                continue
            rows_x.append(xv)
            rows_u.append(uv)

    if len(rows_x) < 2:
        raise ValueError(f"有效行过少 ({len(rows_x)}), 至少需要 2 行以构造一步转移")

    x_all = np.asarray(rows_x, dtype=np.float64)
    u_all = np.asarray(rows_u, dtype=np.float64)
    x = x_all[:-1]
    u = u_all[:-1]
    x_next = x_all[1:]
    dx = x_next - x

    mask = np.isfinite(x).all(1) & np.isfinite(u).all(1) & np.isfinite(dx).all(1)
    x, u, dx = x[mask], u[mask], dx[mask]
    if len(x) < 16:
        raise ValueError(
            f"有效转移过少 ({len(x)}), 需≥16；请检查 CSV 或增加记录长度"
        )
    return TransitionArrays(x=x, u=u, dx=dx)


def compute_scalers(
    x: np.ndarray,
    u: np.ndarray,
    dx: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    u_mean = u.mean(axis=0)
    u_std = u.std(axis=0)
    dx_mean = dx.mean(axis=0)
    dx_std = dx.std(axis=0)
    x_std = np.where(np.abs(x_std) < eps, eps, x_std)
    u_std = np.where(np.abs(u_std) < eps, eps, u_std)
    dx_std = np.where(np.abs(dx_std) < eps, eps, dx_std)
    return x_mean, x_std, u_mean, u_std, dx_mean, dx_std


def save_scaler_json(
    path: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    u_mean: np.ndarray,
    u_std: np.ndarray,
    dx_mean: np.ndarray,
    dx_std: np.ndarray,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "x_mean": x_mean.astype(np.float64).tolist(),
        "x_std": x_std.astype(np.float64).tolist(),
        "u_mean": u_mean.astype(np.float64).tolist(),
        "u_std": u_std.astype(np.float64).tolist(),
        "dx_mean": dx_mean.astype(np.float64).tolist(),
        "dx_std": dx_std.astype(np.float64).tolist(),
    }
    if extra:
        data["meta"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def train_surrogate(
    x: np.ndarray,
    u: np.ndarray,
    dx: np.ndarray,
    *,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    batch_size: int = 256,
    epochs: int = 200,
    val_frac: float = 0.15,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[SurrogateModel, Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_mean, x_std, u_mean, u_std, dx_mean, dx_std = compute_scalers(x, u, dx)

    x_n = (x - x_mean) / x_std
    u_n = (u - u_mean) / u_std
    dx_n = (dx - dx_mean) / dx_std

    n = x_n.shape[0]
    idx = np.random.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if tr_idx.size == 0:
        tr_idx, val_idx = val_idx, tr_idx

    def pack(ii: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(x_n[ii]).float().to(device),
            torch.from_numpy(u_n[ii]).float().to(device),
            torch.from_numpy(dx_n[ii]).float().to(device),
        )

    xt, ut, dxt = pack(tr_idx)
    xv, uv, dxv = pack(val_idx)

    ds_tr = TensorDataset(xt, ut, dxt)
    ds_va = TensorDataset(xv, uv, dxv)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SurrogateModel(state_dim=6, control_dim=3, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: Optional[Dict] = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, ub, db in dl_tr:
            opt.zero_grad(set_to_none=True)
            pred = model.predict_delta(xb, ub)
            loss = loss_fn(pred, db)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            n_tr += xb.size(0)
        tr_loss /= max(n_tr, 1)

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for xb, ub, db in dl_va:
                pred = model.predict_delta(xb, ub)
                loss = loss_fn(pred, db)
                va_loss += float(loss.item()) * xb.size(0)
                n_va += xb.size(0)
        va_loss /= max(n_va, 1)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"  epoch {ep:4d}/{epochs}  train_mse={tr_loss:.6e}  val_mse={va_loss:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "train_mse_norm_dx": tr_loss,
        "val_mse_norm_dx": best_val,
        "n_samples": float(n),
        "n_train": float(tr_idx.size),
        "n_val": float(val_idx.size),
    }
    return model, metrics, x_mean, x_std, u_mean, u_std, dx_mean, dx_std


def main() -> None:
    parser = argparse.ArgumentParser(description="离线训练 AQWA 代理 SurrogateModel")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="输出目录，默认 <PINN_HJB_ANALYSIS_DIR>/surrogate_model",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="", help="cuda / cpu，空则自动")
    parser.add_argument(
        "--print-expected-header",
        action="store_true",
        help="打印参考 CSV 表头后退出",
    )
    args = parser.parse_args()

    if args.print_expected_header:
        print(",".join(_expected_headers()))
        return

    base = DEFAULT_ANALYSIS_DIR
    out_dir = args.out_dir.strip() or os.path.join(base, "surrogate_model")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("  SurrogateModel 离线训练")
    print("=" * 60)
    csv_path = os.path.abspath(TRANSITION_CSV_PATH)
    print(f"  数据: {csv_path}")
    print(f"  输出: {os.path.abspath(out_dir)}")

    data = load_transition_csv(csv_path)
    print(f"  样本数: {data.x.shape[0]}")

    dev = None
    if args.device:
        dev = torch.device(args.device)

    model, metrics, x_mean, x_std, u_mean, u_std, dx_mean, dx_std = train_surrogate(
        data.x,
        data.u,
        data.dx,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_frac=args.val_frac,
        seed=args.seed,
        device=dev,
    )

    scaler_path = os.path.join(out_dir, "scaler.json")
    model_path = os.path.join(out_dir, "surrogate_model.pt")

    save_scaler_json(
        scaler_path,
        x_mean,
        x_std,
        u_mean,
        u_std,
        dx_mean,
        dx_std,
        extra={
            "source_csv": csv_path,
            "expected_header": _expected_headers(),
        },
    )

    info = {
        "script": "surrogate_trainer.py",
        "csv": csv_path,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "hidden_dim": int(args.hidden_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "info": info,
        },
        model_path,
    )

    print("\n[保存]")
    print(f"  scaler: {scaler_path}")
    print(f"  model:  {model_path}")
    print(f"  val_mse (normalized dx): {metrics['val_mse_norm_dx']:.6e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
