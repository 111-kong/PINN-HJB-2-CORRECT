# -*- coding: utf-8 -*-
"""一次性：扫描 r_diag，使 PINN 控制力接近 scaler.json 中历史 u 量级。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from pinn_hjb_aqwa_integration import AQWASimulator  # noqa: E402
from pinn_hjb_config import PINNHJBConfig  # noqa: E402
from pinn_hjb_controller import PINNController, value_network_from_state_dict  # noqa: E402


def main() -> None:
    scaler = json.loads((BASE / "surrogate_model" / "scaler.json").read_text(encoding="utf-8"))
    u_mean = np.array(scaler["u_mean"], dtype=np.float64)
    u_std = np.array(scaler["u_std"], dtype=np.float64)

    sim = AQWASimulator(
        str(BASE / "surrogate_model" / "surrogate_model.pt"),
        str(BASE / "surrogate_model" / "scaler.json"),
    )
    ckpt = torch.load(
        str(BASE / "surrogate_model" / "value_network.pt"),
        map_location="cpu",
        weights_only=False,
    )
    vn = value_network_from_state_dict(ckpt["value_state_dict"])
    vn.eval()

    cols = [
        "η_surge(m)",
        "η_sway(m)",
        "η_yaw(rad)",
        "ν_surge(m/s)",
        "ν_sway(m/s)",
        "ν_yaw(rad/s)",
    ]
    rows = []
    with open(BASE / "pinn_hjb_aqwa_online_log.csv", newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            if i >= 40:
                break
            rows.append([float(row[c]) for c in cols])
    x_batch = np.array(rows, dtype=np.float64)

    def _np6(t):
        return t.detach().cpu().numpy().reshape(6)

    def stats_u(r_diag: tuple) -> tuple:
        cfg = PINNHJBConfig(r_diag=tuple(float(x) for x in r_diag))
        pin = PINNController(
            sim.model,
            vn,
            pinn_cfg=cfg,
            device="cpu",
            surrogate_x_mean=_np6(sim.x_mean),
            surrogate_x_std=_np6(sim.x_std),
            surrogate_u_std=_np6(sim.u_std),
            surrogate_dx_std=_np6(sim.dx_std),
        )
        u_list = []
        for x in x_batch:
            u = pin.compute_control(x, clip_ranges=cfg.clip_ranges(), mooring_regime=0.0)
            u_list.append(u)
        u_arr = np.array(u_list, dtype=np.float64)
        return u_arr.mean(0), np.median(np.abs(u_arr), 0), np.abs(u_arr).mean(0)

    print("u_mean (train CSV)", u_mean)
    print("u_std (train CSV) ", u_std)
    r0 = (1e-2, 1e-2, 1e-1)
    print("baseline r", r0, "->", stats_u(r0))
    best_s = None
    best_err = float("inf")
    scales = [
        500,
        5000,
        25000,
        100000,
        500000,
        2_000_000,
        5_000_000,
        8_000_000,
        12_000_000,
    ]
    for s in scales:
        r = (1e-2 / s, 1e-2 / s, 1e-1 / s)
        mu, _med, am = stats_u(r)
        err = float(np.sum((np.abs(mu) - np.abs(u_mean)) ** 2 / (u_std**2 + 1.0)))
        if err < best_err:
            best_err = err
            best_s = s
        print(f"scale 1/{s:12d}  r={r[0]:.3e} {r[1]:.3e} {r[2]:.3e}  u_mean={mu}  |u|_mean={am}")
    print("picked scale (rough MSE vs |u_mean| in std units):", best_s, "err", best_err)


if __name__ == "__main__":
    main()
