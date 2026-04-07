#!/usr/bin/env python3
"""Real-time plot of env_0 state during training.

Reads binary log written by junpu_place_workpiece._compute_reward.
X-axis = sample index (sequential data points), rolling window of last N points.

Usage:
    python plot_env0_live.py [--window 300]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = "/tmp/geniesim_logs/env0_state_log.bin"
COLS = [
    "timestamp", "step",
    "wp_x", "wp_y", "wp_z",
    "ee_x", "ee_y", "ee_z",
    "ee_roll", "ee_pitch", "ee_yaw",
    "ee_vx", "ee_vy", "ee_vz",
    "ee_wx", "ee_wy", "ee_wz",
    "dist_3d", "dist_xy", "diff_z",
    "orient_diff",
    "reward",
    "r_alive", "r_approach",
    "r_orient_approach", "r_below",
    "r_success",
]
COL = {name: idx for idx, name in enumerate(COLS)}
NUM_COLS = len(COLS)
ROW_BYTES = NUM_COLS * 4


def load_data(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        return np.empty((0, NUM_COLS), dtype=np.float32)
    raw = p.read_bytes()
    n = len(raw) // ROW_BYTES
    if n == 0:
        return np.empty((0, NUM_COLS), dtype=np.float32)
    return np.frombuffer(raw[: n * ROW_BYTES], dtype=np.float32).reshape(n, NUM_COLS)


def make_line(ax, label, color, style="-"):
    ln, = ax.plot([], [], style, color=color, label=label, linewidth=1.0)
    return ln


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=300,
                        help="Number of recent data points to show")
    args = parser.parse_args()
    win = args.window

    plt.ion()
    fig, axes = plt.subplots(7, 1, figsize=(16, 16), sharex=True)
    fig.suptitle("env_0 live state monitor", fontsize=13)
    fig.subplots_adjust(hspace=0.28)

    ax_pos, ax_rpy, ax_linvel, ax_angvel, ax_dist, ax_rtotal, ax_rdetail = axes

    ax_pos.set_ylabel("EE Pos (m)")
    ax_rpy.set_ylabel("EE RPY (rad)")
    ax_linvel.set_ylabel("EE LinVel (m/s)")
    ax_angvel.set_ylabel("EE AngVel (rad/s)")
    ax_dist.set_ylabel("Distance")
    ax_rtotal.set_ylabel("Total Reward")
    ax_rdetail.set_ylabel("Reward Detail")
    ax_rdetail.set_xlabel("Sample #")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    L = {}

    L["ee_x"]  = make_line(ax_pos, "ee_x",  "tab:red")
    L["ee_y"]  = make_line(ax_pos, "ee_y",  "tab:green")
    L["ee_z"]  = make_line(ax_pos, "ee_z",  "tab:blue")
    L["wp_x"]  = make_line(ax_pos, "wp_x",  "tab:red",   "--")
    L["wp_y"]  = make_line(ax_pos, "wp_y",  "tab:green",  "--")
    L["wp_z"]  = make_line(ax_pos, "wp_z",  "tab:blue",   "--")

    L["ee_roll"]  = make_line(ax_rpy, "roll",  "tab:orange")
    L["ee_pitch"] = make_line(ax_rpy, "pitch", "tab:purple")
    L["ee_yaw"]   = make_line(ax_rpy, "yaw",   "tab:cyan")

    L["ee_vx"] = make_line(ax_linvel, "vx", "tab:red")
    L["ee_vy"] = make_line(ax_linvel, "vy", "tab:green")
    L["ee_vz"] = make_line(ax_linvel, "vz", "tab:blue")

    L["ee_wx"] = make_line(ax_angvel, "wx", "tab:orange")
    L["ee_wy"] = make_line(ax_angvel, "wy", "tab:purple")
    L["ee_wz"] = make_line(ax_angvel, "wz", "tab:cyan")

    L["dist_3d"]     = make_line(ax_dist, "dist_3d",     "tab:red")
    L["dist_xy"]     = make_line(ax_dist, "dist_xy",     "tab:green")
    L["diff_z"]      = make_line(ax_dist, "diff_z",      "tab:blue")
    L["orient_diff"] = make_line(ax_dist, "orient_diff", "tab:orange")

    L["reward"] = make_line(ax_rtotal, "reward", "tab:brown")

    L["r_alive"]   = make_line(ax_rdetail, "r_alive",   "tab:green")
    L["r_approach"] = make_line(ax_rdetail, "r_approach", "tab:blue")
    L["r_orient_approach"] = make_line(ax_rdetail, "r_oappr", "tab:orange")
    L["r_below"]   = make_line(ax_rdetail, "r_below",   "tab:red")
    L["r_success"] = make_line(ax_rdetail, "r_success", "tab:purple")

    for ax in axes:
        ax.legend(loc="upper left", fontsize=7, ncol=3)

    prev_len = 0

    while True:
        data = load_data(LOG_PATH)
        if len(data) == 0 or len(data) == prev_len:
            plt.pause(0.5)
            continue
        prev_len = len(data)

        d = data[-win:]
        x = np.arange(max(0, len(data) - win), len(data))

        L["ee_x"].set_data(x, d[:, COL["ee_x"]])
        L["ee_y"].set_data(x, d[:, COL["ee_y"]])
        L["ee_z"].set_data(x, d[:, COL["ee_z"]])
        L["wp_x"].set_data(x, d[:, COL["wp_x"]])
        L["wp_y"].set_data(x, d[:, COL["wp_y"]])
        L["wp_z"].set_data(x, d[:, COL["wp_z"]])

        L["ee_roll"].set_data(x, d[:, COL["ee_roll"]])
        L["ee_pitch"].set_data(x, d[:, COL["ee_pitch"]])
        L["ee_yaw"].set_data(x, d[:, COL["ee_yaw"]])

        L["ee_vx"].set_data(x, d[:, COL["ee_vx"]])
        L["ee_vy"].set_data(x, d[:, COL["ee_vy"]])
        L["ee_vz"].set_data(x, d[:, COL["ee_vz"]])

        L["ee_wx"].set_data(x, d[:, COL["ee_wx"]])
        L["ee_wy"].set_data(x, d[:, COL["ee_wy"]])
        L["ee_wz"].set_data(x, d[:, COL["ee_wz"]])

        L["dist_3d"].set_data(x, d[:, COL["dist_3d"]])
        L["dist_xy"].set_data(x, d[:, COL["dist_xy"]])
        L["diff_z"].set_data(x, d[:, COL["diff_z"]])
        L["orient_diff"].set_data(x, d[:, COL["orient_diff"]])

        L["reward"].set_data(x, d[:, COL["reward"]])

        L["r_alive"].set_data(x, d[:, COL["r_alive"]])
        L["r_approach"].set_data(x, d[:, COL["r_approach"]])
        L["r_orient_approach"].set_data(x, d[:, COL["r_orient_approach"]])
        L["r_below"].set_data(x, d[:, COL["r_below"]])
        L["r_success"].set_data(x, d[:, COL["r_success"]])

        for ax in axes:
            ax.set_xlim(x[0] - 1, x[-1] + 1)
            ax.relim()
            ax.autoscale_view(scalex=False)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.5)


if __name__ == "__main__":
    main()
