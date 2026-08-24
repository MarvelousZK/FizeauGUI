# -*- coding: utf-8 -*-
"""绘制前 11 项 Noll Zernike 模式的独立示例程序。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


MODE_NAMES = (
    "Piston",
    "Tilt X (0°)",
    "Tilt Y (90°)",
    "Defocus",
    "Astigmatism 45°",
    "Astigmatism 0°",
    "Coma Y",
    "Coma X",
    "Trefoil 30°",
    "Trefoil 0°",
    "Primary spherical",
)


def first_11_noll(size: int = 301) -> tuple[np.ndarray, np.ndarray]:
    """返回前 11 项归一化 Noll Zernike 及单位圆掩膜。"""
    if size < 51:
        raise ValueError("size 至少应为 51")

    coord = np.linspace(-1.0, 1.0, int(size), dtype=np.float64)
    x, y = np.meshgrid(coord, coord)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    mask = rho <= 1.0

    modes = np.stack(
        [
            np.ones_like(rho),
            2.0 * rho * np.cos(theta),
            2.0 * rho * np.sin(theta),
            np.sqrt(3.0) * (2.0 * rho**2 - 1.0),
            np.sqrt(6.0) * rho**2 * np.sin(2.0 * theta),
            np.sqrt(6.0) * rho**2 * np.cos(2.0 * theta),
            np.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.sin(theta),
            np.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * np.cos(theta),
            np.sqrt(8.0) * rho**3 * np.sin(3.0 * theta),
            np.sqrt(8.0) * rho**3 * np.cos(3.0 * theta),
            np.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0),
        ],
        axis=0,
    )
    modes[:, ~mask] = np.nan
    return modes, mask


def plot_modes(
    size: int = 301,
    amplitude: float = 1.0,
) -> tuple[plt.Figure, np.ndarray]:
    """生成 4×3 排版的前 11 项 Noll Zernike 图。"""
    if amplitude <= 0.0:
        raise ValueError("amplitude 必须大于 0")
    modes, _ = first_11_noll(size)
    modes *= float(amplitude)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    cmap = plt.colormaps["coolwarm"].copy()
    cmap.set_bad("#f1f3f5")

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(12.6, 15.2),
        constrained_layout=True,
        facecolor="white",
    )
    fig.suptitle("First 11 Noll Zernike modes", fontsize=16, weight="bold")

    circle = np.linspace(0.0, 2.0 * np.pi, 500)
    for index, (ax, mode, name) in enumerate(
        zip(axes.flat, modes, MODE_NAMES), start=1
    ):
        valid = mode[np.isfinite(mode)]
        if index == 1:
            vmin, vmax = 0.0, float(amplitude)
            if vmax == 0.0:
                vmax = 1.0
        else:
            limit = float(np.max(np.abs(valid)))
            if limit == 0.0:
                limit = 1.0
            vmin, vmax = -limit, limit

        image = ax.imshow(
            mode,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.plot(np.cos(circle), np.sin(circle), color="#374151", linewidth=0.7)
        ax.set_title(f"Z{index}  ·  {name}", pad=7)
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("x / R")
        ax.set_ylabel("y / R")
        ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        ax.tick_params(direction="in", length=3, width=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#4b5563")

        colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.025)
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
        colorbar.outline.set_linewidth(0.6)
        colorbar.set_label("Normalized amplitude", labelpad=5)

    axes.flat[-1].axis("off")
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 coolwarm 绘制前 11 项 Noll Zernike 模式。"
    )
    parser.add_argument("--size", type=int, default=301, help="每项的采样尺寸")
    parser.add_argument(
        "--amplitude", type=float, default=1.0, help="所有模式的统一幅值系数"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zernike_noll_coolwarm.png"),
        help="输出 PNG/PDF/SVG 路径",
    )
    parser.add_argument("--dpi", type=int, default=220, help="位图输出分辨率")
    parser.add_argument("--show", action="store_true", help="保存后显示绘图窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig, _ = plot_modes(size=args.size, amplitude=args.amplitude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"已保存：{args.output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
