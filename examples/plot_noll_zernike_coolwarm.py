# -*- coding: utf-8 -*-
"""以无标注塔形布局绘制 Noll Zernike 模式。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def noll_indices(j: int) -> tuple[int, int]:
    """返回 Noll 序号 j 对应的径向阶数 n 和角向频率 |m|。"""
    if j < 1:
        raise ValueError("Noll 序号 j 必须从 1 开始")

    n = 0
    while j > (n + 1) * (n + 2) // 2:
        n += 1

    cursor = n * (n + 1) // 2
    for m in range(n + 1):
        if (n - m) % 2 != 0:
            continue
        cursor += 1
        if cursor == j:
            return n, m
        if m != 0:
            cursor += 1
            if cursor == j:
                return n, m
    raise RuntimeError(f"无法确定 Z{j} 的 Noll 指标")


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """计算 Zernike 径向多项式 R_n^m。"""
    radial = np.zeros_like(rho, dtype=np.float64)
    for s in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** s
            * math.factorial(n - s)
            / (
                math.factorial(s)
                * math.factorial((n + m) // 2 - s)
                * math.factorial((n - m) // 2 - s)
            )
        )
        radial += coefficient * rho ** (n - 2 * s)
    return radial


def noll_mode(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """计算与软件 zStd 相同归一化及正弦/余弦约定的 Noll Zernike。"""
    n, m = noll_indices(j)
    radial = radial_polynomial(n, m, rho)
    if m == 0:
        return math.sqrt(n + 1) * radial

    normalization = math.sqrt(2.0 * (n + 1))
    angular = np.cos(m * theta) if j % 2 == 0 else np.sin(m * theta)
    return normalization * radial * angular


def signed_m(j: int, m: int) -> int:
    """用正 m 表示余弦项、负 m 表示正弦项。"""
    if m == 0:
        return 0
    return m if j % 2 == 0 else -m


def plot_tower(
    max_n: int = 4,
    size: int = 301,
    cmap_name: str = "coolwarm",
) -> plt.Figure:
    """从 n=0 开始按同阶同行的塔形布局绘制 Zernike 模式。"""
    if max_n < 1:
        raise ValueError("max_n 必须大于等于 1")
    if size < 51:
        raise ValueError("size 至少应为 51")

    coordinate = np.linspace(-1.0, 1.0, int(size), dtype=np.float64)
    x, y = np.meshgrid(coordinate, coordinate)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    aperture = rho <= 1.0

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    row_count = max_n + 1
    widest_row = max_n + 1
    fig = plt.figure(
        figsize=(2.65 * widest_row, 2.0 * row_count),
        facecolor="white",
    )

    horizontal_margin = 0.018
    vertical_margin = 0.025
    row_gap = 0.012
    cell_gap = 0.008
    usable_height = 1.0 - 2.0 * vertical_margin - row_gap * (row_count - 1)
    cell_height = usable_height / row_count
    widest_width = 1.0 - 2.0 * horizontal_margin
    cell_width = (widest_width - cell_gap * (widest_row - 1)) / widest_row

    for row_index, n in enumerate(range(0, max_n + 1)):
        count = n + 1
        row_width = count * cell_width + (count - 1) * cell_gap
        row_left = 0.5 - row_width / 2.0
        bottom = (
            1.0
            - vertical_margin
            - (row_index + 1) * cell_height
            - row_index * row_gap
        )
        first_j = n * (n + 1) // 2 + 1

        for column_index in range(count):
            j = first_j + column_index
            _, m_abs = noll_indices(j)
            m = signed_m(j, m_abs)
            mode = noll_mode(j, rho, theta)
            mode = np.where(aperture, mode, np.nan)
            limit = float(np.nanmax(np.abs(mode)))

            left = row_left + column_index * (cell_width + cell_gap)
            image_width = 0.75 * cell_width
            ax = fig.add_axes((left, bottom, image_width, cell_height))
            ax.imshow(
                mode,
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                interpolation="bilinear",
            )
            ax.set_axis_off()
            ax.set_aspect("equal")

            m_text = "0" if m == 0 else f"{m:+d}".replace("-", "−")
            fig.text(
                left + 0.79 * cell_width,
                bottom + 0.50 * cell_height,
                f"$n={n}$\n$m={m_text}$\n$j={j}$",
                ha="left",
                va="center",
                fontsize=max(7.0, 11.5 - 0.75 * (max_n - 4)),
                linespacing=1.45,
                color="#374151",
            )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 n=0 开始，以塔形布局绘制 Noll Zernike。"
    )
    parser.add_argument("--max-n", type=int, default=4, help="绘制到的最大径向阶数")
    parser.add_argument("--size", type=int, default=301, help="每个圆的采样尺寸")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zernike_noll_tower.png"),
        help="输出 PNG/PDF/SVG 路径",
    )
    parser.add_argument("--dpi", type=int, default=220, help="位图输出分辨率")
    parser.add_argument("--show", action="store_true", help="保存后显示绘图窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = plot_tower(max_n=args.max_n, size=args.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    print(f"已保存：{args.output.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
