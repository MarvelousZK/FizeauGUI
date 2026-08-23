# -*- coding: utf-8 -*-
"""Luo 单载频 adapt2 的确定性数值回归。"""

import numpy as np
from scipy.ndimage import binary_erosion

from _paths import SRC_DIR  # noqa: F401 - ensures src/ is importable
from fizeau_gui.core import _luo_dominant_axis, adapt2_phase


def wrapped_rms(estimated: np.ndarray, expected: np.ndarray,
                margin: int = 20) -> float:
    """去除不可观测的常相位后，计算截断相位圆周 RMS。"""
    region = np.s_[margin:-margin, margin:-margin]
    difference = np.angle(np.exp(1j * (estimated - expected)))
    piston = np.angle(np.mean(np.exp(1j * difference[region])))
    error = np.angle(np.exp(1j * (difference - piston)))
    return float(np.sqrt(np.mean(error[region] ** 2)))


def synthetic_fringe(axis: int, noise: float = 0.0, seed: int = 7):
    size = 128
    period = 15
    y, x = np.mgrid[:size, :size]
    wavefront = (0.30 * np.sin(2.0 * np.pi * y / size)
                 + 0.18 * np.cos(2.0 * np.pi * x / size)
                 + 0.00035 * (x - size / 2.0) * (y - size / 2.0))
    carrier = 2.0 * np.pi * (x if axis == 1 else y) / period
    total_phase = carrier + wavefront
    rng = np.random.default_rng(seed)
    image = 120.0 + 80.0 * np.cos(total_phase)
    image += noise * 80.0 * rng.standard_normal(image.shape)
    return image, total_phase, period


def main():
    for axis, name in ((1, "x"), (0, "y")):
        image, truth, period = synthetic_fringe(axis)
        assert _luo_dominant_axis(image) == axis, f"{name} 主载频方向判断错误"
        estimated = adapt2_phase(image, period)
        # adapt2_phase 保留原工程 phasex=-phasex 的符号约定。
        error = wrapped_rms(-estimated, truth)
        print(f"理想 {name} 载频: wrapped RMS = {error:.6f} rad")
        assert error < 0.003

    image, truth, period = synthetic_fringe(1, noise=0.03)
    noisy = adapt2_phase(image, period)
    noise_error = wrapped_rms(-noisy, truth)
    print(f"3% 高斯噪声: wrapped RMS = {noise_error:.6f} rad")
    assert noise_error < 0.10

    # 让局部周期随列位置缓慢变化，验证增量不是固定 2pi/T。
    size = 128
    period = 15
    y, x = np.mgrid[:size, :size]
    chirped_phase = (2.0 * np.pi * x / period
                     + 0.00045 * (x - size / 2.0) ** 2
                     + 0.25 * np.sin(2.0 * np.pi * y / size))
    chirped_image = 110.0 + 75.0 * np.cos(chirped_phase)
    chirped = adapt2_phase(chirped_image, period)
    chirp_error = wrapped_rms(-chirped, chirped_phase)
    print(f"缓变局部周期: wrapped RMS = {chirp_error:.6f} rad")
    assert chirp_error < 0.01

    # 圆口径外设为黑色，专门回归旧实现最明显的边缘污染问题。
    size = 160
    y, x = np.mgrid[:size, :size]
    circular_mask = ((x - (size - 1) / 2.0) ** 2
                     + (y - (size - 1) / 2.0) ** 2 <= 65.0 ** 2)
    circular_truth = (
        2.0 * np.pi * x / period
        + 0.30 * np.sin(2.0 * np.pi * y / size)
        + 0.18 * np.cos(2.0 * np.pi * x / size)
        + 0.0002 * (x - size / 2.0) * (y - size / 2.0))
    circular_image = np.zeros((size, size), dtype=np.float64)
    circular_image[circular_mask] = (
        120.0 + 80.0 * np.cos(circular_truth[circular_mask]))
    circular = adapt2_phase(circular_image, period, mask=circular_mask)
    difference = np.angle(np.exp(1j * (-circular - circular_truth)))
    piston = np.angle(np.mean(np.exp(1j * difference[circular_mask])))
    circular_error = np.angle(np.exp(1j * (difference - piston)))
    rim = circular_mask & ~binary_erosion(circular_mask, iterations=8)
    full_rms = float(np.sqrt(np.mean(circular_error[circular_mask] ** 2)))
    rim_rms = float(np.sqrt(np.mean(circular_error[rim] ** 2)))
    print(f"圆口径全局/边缘: wrapped RMS = "
          f"{full_rms:.6f}/{rim_rms:.6f} rad")
    assert np.isfinite(circular[circular_mask]).all()
    assert full_rms < 0.02 and rim_rms < 0.05

    try:
        adapt2_phase(np.zeros((10, 10)), period)
    except ValueError:
        pass
    else:
        raise AssertionError("小于增量窗口的图像未被拒绝")

    print("LUO ADAPT2 TEST PASSED")


if __name__ == "__main__":
    main()
