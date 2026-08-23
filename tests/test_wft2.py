# -*- coding: utf-8 -*-
"""掩膜自适应 Qian WFF 的确定性数值回归。"""

import numpy as np

from _paths import SRC_DIR  # noqa: F401 - ensures src/ is importable
from fizeau_gui.core import WFT2Result, wft2_phase


def wrapped_rms(estimated: np.ndarray, expected: np.ndarray,
                mask: np.ndarray, margin: int = 24) -> float:
    """去掉不可观测 piston 后计算孔径内部包裹相位 RMS。"""
    region = mask.copy()
    region[:margin] = False
    region[-margin:] = False
    region[:, :margin] = False
    region[:, -margin:] = False
    difference = np.angle(np.exp(1j * (estimated - expected)))
    piston = np.angle(np.mean(np.exp(1j * difference[region])))
    error = np.angle(np.exp(1j * (difference - piston)))
    return float(np.sqrt(np.mean(error[region] ** 2)))


def synthetic_fringe(period: int, direction: str):
    size = max(160, 4 * period)
    y, x = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    mask = ((x - center) ** 2 + (y - center) ** 2
            < (0.44 * size) ** 2)
    if direction == "x":
        carrier = 2.0 * np.pi * x / period
    elif direction == "y":
        carrier = 2.0 * np.pi * y / period
    else:
        carrier = 2.0 * np.pi * (x / period + y / (3.0 * period))
    phase = (carrier + 0.25 * np.sin(2.0 * np.pi * y / size)
             + 0.12 * np.cos(2.0 * np.pi * x / size))
    background = 95.0 + 30.0 * x / size + 18.0 * y / size
    image = np.zeros_like(phase)
    image[mask] = (background + 65.0 * np.cos(phase))[mask]
    return image, phase, mask


def main():
    for direction in ("x", "y", "slanted"):
        image, truth, mask = synthetic_fringe(15, direction)
        result = wft2_phase(
            image, 15, mask=mask, return_result=True)
        assert isinstance(result, WFT2Result)
        error = wrapped_rms(result.phase_wrapped, truth, mask)
        print(f"周期 15 / {direction}: wrapped RMS = {error:.6f} rad")
        assert error < 0.02
        assert np.isfinite(result.phase_wrapped).all()
        assert np.all(result.confidence[~mask] == 0.0)
        assert 0.0 < float(np.mean(result.confidence[mask])) <= 1.0

    # 固定 ±0.3 rad/pixel 搜索带会在长周期时跨过 DC；自适应频带必须
    # 保持在同一共轭半平面，并显著低于旧实现约 0.77 rad 的误差。
    image, truth, mask = synthetic_fringe(50, "slanted")
    result = wft2_phase(image, 50, mask=mask, return_result=True)
    long_error = wrapped_rms(result.phase_wrapped, truth, mask)
    print(f"周期 50 / slanted: wrapped RMS = {long_error:.6f} rad")
    assert long_error < 0.08
    assert result.frequency_band < np.hypot(*result.carrier_rad) / 2.0

    try:
        wft2_phase(np.zeros((64, 64)), 15)
    except ValueError as exc:
        assert "调制度" in str(exc)
    else:
        raise AssertionError("常量图像没有被 WFT 拒绝")

    try:
        wft2_phase(np.ones((64, 64)), 15, mask=np.ones((32, 32)))
    except ValueError as exc:
        assert "尺寸一致" in str(exc)
    else:
        raise AssertionError("错误尺寸 mask 没有被 WFT 拒绝")

    print("WFT2 TEST PASSED")


if __name__ == "__main__":
    main()
