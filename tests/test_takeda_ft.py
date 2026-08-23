# -*- coding: utf-8 -*-
"""经典 Takeda FT 核心与完整流水线回归测试。"""

import os

import numpy as np

from _paths import DATA_DIR
from fizeau_gui.core import (
    auto_detect_circle,
    build_report,
    process_fizeau,
    read_image,
    takeda_ft_phase,
    takeda_ft_spectrum,
)


def synthetic_case(size=256, noise=0.015, seed=7):
    yy, xx = np.mgrid[:size, :size]
    x = (xx - (size - 1) / 2) / (size / 2)
    y = (yy - (size - 1) / 2) / (size / 2)
    radius = np.hypot(x, y)
    mask = radius <= 0.90
    evaluation = radius <= 0.65
    phase = (0.70 * (2 * radius ** 2 - 1)
             + 0.25 * x * y
             + 0.20 * np.exp(-((x + 0.20) ** 2 + (y - 0.15) ** 2) / 0.04))
    carrier = (24, 5)
    rng = np.random.default_rng(seed)
    image = (0.50 + 0.40 * np.cos(
        phase + 2 * np.pi * (carrier[0] * xx / size
                             + carrier[1] * yy / size))
             + noise * rng.normal(size=(size, size)))
    image[~mask] = 0.50
    return image, mask, evaluation, phase, carrier


def wrapped_rms(recovered, truth, evaluation):
    error = np.angle(np.exp(1j * (recovered - truth)))[evaluation]
    error -= np.mean(error)
    return float(np.sqrt(np.mean(error ** 2)))


image, mask, evaluation, truth, carrier = synthetic_case()

print("=== 1. 手动教学频谱预览 ===")
preview = takeda_ft_spectrum(image, mask=mask, apply_hann=True)
assert preview.spectrum_log.shape == image.shape
assert preview.spectrum.shape == image.shape
assert preview.window.shape == image.shape
assert np.iscomplexobj(preview.spectrum)

print("=== 2. 自动边带 Takeda FT（对照） ===")
auto = takeda_ft_phase(image, mask=mask)
assert auto.carrier_cycles == carrier, auto.carrier_cycles
assert auto.sideband_center == (image.shape[0] // 2 + carrier[1],
                                image.shape[1] // 2 + carrier[0])
assert auto.spectrum_log.shape == image.shape
assert auto.sideband_filter.shape == image.shape
assert auto.filtered_spectrum_log.shape == image.shape
assert np.isnan(auto.amplitude[~mask]).all()
assert np.isnan(auto.confidence[~mask]).all()
assert np.isfinite(auto.confidence[mask]).all()
assert 0.0 <= np.nanmin(auto.confidence) <= np.nanmax(auto.confidence) <= 1.0
auto_rms = wrapped_rms(auto.phase_wrapped, truth, evaluation)
assert auto_rms < 0.05, auto_rms
print(f"自动检测载频={auto.carrier_cycles}, 相位 RMS={auto_rms:.6f} rad")

assert np.allclose(auto.spectrum_log, preview.spectrum_log)

print("=== 3. 手动载频、滤波宽度与符号 ===")
manual = takeda_ft_phase(
    image, mask=mask, carrier_cycles=carrier, filter_sigma=8.0,
    center_exclusion_radius=18, apply_hann=True, phase_sign=1)
manual_rms = wrapped_rms(manual.phase_wrapped, truth, evaluation)
assert manual_rms < 0.03, manual_rms
negative = takeda_ft_phase(
    image, mask=mask, carrier_cycles=carrier, filter_sigma=8.0,
    apply_hann=True, phase_sign=-1)
assert np.allclose(negative.phase_wrapped, -manual.phase_wrapped)
wide = takeda_ft_phase(
    image, mask=mask, carrier_cycles=carrier, filter_sigma=15.0)
assert wide.filter_too_wide
print(f"手动载频 RMS={manual_rms:.6f} rad，宽滤波警告={wide.filter_too_wide}")

print("=== 4. 项目仿真数据完整流水线 ===")
data = str(DATA_DIR)
ref = os.path.join(data, "仿真_参考元件.bmp")
test = os.path.join(data, "仿真_待测元件.bmp")
cx, cy, radius = auto_detect_circle(read_image(ref))
result = process_fizeau(
    ref, test, 15, cx, cy, int(radius), max_term=20, n_remove=4,
    phase_method="takeda")
assert result.phase_method == "takeda"
assert result.ft_diagnostics is not None
assert result.ft_diagnostics["test"].carrier_cycles == (36, 0)
assert result.ft_diagnostics["reference"].carrier_cycles == (36, 0)
assert result.global_pv < 400.0, result.global_pv
report = build_report(result)
assert "经典 Fourier-transform 法" in report
assert "FT Gaussian" in report
print(f"流水线 RMS={result.global_rms:.3f} nm, PV={result.global_pv:.3f} nm")

print("\nTAKEDA FT TEST PASSED")
