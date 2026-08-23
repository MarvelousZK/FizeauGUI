# -*- coding: utf-8 -*-
"""质量图引导展开的性能与全流程验证。"""
import os
import time

import numpy as np

from _paths import DATA_DIR

from fizeau_gui.core import (read_image, auto_detect_circle, build_mask,
                             s_T_shift, qg_dunwrap,
                             phase_derivative_variance, process_fizeau)

DATA = str(DATA_DIR)
REF = os.path.join(DATA, "仿真_参考元件.bmp")
TEST = os.path.join(DATA, "仿真_待测元件.bmp")

img = read_image(REF)
cx, cy, r = auto_detect_circle(img)
info = build_mask(img.shape, cx, cy, int(r))
rmin, rmax, cmin, cmax = info.rowmin, info.rowmax, info.colmin, info.colmax
mask = info.mask

Imd = read_image(TEST)[rmin:rmax + 1, cmin:cmax + 1]
print(f"ROI: {Imd.shape}  掩膜像素数: {int(mask.sum())}")

pd = s_T_shift(Imd, 15, 1, 21)

t0 = time.perf_counter()
qmap = phase_derivative_variance(pd, 3)
t_q = time.perf_counter() - t0
print(f"PDV 质量图计算     : {t_q*1000:8.1f} ms")

t0 = time.perf_counter()
u_qg = qg_dunwrap(pd, mask, quality=qmap)
t_qg = time.perf_counter() - t0
print(f"质量图引导(含质图) : {(t_q+t_qg)*1000:8.1f} ms  (纯展开 {t_qg*1000:.1f} ms)")

assert np.all(np.isfinite(u_qg[mask])), "有效孔径内出现非有限相位值"

# ---- 全流程耗时 ----
t0 = time.perf_counter()
res = process_fizeau(REF, TEST, 15, cx, cy, int(r))
t_all = time.perf_counter() - t0
print(f"\n全流程(质量图引导, 36项): {t_all:.2f} s")
print(f"全局 RMS={res.global_rms:.2f} nm  PV={res.global_pv:.2f} nm")
for r_ in res.residuals[:4]:
    print(f"  k={r_['k']}  RMS={r_['rms']:.2f}  PV={r_['pv']:.2f}")
