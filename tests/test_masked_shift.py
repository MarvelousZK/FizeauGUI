# -*- coding: utf-8 -*-
"""验证 ChatGPT 新增的 masked_s_T_shift 与全口径统计的正确性。"""
import os
import numpy as np
from _paths import DATA_DIR
from fizeau_gui.core import (zStd, s_T_shift, masked_s_T_shift,
                             auto_detect_circle, build_mask,
                             read_image, process_fizeau)
from PIL import Image

OUT = str(DATA_DIR)
REF = os.path.join(OUT, "仿真_参考元件.bmp")
TEST = os.path.join(OUT, "仿真_待测元件.bmp")

IMAGE, CX, CY, RADIUS, LAMBDA = 720, 360.0, 360.0, 270.0, 635.0
ABER_REF = {4: 5.0, 6: 4.0, 8: 3.0}
ABER_TEST = {4: 40.0, 5: 30.0, 6: 25.0, 7: 20.0, 9: 12.0, 11: 15.0}

N = IMAGE
X, Y = np.meshgrid(np.arange(N, dtype=np.float64), np.arange(N, dtype=np.float64))
u, v = (X - CX) / RADIUS, (Y - CY) / RADIUS
rho, theta = np.hypot(u, v), np.arctan2(v, u)
mask = rho <= 1.0
Zfull, _, _, _ = zStd(np.clip(rho, 0, 1).ravel(), theta.ravel(), 11)
Zfull = Zfull.reshape(N, N, 11)

def surface(amps):
    h = np.zeros((N, N))
    for j, a in amps.items():
        h += a * Zfull[:, :, j - 1]
    return h * mask

h_diff = (surface(ABER_REF) - surface(ABER_TEST)) * mask

I_test = np.asarray(Image.open(TEST).convert("L"), dtype=np.float64)

print("=== 1. 孔径内部 masked_s_T_shift 与原 s_T_shift 一致性 ===")
info = build_mask((N, N), 360.0, 360.0, 268)
rmin, rmax, cmin, cmax = info.rowmin, info.rowmax, info.colmin, info.colmax
Imd = I_test[rmin:rmax + 1, cmin:cmax + 1]
m = info.mask
pd_old = s_T_shift(Imd, 15, 1, 21)
pd_new = masked_s_T_shift(Imd, m, 15, 1, 21)
# 严格内部：离每行弦两端各 16px 以内不算（旧算法在那里会用到孔径外像素）
from scipy.ndimage import binary_erosion
inner = binary_erosion(m, structure=np.ones((1, 33), dtype=bool))
d = np.angle(np.exp(1j * (pd_old - pd_new)))[inner]
print(f"  严格内部差异: max={np.abs(d).max():.3e} rad, 应≈0")

print("\n=== 2. 边缘覆盖 ===")
print(f"  原算法孔径内零值像素: {int(((pd_old == 0) & m).sum())}")
print(f"  新算法孔径内零值像素: {int(((pd_new == 0) & m).sum())}")

print("\n=== 3. 新算法在仿真数据上的全口径面形恢复 ===")
det = auto_detect_circle(read_image(REF))
res = process_fizeau(REF, TEST, 15, *det)
m1 = res.mask_info.mask
W_rec = res.residuals[0]["W"]
W_truth = h_diff[res.mask_info.rowmin:res.mask_info.rowmax + 1,
                 res.mask_info.colmin:res.mask_info.colmax + 1] * m1
d = W_rec[m1] - W_truth[m1]
print(f"  恢复面形 vs 真实面形 (全口径, 去平移): RMS 误差 {np.std(d):.2f} nm")
print(f"  真实面形(全口径): RMS={W_truth[m1].std():.2f} nm, "
      f"PV={W_truth[m1].max()-W_truth[m1].min():.2f} nm")
print(f"  软件输出: 全局 RMS={res.global_rms:.2f} nm, PV={res.global_pv:.2f} nm")
for r_ in res.residuals[1:4]:
    print(f"    去前{r_['k']}项: RMS={r_['rms']:.2f} nm, PV={r_['pv']:.2f} nm")
assert np.std(d) < 5.0, "全口径恢复误差过大!"
print("\nmasked_s_T_shift 验证通过")
