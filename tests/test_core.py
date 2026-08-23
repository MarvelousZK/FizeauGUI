# -*- coding: utf-8 -*-
"""核心模块验证脚本：跑通真实数据并核对基础计算。"""
import os
import sys

import numpy as np
from scipy.io import loadmat

from _paths import DATA_DIR, PROJECT_ROOT

LEGACY_DATA = os.path.join(PROJECT_ROOT.parent, "Fizeau interferometer Code")
HAS_LEGACY = os.path.isfile(os.path.join(LEGACY_DATA, "20231031-162217.bmp"))
if HAS_LEGACY:
    sys.path.insert(0, os.path.join(LEGACY_DATA, "test"))

from fizeau_gui.core import (read_image, auto_detect_circle, build_mask,
                             process_fizeau, build_report, zernike_name,
                             s_T_shift, zStd)

# 锁定 Noll 表 I 的前 15 项，避免编号、正弦/余弦分支与像差名称再次漂移。
_, n_noll, m_noll, j_noll = zStd(np.array([0.5]), np.array([0.3]), 15)
assert list(zip(j_noll, n_noll, m_noll)) == [
    (1, 0, 0), (2, 1, 1), (3, 1, 1), (4, 2, 0), (5, 2, 2),
    (6, 2, 2), (7, 3, 1), (8, 3, 1), (9, 3, 3), (10, 3, 3),
    (11, 4, 0), (12, 4, 2), (13, 4, 2), (14, 4, 4), (15, 4, 4),
]
assert "Coma Y" in zernike_name(7) and "Coma X" in zernike_name(8)
assert "Tetrafoil" in zernike_name(14) and "Tetrafoil" in zernike_name(15)

if HAS_LEGACY:
    DATA = LEGACY_DATA
    REF = os.path.join(DATA, "20231031-162217.bmp")
    TEST = os.path.join(DATA, "20231031-162300.bmp")
else:
    DATA = str(DATA_DIR)
    REF = os.path.join(DATA, "仿真_参考元件.bmp")
    TEST = os.path.join(DATA, "仿真_待测元件.bmp")

print("=== 1. 读图与自动检测孔径 ===")
img_ref = read_image(REF)
img_test = read_image(TEST)
print("参考图:", img_ref.shape, "待测图:", img_test.shape)
det = auto_detect_circle(img_ref)
print("自动检测: 中心(col,row)=({:.1f},{:.1f}) 半径={:.1f}".format(*det))

cx_auto, cy_auto, r_auto = det
info = build_mask(img_ref.shape, cx_auto, cy_auto, int(r_auto))
print(f"新版 ROI: 行 {info.rowmin}~{info.rowmax}, 列 {info.colmin}~{info.colmax}, 半径 {info.maskr}")

if HAS_LEGACY:
    print("\n=== 2. 与原版 mask.mat 对照 ROI ===")
    try:
        mat = loadmat(os.path.join(DATA, "mask.mat"))
        rowmin = int(mat['rowmin'].ravel()[0]); rowmax = int(mat['rowmax'].ravel()[0])
        colmin = int(mat['colmin'].ravel()[0]); colmax = int(mat['colmax'].ravel()[0])
        mask_orig = mat['mask'].astype(bool)
    except NotImplementedError:
        import pickle
        with open(os.path.join(DATA, "test", "mask.pkl"), "rb") as f:
            data = pickle.load(f)
        rowmin = int(data['rowmin'].ravel()[0]); rowmax = int(data['rowmax'].ravel()[0])
        colmin = int(data['colmin'].ravel()[0]); colmax = int(data['colmax'].ravel()[0])
        mask_orig = data['mask'].astype(bool)
        print("使用 test/mask.pkl 备份")
    print(f"原版 ROI: 行 {rowmin}~{rowmax}, 列 {colmin}~{colmax}")

    print("\n=== 3. 算法 A/B 对照 (原版 test 文件夹 vs 新核心) ===")
    Imd_old = img_test[rowmin:rowmax + 1, colmin:colmax + 1]
    from s_T_shift import s_T_shift as s_shift_old
    from zStd import zStd as zstd_old

    pd_old = s_shift_old(Imd_old, 15, 1, 21)
    pd_new = s_T_shift(Imd_old, 15, 1, 21)
    print("s_T_shift 最大差异:", np.max(np.abs(pd_old - pd_new)))

    mask_old = mask_orig.astype(bool)
    rows, cols = mask_old.shape
    x = np.linspace(-1, 1, cols); y = np.linspace(-1, 1, rows)
    X0, Y0 = np.meshgrid(x, y)
    R0 = np.sqrt(X0 ** 2 + Y0 ** 2)
    idx = np.nonzero(mask_old.ravel())[0]
    rho = R0.ravel()[idx]
    theta = np.arctan2(Y0.ravel()[idx], X0.ravel()[idx])
    Zp_old, n1, m1, _ = zstd_old(rho, theta, 36)
    Zp_new, n2, m2, _ = zStd(rho, theta, 36)
    print("zStd 最大差异:", np.max(np.abs(Zp_old - Zp_new)),
          "n一致:", np.array_equal(n1, n2), "m一致:", np.array_equal(m1, m2))
else:
    print("\n=== 2~3. 未找到原版工程，跳过 A/B；继续使用项目内仿真数据 ===")

print("\n=== 4. 完整流程 (自动检测掩膜) ===")
res = process_fizeau(REF, TEST, 15, cx_auto, cy_auto, int(r_auto),
                     progress=lambda p, m: print(f"  {p:3d}% {m}"))
print(f"\n全局 RMS = {res.global_rms:.3f} nm, PV = {res.global_pv:.3f} nm")
print("去项残差 (k, RMS, PV):")
for r_ in res.residuals:
    print(f"  k={r_['k']:2d}  RMS={r_['rms']:8.3f} nm  PV={r_['pv']:8.3f} nm")
print("\n前 15 项 Zernike 系数 (nm):")
for i in range(15):
    print(f"  Z{res.jVec[i]:3d} (n={res.nVec[i]},m={res.mVec[i]:2d}) "
          f"{zernike_name(int(res.jVec[i])):<28} {res.coefficients_nm[i]:9.3f}")

print("\n=== 5. 报告示例 ===")
print(build_report(res))
