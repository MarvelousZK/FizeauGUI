# -*- coding: utf-8 -*-
"""
仿真生成菲索干涉仪条纹图
==========================

生成一对与软件约定一致的干涉图（参考元件 + 待测元件），用于后续测试软件。

仿真模型 (双光束干涉):
    I(x,y) = I0(x,y) + I1(x,y) * cos[ 载波相位 + 4*pi*h(x,y)/lambda ]
    - 圆形孔径 + 高斯照明 I0，边缘调制度略降
    - 竖直载波条纹，周期 = PERIOD 像素，"右边相位小"(s=21 约定)
    - 表面面形 h 用标准 Zernike 多项式合成 (系数单位 nm，与软件报告一致)
    - 高斯噪声 + 8bit 量化

用法:
    python examples/generate_simulation.py
输出:
    examples/simulated_data/仿真_参考元件.bmp       <- 软件里选"参考元件"
    examples/simulated_data/仿真_待测元件.bmp       <- 软件里选"待测元件"
    examples/simulated_data/仿真数据说明.txt        <- 全部参数与预期结果
    examples/simulated_data/仿真_真实面形_*.png     <- 真实面形图(对答案用)

脚本会自动用软件核心流水线跑一遍生成的数据做自校验，
打印"真实面形 vs 软件恢复面形"的误差。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fizeau_gui.core import (zStd, s_T_shift, process_fizeau,
                             auto_detect_circle, build_mask)

# ================= 可调参数 =================
IMAGE = 720              # 图像边长 (像素)
CENTER = (360.0, 360.0)  # 孔径圆心 (列, 行)
RADIUS = 270.0           # 孔径半径 (像素)
PERIOD = 15              # 条纹周期 (像素)，与软件默认参数一致
LAMBDA_NM = 635.0        # 波长 (nm)，与软件默认一致
NOISE = 1.5              # 高斯噪声 (灰度级)
SEED = 20231031          # 随机种子，固定可复现

# 面形像差: {Zernike 项号: 系数(nm)}，与软件报告的系数单位一致
#   4=离焦  5=45°像散  6=0°像散  7/8=彗差  9/10=三叶  11=球差
ABER_REF = {4: 5.0, 6: 4.0, 8: 3.0}      # 参考元件：接近理想，微小残差
ABER_TEST = {4: 40.0, 5: 30.0, 6: 25.0,   # 待测元件：含明显像差
             7: 20.0, 9: 12.0, 11: 15.0}
# =============================================

OUTDIR = str(HERE / "simulated_data")


def make_interferogram(surface_nm: np.ndarray, mask: np.ndarray,
                       carrier: np.ndarray, rho: np.ndarray,
                       rng: np.random.Generator,
                       noise_sigma: float = NOISE) -> np.ndarray:
    """由面形(nm)、载波生成干涉图 (float, 0~255)。"""
    phase = carrier + 4.0 * np.pi * surface_nm / LAMBDA_NM     # 双程反射
    I0 = 50.0 + 100.0 * np.exp(-(rho / 0.75) ** 2)             # 高斯照明
    I1 = 0.65 * I0 * (1.0 - 0.10 * rho ** 2)                   # 调制度，边缘略降
    I = I0 + I1 * np.cos(phase)
    I = I * mask                                               # 孔径外背景 0
    if noise_sigma > 0:
        I = I + rng.standard_normal(I.shape) * noise_sigma
    return I


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- 坐标与 Zernike 基 ----
    N = IMAGE
    xs = np.arange(N, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)
    cx, cy = CENTER
    u = (X - cx) / RADIUS
    v = (Y - cy) / RADIUS
    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    mask = (rho <= 1.0)

    maxj = max(max(ABER_REF), max(ABER_TEST))
    Zfull, nV, mV, jV = zStd(np.clip(rho, 0, 1).ravel(), theta.ravel(), maxj)
    Zfull = Zfull.reshape(N, N, maxj)

    def surface(amps: dict) -> np.ndarray:
        h = np.zeros((N, N), dtype=np.float64)
        for j, a in amps.items():
            h += a * Zfull[:, :, j - 1]
        return h * mask

    h_ref = surface(ABER_REF)
    h_test = surface(ABER_TEST)
    h_diff_truth = (h_ref - h_test) * mask      # 软件相位差对应的真实面形 (nm)

    # ---- 载波: 竖直条纹, 右边相位小 (s=21 约定) ----
    carrier = -2.0 * np.pi * (X - cx) / PERIOD

    I_ref = make_interferogram(h_ref, mask, carrier, rho, rng)
    I_test = make_interferogram(h_test, mask, carrier, rho, rng)

    ref_bmp = os.path.join(OUTDIR, "仿真_参考元件.bmp")
    test_bmp = os.path.join(OUTDIR, "仿真_待测元件.bmp")

    from PIL import Image   # cv2.imwrite 不支持中文路径，用 PIL 保存
    Image.fromarray(np.uint8(np.clip(np.round(I_ref), 0, 255))).save(ref_bmp)
    Image.fromarray(np.uint8(np.clip(np.round(I_test), 0, 255))).save(test_bmp)
    print(f"[1] 已生成干涉图:\n    {ref_bmp}\n    {test_bmp}")
    print(f"    图像 {N}x{N}, 孔径圆心 {CENTER}, 半径 {RADIUS}px, "
          f"条纹周期 {PERIOD}px, 噪声 {NOISE} 灰度级")

    # ---- 保存真实面形图 (对答案用) ----
    def save_truth(h, name, title):
        fig_p = os.path.join(OUTDIR, name)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        hh = np.where(mask, h, np.nan)
        im = ax.imshow(hh, cmap="coolwarm")
        fig.colorbar(im, ax=ax, fraction=0.046, label="nm")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(fig_p, dpi=150, bbox_inches="tight")
        plt.close(fig)

    save_truth(h_ref, "仿真_真实面形_参考.png", "True Surface - Reference (nm)")
    save_truth(h_test, "仿真_真实面形_待测.png", "True Surface - Test (nm)")
    save_truth(h_diff_truth, "仿真_真实面形_差值.png",
               "True Surface = Ref - Test (nm)")

    # ---- 自校验 1: 相位提取的噪声水平 (含噪图 vs 无噪图的提取结果) ----
    print("\n[2] 自校验...")
    det = auto_detect_circle(I_ref)
    print(f"    自动检测孔径: 中心 {tuple(round(c, 1) for c in det[:2])}, "
          f"半径 {det[2]:.1f} px")
    info = build_mask((N, N), *det)
    rmin, rmax, cmin, cmax = info.rowmin, info.rowmax, info.colmin, info.colmax

    Imd = I_test[rmin:rmax + 1, cmin:cmax + 1]
    pd_noisy = s_T_shift(Imd, PERIOD, 1, 21)
    I_clean = make_interferogram(h_test, mask, carrier, rho, rng, noise_sigma=0.0)
    pd_clean = s_T_shift(I_clean[rmin:rmax + 1, cmin:cmax + 1], PERIOD, 1, 21)
    interior = (np.arange(pd_noisy.shape[1]) >= 8) & \
               (np.arange(pd_noisy.shape[1]) < pd_noisy.shape[1] - 7)
    d = (pd_noisy[:, interior] - pd_clean[:, interior]) * info.mask[:, interior]
    d = np.angle(np.exp(1j * d))       # 按 2π 包裹，消除支路翻转假象
    err_extract = float(np.sqrt(np.mean(d ** 2)))
    print(f"    相位提取噪声水平: RMS = {err_extract:.4f} rad (应 < 0.05)")

    # ---- 自校验 2: 完整流水线 vs 真实面形 ----
    res = process_fizeau(ref_bmp, test_bmp, PERIOD, *det,
                         progress=lambda p, m: None)
    valid_mask = res.mask_info.mask
    W_rec = res.residuals[0]["W"]                     # 软件恢复的面形 (nm)
    W_truth = h_diff_truth[rmin:rmax + 1, cmin:cmax + 1]
    # 去掉无物理意义的全局平移(piston)后对比
    d = W_rec[valid_mask] - W_truth[valid_mask]
    piston = float(np.mean(d))
    err_rms = float(np.std(d))
    rms_true = float(np.std(W_truth[valid_mask]))
    pv_true = float(W_truth[valid_mask].max() - W_truth[valid_mask].min())
    print(f"    软件恢复面形 vs 真实面形: RMS 误差 {err_rms:.2f} nm "
          f"(全局平移 {piston:.1f} nm 已剔除, 无物理意义)")
    print(f"    真实面形(参考-待测, 全口径): RMS = {rms_true:.2f} nm, PV = {pv_true:.2f} nm")
    print(f"    软件输出: 全局 RMS = {res.global_rms:.2f} nm, "
          f"PV = {res.global_pv:.2f} nm")
    for r_ in res.residuals[1:4]:
        print(f"      去前{r_['k']}项: RMS = {r_['rms']:.2f} nm, PV = {r_['pv']:.2f} nm")

    # ---- 写说明文件 ----
    info_txt = os.path.join(OUTDIR, "仿真数据说明.txt")
    lines = [
        "菲索干涉仪仿真数据说明",
        "=" * 50,
        f"图像尺寸        : {N} x {N}",
        f"孔径            : 圆心 {CENTER}, 半径 {RADIUS} px",
        f"条纹周期        : {PERIOD} px (竖直条纹, 右边相位小)",
        f"波长            : {LAMBDA_NM} nm (双程反射)",
        f"噪声            : 高斯 sigma={NOISE} 灰度级, 8bit 量化",
        "",
        "参考元件 Zernike 像差 (nm):",
        "  项号  名称         系数",
    ]
    names = {4: "离焦", 5: "45°像散", 6: "0°像散", 7: "X彗差",
             8: "Y彗差", 9: "X三叶", 10: "Y三叶", 11: "球差"}
    for j, a in ABER_REF.items():
        lines.append(f"  {j:>3}   {names.get(j, ''):<10} {a:>8.2f}")
    lines.append("")
    lines.append("待测元件 Zernike 像差 (nm):")
    lines.append("  项号  名称         系数")
    for j, a in ABER_TEST.items():
        lines.append(f"  {j:>3}   {names.get(j, ''):<10} {a:>8.2f}")
    lines.append("")
    lines.append(f"真实面形 (参考-待测): RMS = {rms_true:.2f} nm, PV = {pv_true:.2f} nm")
    lines.append(f"软件实测 (本次自校验): 全局 RMS = {res.global_rms:.2f} nm, "
                 f"PV = {res.global_pv:.2f} nm")
    lines.append(f"软件恢复误差: {err_rms:.2f} nm RMS")
    lines.append("")
    lines.append("软件中用法:")
    lines.append("  参考元件 -> 仿真_参考元件.bmp")
    lines.append("  待测元件 -> 仿真_待测元件.bmp")
    lines.append("  条纹周期填 15，其余默认，点【开始处理】")
    with open(info_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[3] 说明文件: {info_txt}")
    print(f"    面形真值图与预期数值见 {OUTDIR}")
    print("\n完成! 打开软件载入 仿真_参考元件.bmp 与 仿真_待测元件.bmp 即可测试。")


if __name__ == "__main__":
    main()
