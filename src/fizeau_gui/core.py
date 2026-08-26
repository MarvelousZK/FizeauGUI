# -*- coding: utf-8 -*-
"""
菲索干涉仪数据处理 — 核心算法模块
====================================

菲索干涉图的完整处理流程：
    读图 → 掩膜(圆形孔径) → 单帧载波法提取相位（多种算法可选）
    → 相位展开 → 相位差 → Zernike 拟合(默认 80 项)
    → 依次去前 1..N 项残差 (RMS/PV)

本模块不依赖任何 GUI 库，可独立测试。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # 自动检测圆需要 opencv；没有时仍可手动选掩膜
    cv2 = None
    _HAS_CV2 = False

from scipy.ndimage import (binary_erosion, distance_transform_edt,
                           gaussian_filter, uniform_filter)


# ---------------------------------------------------------------- 基本工具

def read_image(path: str) -> np.ndarray:
    """读取图像为灰度 float64 数组。中文路径优先用 PIL（cv2 不支持）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    def _pil_read():
        try:
            from PIL import Image
            with Image.open(path) as im:
                return np.asarray(im.convert("L"), dtype=np.float64)
        except Exception:
            return None

    non_ascii = any(ord(ch) > 127 for ch in path)
    if not non_ascii and cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float64)
    img = _pil_read()
    if img is not None:
        return img
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img.astype(np.float64)
    raise IOError(f"无法读取图像文件: {path}")


def s_T_shift(I: np.ndarray, step: int, P: int = 1, s: int = 21) -> np.ndarray:
    """
    空间载波相移算法提取截断相位（对应 MATLAB s_T_shift.m）。

    参数:
        I    : 单张干涉图 (m x n)
        step : 条纹周期的像素数 (>=3)
        P    : 处理方向, 1=垂直, 2=水平
        s    : 12=左边相位小, 21=右边相位小

    返回:
        截断相位 (-pi, pi]，行列与 I 相同（边界处为 0）。
    """
    if not isinstance(I, np.ndarray):
        raise TypeError("I 必须是 numpy 数组")
    if int(step) < 3:
        raise ValueError("相移步数(条纹周期)必须 >= 3")
    step = int(step)
    if P not in (1, 2):
        raise ValueError("P 必须是 1(垂直) 或 2(水平)")
    if s not in (12, 21):
        raise ValueError("s 必须是 12 或 21")

    I = np.asarray(I, dtype=np.float64).copy()
    if P == 2:
        I = I.T
    if s == 21:
        I = np.fliplr(I)

    m, n = I.shape
    if step > n:
        direction = "宽度" if P == 1 else "高度"
        raise ValueError(
            f"条纹周期 {step} 像素大于处理区域{direction} {n} 像素，无法提取相位")
    out = np.zeros((m, n), dtype=np.float64)

    angles = 2.0 * np.pi * np.arange(step) / step
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)

    a = int(math.ceil(step / 2))
    b = int(math.floor(step / 2))

    for j in range(a - 1, n - b):  # 与 MATLAB j = a:n-b (1-based) 对应
        if step % 2 == 1:
            window = I[:, j - b: j - b + step]
        else:
            window = I[:, j - b + 1: j - b + 1 + step]
        fenzi = window @ sin_a
        fenmu = window @ cos_a
        out[:, j] = np.arctan2(-fenzi, fenmu) - 2.0 * np.pi * j / step

    if P == 2:
        out = out.T
    if s == 21:
        out = np.fliplr(out)
    return out


def masked_s_T_shift(I: np.ndarray, mask: np.ndarray, step: int,
                     P: int = 1, s: int = 21) -> np.ndarray:
    """带有效孔径边界约束的空间载波相移。

    内部像素与 :func:`s_T_shift` 使用完全相同的单周期正弦投影。靠近圆周时，
    将长度为 ``step`` 的窗口向孔径内部平移，保证只使用 mask 内有效像素；
    极端顶部/底部不足一个周期的短弦则用其全部有效像素做三参数正弦拟合。
    因而既保留原算法对斜载波逐行解调的特性，又不会把孔径外黑背景或方形
    ROI 的缺失窗口作为有效相位参与全口径 RMS/PV。
    """
    if not isinstance(I, np.ndarray):
        raise TypeError("I 必须是 numpy 数组")
    I = np.asarray(I, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if I.ndim != 2 or I.shape != mask.shape:
        raise ValueError("I 与 mask 必须是尺寸一致的二维数组")
    step = int(step)
    if step < 3:
        raise ValueError("相移步数(条纹周期)必须 >= 3")
    if P not in (1, 2):
        raise ValueError("P 必须是 1(垂直) 或 2(水平)")
    if s not in (12, 21):
        raise ValueError("s 必须是 12 或 21")
    if not mask.any():
        return np.zeros_like(I)

    if P == 2:
        I = I.T
        mask = mask.T
    if s == 21:
        I = np.fliplr(I)
        mask = np.fliplr(mask)

    rows, cols = I.shape
    out = np.zeros_like(I)
    omega = 2.0 * np.pi / step
    k = np.arange(step, dtype=np.float64)
    sin_k = np.sin(omega * k)
    cos_k = np.cos(omega * k)
    center_offset = (step - 1) // 2

    for row_idx in range(rows):
        valid_cols = np.flatnonzero(mask[row_idx])
        if valid_cols.size == 0:
            continue
        left, right = int(valid_cols[0]), int(valid_cols[-1])
        length = right - left + 1
        row = I[row_idx]

        if length >= step:
            segment = row[left:right + 1]
            windows = np.lib.stride_tricks.sliding_window_view(segment, step)
            numerator = windows @ sin_k
            denominator = windows @ cos_k
            starts = np.arange(left, right - step + 2)
            window_phase = (np.arctan2(-numerator, denominator)
                            - omega * (starts + center_offset))

            targets = np.arange(left, right + 1)
            selected_starts = np.clip(
                targets - center_offset, left, right - step + 1)
            out[row_idx, targets] = window_phase[selected_starts - left]
        elif length >= 3:
            # 单帧图在一个短弦上信息有限；三参数拟合是使用全部实测像素的
            # 最小可辨识模型，不以邻域外推或裁口径替代测量。
            x = np.arange(left, right + 1, dtype=np.float64)
            design = np.column_stack((
                np.ones(length), np.cos(omega * x), np.sin(omega * x)))
            coef = np.linalg.lstsq(design, row[left:right + 1], rcond=None)[0]
            phase = np.arctan2(-coef[2], coef[1]) - omega * center_offset
            out[row_idx, left:right + 1] = phase

    if P == 2:
        out = out.T
    if s == 21:
        out = np.fliplr(out)
    return out


# ---------------------------------------------------------------- 经典算法移植
# LS-SCPS（内部兼容键 adapt2）：最小二乘空间载波相移
# WFT（内部兼容键 wft2）：加窗傅里叶变换（原 MATLAB wft2.m / onestep_wff.m）


def _luo_dominant_axis(I: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> int:
    """返回主载频变化方向：1=x/列，0=y/行。"""
    if mask is None:
        rows, cols = I.shape
        pad = max(1, min(rows, cols) // 10)
        core = I[pad:rows - pad, pad:cols - pad]
        if min(core.shape) < 3:
            core = I
        gx = float(np.mean(np.diff(core, axis=1) ** 2))
        gy = float(np.mean(np.diff(core, axis=0) ** 2))
        return 1 if gx >= gy else 0

    valid = np.asarray(mask, dtype=bool)
    dx = np.diff(I, axis=1)
    dy = np.diff(I, axis=0)
    valid_x = valid[:, :-1] & valid[:, 1:]
    valid_y = valid[:-1, :] & valid[1:, :]
    gx = float(np.mean(dx[valid_x] ** 2)) if np.any(valid_x) else -np.inf
    gy = float(np.mean(dy[valid_y] ** 2)) if np.any(valid_y) else -np.inf
    return 1 if gx >= gy else 0


def _luo_phase_increment(I: np.ndarray, period: int, axis: int,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
    """按 LS-SCPS 参考模型估计指定方向的逐像素相位增量。"""
    work = I if axis == 1 else I.T
    work_mask = (np.ones_like(work, dtype=bool) if mask is None else
                 (np.asarray(mask, dtype=bool)
                  if axis == 1 else np.asarray(mask, dtype=bool).T))
    n_lines, n_samples = work.shape

    # 论文要求增量估计窗口覆盖至少一个条纹周期：2N+1 > T。
    half_window = max(2, int(np.ceil(period / 2.0)))
    if n_samples <= 2 * half_window:
        direction = "列(x)" if axis == 1 else "行(y)"
        raise ValueError(
            f"图像在{direction}方向太小，最小二乘空间载波相移LS-SCPS至少需要 "
            f"{2 * half_window + 1} 个像素")

    delta = np.full((n_lines, n_samples), np.nan, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    for center in range(half_window, n_samples - half_window):
        d1 = (work[:, center - half_window + 1:center + half_window - 1]
              - work[:, center - half_window + 2:center + half_window])
        d3 = (work[:, center - half_window + 3:center + half_window + 1]
              - work[:, center - half_window:center + half_window - 2])
        valid_terms = (
            work_mask[:, center - half_window + 1:center + half_window - 1]
            & work_mask[:, center - half_window + 2:center + half_window]
            & work_mask[:, center - half_window + 3:center + half_window + 1]
            & work_mask[:, center - half_window:center + half_window - 2]
        )
        A_terms = np.where(valid_terms, d1 * d3, 0.0)
        B_terms = np.where(valid_terms, d1 * d1 - d3 * d3, 0.0)
        A = np.sum(A_terms, axis=1)
        B = np.sum(B_terms, axis=1)
        support = np.count_nonzero(valid_terms, axis=1)

        # 原 MATLAB 在 A=0 时会产生 Inf/NaN；这里仅把数值不可辨识点
        # 留给同一扫描线内的有效估计插值，不把异常强制伪装成零相移。
        scale = np.maximum(np.sum(np.abs(A_terms), axis=1), 1.0)
        valid = ((support >= 3) & work_mask[:, center]
                 & (np.abs(A) > 64.0 * eps * scale))
        argument = np.full(n_lines, np.nan, dtype=np.float64)
        argument[valid] = (
            (B[valid] - np.hypot(B[valid], 2.0 * A[valid]))
            / (4.0 * A[valid]) - 0.5
        )
        valid &= np.isfinite(argument)
        delta[valid, center] = np.arccos(
            np.clip(argument[valid], -1.0, 1.0))

    # 逐扫描线延拓边界；这修正了原 adapt2_iterators2.m 把第一行的
    # 单个标量广播到整幅边界、从而破坏空间变化的问题。
    nominal = 2.0 * np.pi / float(period)
    samples = np.arange(n_samples)
    for line in range(n_lines):
        targets = np.flatnonzero(work_mask[line])
        good = np.isfinite(delta[line]) & work_mask[line]
        if np.count_nonzero(good) >= 2:
            delta[line, targets] = np.interp(
                targets, samples[good], delta[line, good])
        else:
            delta[line, targets] = nominal

    # 口径外值不参与拟合，但给累积和一个有限占位，避免 NaN 穿过扫描线。
    delta[~work_mask] = nominal

    return delta if axis == 1 else delta.T


def _luo_single_carrier_fit(I: np.ndarray, delta: np.ndarray,
                            period: int, axis: int,
                            mask: Optional[np.ndarray] = None) -> np.ndarray:
    """用 LS-SCPS 五参数模型的单载频三参数退化式恢复中心相位。"""
    work = I if axis == 1 else I.T
    shifts = delta if axis == 1 else delta.T
    work_mask = (np.ones_like(work, dtype=bool) if mask is None else
                 (np.asarray(mask, dtype=bool)
                  if axis == 1 else np.asarray(mask, dtype=bool).T))
    n_lines, n_samples = work.shape

    # 局部重建窗口不超过一个周期；至少 3 点才能辨识 [a, B, C]。
    half_window = max(1, int(np.floor((period - 2) / 2.0)))
    if n_samples <= 2 * half_window:
        raise ValueError("图像太小，无法进行最小二乘空间载波相移LS-SCPS拟合")

    inner = n_samples - 2 * half_window
    centers = slice(half_window, n_samples - half_window)
    cumulative = np.cumsum(shifts, axis=1)
    H = np.zeros((n_lines, inner, 3, 3), dtype=np.float64)
    M = np.zeros((n_lines, inner, 3, 1), dtype=np.float64)
    support = np.zeros((n_lines, inner), dtype=np.int16)

    for offset in range(-half_window, half_window + 1):
        neighbors = slice(half_window + offset,
                          n_samples - half_window + offset)
        alpha = cumulative[:, neighbors] - cumulative[:, centers]
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        intensity = work[:, neighbors]
        valid = work_mask[:, neighbors]
        weight = valid.astype(np.float64)
        support += valid

        H[:, :, 0, 0] += weight
        H[:, :, 0, 1] += weight * cos_alpha
        H[:, :, 0, 2] += weight * sin_alpha
        H[:, :, 1, 1] += weight * cos_alpha * cos_alpha
        H[:, :, 1, 2] += weight * cos_alpha * sin_alpha
        H[:, :, 2, 2] += weight * sin_alpha * sin_alpha
        M[:, :, 0, 0] += weight * intensity
        M[:, :, 1, 0] += weight * intensity * cos_alpha
        M[:, :, 2, 0] += weight * intensity * sin_alpha

    H[:, :, 1, 0] = H[:, :, 0, 1]
    H[:, :, 2, 0] = H[:, :, 0, 2]
    H[:, :, 2, 1] = H[:, :, 1, 2]

    # 三参数模型在一个载频周期内应满秩；只加入与矩阵尺度相关的机器精度
    # 扰动以消除浮点奇异，不再用固定正则项掩盖五参数模型的秩亏。
    trace_scale = np.maximum(np.trace(H, axis1=2, axis2=3) / 3.0, 1.0)
    ridge = 64.0 * np.finfo(np.float64).eps * trace_scale
    H += ridge[:, :, None, None] * np.eye(3)[None, None, :, :]
    coefficients = np.linalg.solve(H, M)
    phase = np.arctan2(-coefficients[:, :, 2, 0],
                       coefficients[:, :, 1, 0])
    window_length = 2 * half_window + 1
    center_mask = work_mask[:, half_window:n_samples - half_window]
    fully_supported = center_mask & (support == window_length)

    # 只让完整落在有效口径内的局部窗充当相位锚点。靠圆孔径边缘的
    # 截断窗常常病态，是旧实现边缘 PV 偏大的主要来源。
    full = np.full_like(work, np.nan)
    inner_phase = full[:, half_window:n_samples - half_window]
    inner_phase[fully_supported] = phase[fully_supported]

    # 对每条有效弦，依据 LS-SCPS 已估计的局部相位增量从最近锚点延拓到
    # 边缘；这保留相位斜率，不再把矩形首尾相位平铺到圆口径边缘。
    for line in range(n_lines):
        targets = np.flatnonzero(work_mask[line])
        if targets.size == 0:
            continue
        anchors = np.flatnonzero(np.isfinite(full[line]))
        if anchors.size == 0 and targets.size >= 3:
            # 极短圆弦容不下完整局部窗时，用该弦全部实测像素进行同一个
            # 三参数退化模型拟合；仍不引入口径外灰度。
            center = int(targets[targets.size // 2])
            alpha = cumulative[line, targets] - cumulative[line, center]
            design = np.column_stack((
                np.ones(targets.size), np.cos(alpha), np.sin(alpha)))
            if np.linalg.matrix_rank(design) == 3:
                coef = np.linalg.lstsq(
                    design, work[line, targets], rcond=None)[0]
                center_phase = np.arctan2(-coef[2], coef[1])
                full[line, targets] = np.angle(np.exp(
                    1j * (center_phase + alpha)))
                continue
        if anchors.size == 0:
            continue

        missing = targets[~np.isfinite(full[line, targets])]
        if missing.size == 0:
            continue
        insertion = np.searchsorted(anchors, missing)
        right_pos = np.clip(insertion, 0, anchors.size - 1)
        left_pos = np.clip(insertion - 1, 0, anchors.size - 1)
        left_anchor = anchors[left_pos]
        right_anchor = anchors[right_pos]
        choose_right = (np.abs(right_anchor - missing)
                        < np.abs(missing - left_anchor))
        nearest = np.where(choose_right, right_anchor, left_anchor)
        propagated = (full[line, nearest] + cumulative[line, missing]
                      - cumulative[line, nearest])
        full[line, missing] = np.angle(np.exp(1j * propagated))

    # 只有小于 3 像素的极端圆弦无法从单帧强度辨识三参数；用最近的
    # 已测口径相位补齐，保证后续展开和完整口径统计没有空洞。
    solved = work_mask & np.isfinite(full)
    missing = work_mask & ~solved
    if np.any(missing) and np.any(solved):
        nearest_indices = distance_transform_edt(
            ~solved, return_distances=False, return_indices=True)
        nearest_values = full[tuple(nearest_indices)]
        full[missing] = nearest_values[missing]

    # 圆周最后几个像素的局部窗不可避免地被截断，且真实相机在孔径边缘
    # 往往调制度最低。将相位拆为“名义载频 + 慢变余相位”，再从完整窗
    # 区域把慢变项延拓到边缘，可抑制少量近 ±2π 的展开枝错，同时保留
    # 主载频，不以裁小统计口径来掩盖边缘误差。
    edge_width = max(2, half_window)
    trusted = binary_erosion(work_mask, iterations=edge_width,
                             border_value=0)
    edge = work_mask & ~trusted
    if np.any(edge) and np.any(trusted):
        trusted_indices = distance_transform_edt(
            ~trusted, return_distances=False, return_indices=True)
        carrier = ((2.0 * np.pi / float(period))
                   * np.arange(n_samples, dtype=np.float64)[None, :])
        residual = np.angle(np.exp(1j * (full - carrier)))
        nearest_residual = residual[tuple(trusted_indices)]
        full[edge] = np.angle(np.exp(
            1j * (nearest_residual[edge]
                  + np.broadcast_to(carrier, full.shape)[edge])))
    full[~work_mask] = 0.0
    return full if axis == 1 else full.T


def adapt2_phase(I: np.ndarray, period: int,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    最小二乘空间载波相移 LS-SCPS 相位提取。

    论文正式模型针对正交双载频，局部强度写成五参数线性模型。本程序的
    菲索图只有一组主载频，因此先自动判断 x/y 主方向，再保留论文模型中
    对应的 ``[1, cos(alpha), sin(alpha)]`` 三列。这是五参数法在另一组条纹
    振幅为零时的代数退化形式，可避免把单载频数据硬塞入秩亏的五参数方程。

    步骤：
      1. 按参考文献等式 (6)--(9) 估计主方向逐像素相位增量；
      2. 累加邻点相对于中心像素的空间相移量 alpha；
      3. 按等式 (12)--(17) 的单载频三参数形式逐像素最小二乘求相位。

    参考：Optical Engineering 59(2), 024103 (2020),
    doi:10.1117/1.OE.59.2.024103。

    参数:
        I      : 单张干涉图 (m x n)
        period : 条纹周期的像素数
        mask   : 可选有效口径。提供时，方向判断、相位增量和局部拟合均
                 排除口径外像素，并沿局部相位增量恢复圆孔径边缘。

    返回:
        截断相位 (-pi, pi]，行列与 I 相同。
    """
    I = np.asarray(I, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("I 必须是二维数组")
    period = int(period)
    if period < 3:
        raise ValueError("条纹周期必须 >= 3")
    if min(I.shape) < 3:
        raise ValueError("图像太小，无法进行最小二乘空间载波相移LS-SCPS")
    if mask is None:
        valid_mask = np.ones(I.shape, dtype=bool)
    else:
        valid_mask = np.asarray(mask, dtype=bool)
        if valid_mask.shape != I.shape:
            raise ValueError("mask 必须与 I 形状相同")
        if np.count_nonzero(valid_mask) < 3:
            raise ValueError("有效口径像素太少，无法进行最小二乘空间载波相移LS-SCPS")

    axis = _luo_dominant_axis(I, valid_mask)
    delta = _luo_phase_increment(I, period, axis, valid_mask)
    fitted = _luo_single_carrier_fit(
        I, delta, period, axis, valid_mask)

    # 保持原工程 phasex=-phasex 的符号约定，确保参考/待测差分方向不变。
    return -fitted


@dataclass
class WFT2Result:
    """加窗傅里叶变换 WFT 的复场、质量量和实际搜索参数。"""
    phase_wrapped: np.ndarray
    amplitude: np.ndarray
    confidence: np.ndarray
    complex_field: np.ndarray
    background: np.ndarray
    carrier_rad: tuple[float, float]       # (wx, wy), rad / pixel
    frequency_band: float
    frequency_step: float
    threshold: float


def _wft2_carrier(I: np.ndarray, mask: np.ndarray, period: int,
                  centered: Optional[np.ndarray] = None) -> tuple[float, float]:
    """在期望周期附近从掩膜频谱估计单个二维载频分支。"""
    image = np.asarray(I, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(image)
    w0 = 2.0 * np.pi / float(period)
    if centered is None:
        centered = np.zeros_like(image)
        centered[valid] = image[valid] - float(np.mean(image[valid]))

    rows, cols = image.shape
    row_window = np.hanning(rows) if rows > 2 else np.ones(rows)
    col_window = np.hanning(cols) if cols > 2 else np.ones(cols)
    spectrum = np.fft.fftshift(np.fft.fft2(
        centered * row_window[:, None] * col_window[None, :]))
    magnitude = np.abs(spectrum)
    wx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(cols))
    wy = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(rows))
    WX, WY = np.meshgrid(wx, wy)
    radius = np.hypot(WX, WY)

    # 周期输入只作为径向先验。保留一个确定的半平面，避免两幅图分别
    # 选到共轭谱峰而产生相位符号翻转。
    bin_width = 2.0 * np.pi / float(max(rows, cols))
    positive_half = ((WX > 0.5 * bin_width)
                     | ((np.abs(WX) <= 0.5 * bin_width) & (WY > 0)))
    search = (radius >= max(0.55 * w0, 2.5 * bin_width))
    search &= radius <= min(1.60 * w0, 0.95 * np.pi)
    search &= positive_half
    if not np.any(search):
        axis = _luo_dominant_axis(np.where(valid, image, 0.0))
        return (w0, 0.0) if axis == 1 else (0.0, w0)

    score = np.where(search, magnitude, -np.inf)
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    if not np.isfinite(score[row, col]) or score[row, col] <= 0:
        axis = _luo_dominant_axis(np.where(valid, image, 0.0))
        return (w0, 0.0) if axis == 1 else (0.0, w0)
    return float(WX[row, col]), float(WY[row, col])


def wft2_phase(I: np.ndarray, period: int,
               sigma: float = 10.0, thr: Optional[float] = None,
               freq_band: Optional[float] = 0.3,
               freq_step: float = 0.1,
               mask: Optional[np.ndarray] = None,
               carrier_rad: Optional[tuple[float, float]] = None,
               return_result: bool = False) -> np.ndarray | WFT2Result:
    """
    加窗傅里叶变换 WFT 相位提取（对应原 MATLAB 局部时频滤波实现）。

    先在有效孔径内去除缓变背景并自动估计二维载频，再在该载频附近做
    Gaussian WFT 重构。搜索带会随载频收缩，保证不跨过 DC 或混入共轭边带。

    参数:
        I          : 单张干涉图 (m x n)
        period     : 条纹周期的像素数
        sigma      : 高斯窗标准差 (原程序固定 10)
        thr        : 频率分量幅度阈值；None 时按局部调制度自适应
        freq_band  : 载频搜索最大半宽；实际值不超过载频模长的 45%
        freq_step  : 频率搜索步长 (原程序固定 0.1)
        mask       : 有效孔径；孔径外像素不参与背景和载频估计
        carrier_rad: 可选统一载频 (wx, wy)，单位 rad/pixel
        return_result: True 时返回相位、幅值、置信度和实际参数

    返回:
        默认返回截断相位；return_result=True 时返回 WFT2Result。
    """
    I = np.asarray(I, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("I 必须是二维数组")
    period = int(period)
    if period < 3:
        raise ValueError("条纹周期必须 >= 3")
    finite = np.isfinite(I)
    if mask is None:
        valid = finite
    else:
        aperture = np.asarray(mask, dtype=bool)
        if aperture.shape != I.shape:
            raise ValueError("mask 必须与 I 尺寸一致")
        valid = aperture & finite
    if not valid.any():
        raise ValueError("有效孔径内没有有限灰度像素")

    s = int(round(2 * sigma))
    if s < 1:
        raise ValueError("sigma 过小")
    m, n = I.shape
    # WFT 对零频泄漏非常敏感。用归一化 Gaussian 卷积估计缓变背景，
    # 再把孔径外置零；这样黑色圆外区域不会作为强 DC 边缘参与卷积。
    background_sigma = max(float(period), float(sigma))
    support = gaussian_filter(valid.astype(np.float64), background_sigma,
                              mode="nearest")
    weighted = gaussian_filter(np.where(valid, I, 0.0), background_sigma,
                               mode="nearest")
    background = weighted / np.maximum(support, 1e-12)
    centered = np.zeros_like(I)
    centered[valid] = I[valid] - background[valid]

    if carrier_rad is None:
        carrier = _wft2_carrier(I, valid, period, centered=centered)
    else:
        if len(carrier_rad) != 2:
            raise ValueError("carrier_rad 必须是 (wx, wy)")
        carrier = (float(carrier_rad[0]), float(carrier_rad[1]))
    carrier_norm = float(np.hypot(*carrier))
    if not np.isfinite(carrier_norm) or carrier_norm <= 0:
        raise ValueError("无法确定有效的 WFT 载频")

    requested_band = 0.3 if freq_band is None else float(freq_band)
    if requested_band <= 0 or freq_step <= 0:
        raise ValueError("freq_band 和 freq_step 必须为正数")
    effective_band = min(requested_band, 0.45 * carrier_norm)
    effective_step = min(float(freq_step), effective_band / 3.0)

    values = centered[valid]
    contrast = 0.5 * (float(np.percentile(values, 95.0))
                      - float(np.percentile(values, 5.0)))
    intensity_scale = max(float(np.max(np.abs(I[valid]))), 1.0)
    if contrast <= 128.0 * np.finfo(float).eps * intensity_scale:
        raise ValueError("有效孔径内没有可辨识的条纹调制度")
    effective_thr = (0.075 * max(contrast, np.finfo(float).eps)
                     if thr is None else float(thr))
    if effective_thr < 0:
        raise ValueError("thr 不能为负数")

    # 与 MATLAB [y,x]=meshgrid 一致: xx 沿列(水平)方向变化, yy 沿行(竖直)方向变化
    xx, yy = np.meshgrid(np.arange(-s, s + 1), np.arange(-s, s + 1))
    w = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    w = w / np.sqrt(np.sum(w * w))

    fshape = (m + 2 * s, n + 2 * s)
    F = np.fft.fft2(centered, fshape)
    sl = (slice(s, s + m), slice(s, s + n))
    g = np.zeros((m, n), dtype=np.complex128)

    half_steps = max(1, int(np.ceil(effective_band / effective_step)))
    offsets = np.linspace(-effective_band, effective_band, 2 * half_steps + 1)
    actual_step = float(offsets[1] - offsets[0])
    wx_grid = carrier[0] + offsets
    wy_grid = carrier[1] + offsets
    for wyt in wy_grid:
        for wxt in wx_grid:
            wave = w * np.exp(1j * (wxt * xx + wyt * yy))
            K = np.fft.fft2(wave, fshape)
            sf = np.fft.ifft2(F * K)[sl]
            sf = sf * (np.abs(sf) >= effective_thr)
            sfF = np.fft.fft2(sf, fshape)
            g += np.fft.ifft2(sfF * K)[sl]

    g *= actual_step * actual_step / (4.0 * np.pi * np.pi)
    phase = np.angle(g)
    amplitude = np.abs(g)
    amp_scale = float(np.percentile(amplitude[valid], 90.0))
    amp_conf = np.clip(amplitude / max(amp_scale, np.finfo(float).eps), 0.0, 1.0)
    local_support = gaussian_filter(valid.astype(np.float64), float(sigma),
                                    mode="constant", cval=0.0)
    support_scale = float(np.max(local_support[valid]))
    local_support /= max(support_scale, np.finfo(float).eps)
    confidence = amp_conf * np.sqrt(np.clip(local_support, 0.0, 1.0))
    confidence[~valid] = 0.0
    phase[~valid] = 0.0

    result = WFT2Result(
        phase_wrapped=phase,
        amplitude=amplitude,
        confidence=confidence,
        complex_field=g,
        background=background,
        carrier_rad=carrier,
        frequency_band=effective_band,
        frequency_step=actual_step,
        threshold=effective_thr,
    )
    return result if return_result else result.phase_wrapped


@dataclass
class TakedaFTResult:
    """傅里叶变换 FT 单帧解调结果与完整诊断量。"""
    phase_wrapped: np.ndarray
    amplitude: np.ndarray
    confidence: np.ndarray
    spectrum_log: np.ndarray
    sideband_filter: np.ndarray
    filtered_spectrum_log: np.ndarray
    sideband_center: tuple[int, int]       # (row, col), Python 0-based
    carrier_cycles: tuple[int, int]        # (fx, fy), cycles / cropped image
    filter_sigma: float
    center_exclusion_radius: float
    complex_field: np.ndarray
    window: np.ndarray
    filter_too_wide: bool = False


@dataclass
class TakedaSpectrum:
    """傅里叶变换 FT 的预处理频谱，供手动选边带和算法本体共用。"""
    spectrum: np.ndarray
    magnitude: np.ndarray
    spectrum_log: np.ndarray
    window: np.ndarray
    aperture: np.ndarray
    valid: np.ndarray


def takeda_ft_spectrum(
        I: np.ndarray,
        mask: Optional[np.ndarray] = None,
        apply_hann: bool = True) -> TakedaSpectrum:
    """计算傅里叶变换 FT 使用的二维频谱，但不自动搜索或滤波一级边带。

    这个入口用于手动选峰流程：用户可以先观察中心零频与成对的
    一级谱峰，再自行选择其中一个边带。其归一化、孔径外填充和 Hann
    加窗步骤与 :func:`takeda_ft_phase` 完全一致。
    """
    image = np.asarray(I, dtype=np.float64)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("I 必须是非空二维数组")

    finite = np.isfinite(image)
    if mask is None:
        valid = finite
        aperture = finite.copy()
    else:
        aperture = np.asarray(mask, dtype=bool)
        if aperture.shape != image.shape:
            raise ValueError("mask 必须与 I 尺寸一致")
        valid = finite & aperture
    if not valid.any():
        raise ValueError("有效孔径内没有有限灰度像素")

    values = image[valid]
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("有效孔径内图像为常数，无法进行 FT 解调")
    normalized = (image - low) / (high - low)
    valid_mean = float(np.mean(normalized[valid]))
    normalized[~finite] = valid_mean
    normalized[~aperture] = valid_mean
    centered = normalized - valid_mean

    n_rows, n_cols = image.shape
    if apply_hann:
        row_index = np.arange(n_rows, dtype=np.float64)
        col_index = np.arange(n_cols, dtype=np.float64)
        row_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * row_index / n_rows)
        col_window = 0.5 - 0.5 * np.cos(2.0 * np.pi * col_index / n_cols)
        window = row_window[:, None] * col_window[None, :]
    else:
        window = np.ones_like(centered)

    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    magnitude = np.abs(spectrum)
    return TakedaSpectrum(
        spectrum=spectrum,
        magnitude=magnitude,
        spectrum_log=np.log1p(magnitude),
        window=window,
        aperture=aperture,
        valid=valid,
    )


def takeda_ft_phase(
        I: np.ndarray,
        mask: Optional[np.ndarray] = None,
        sideband_center: Optional[tuple[float, float]] = None,
        carrier_cycles: Optional[tuple[float, float]] = None,
        center_exclusion_radius: Optional[float] = None,
        filter_sigma: Optional[float] = None,
        apply_hann: bool = True,
        phase_sign: int = 1) -> TakedaFTResult:
    """用经典二维 Fourier-transform 法恢复单载频干涉相位。

    参数
    ----
    I
        二维灰度干涉图。
    mask
        有效孔径。仅用于归一化、背景均值、置信度和显示掩膜；FFT 前
        将孔径外填成孔径内均值，避免黑色画幅制造额外强边缘。
    sideband_center
        手动一级边带中心 ``(row, col)``，Python 0-based。为 ``None``
        时自动搜索。不能与 ``carrier_cycles`` 同时给出。
    carrier_cycles
        手动载频 ``(fx, fy)``，单位 cycles / cropped image。GUI 使用此
        形式，因为它比绝对 FFT 数组坐标更容易理解。
    center_exclusion_radius
        自动搜索时排除 DC 的半径，单位 FFT pixel；``None`` 自动设置。
    filter_sigma
        Gaussian 一级边带滤波宽度，单位 FFT pixel；``None`` 自动设置。
    apply_hann
        是否在 FFT 前乘二维 periodic Hann 窗。
    phase_sign
        ``+1`` 或 ``-1``，用于统一共轭边带导致的相位符号。

    返回
    ----
    TakedaFTResult
        包裹相位、复场幅值/置信度以及频谱、滤波窗等完整诊断量。
        相位展开由主流水线统一完成。
    """
    if phase_sign not in (-1, 1):
        raise ValueError("phase_sign 必须是 +1 或 -1")
    if sideband_center is not None and carrier_cycles is not None:
        raise ValueError("sideband_center 与 carrier_cycles 不能同时设置")

    image = np.asarray(I, dtype=np.float64)
    prepared = takeda_ft_spectrum(image, mask=mask, apply_hann=apply_hann)
    valid = prepared.valid
    aperture = prepared.aperture
    n_rows, n_cols = image.shape
    window = prepared.window
    spectrum = prepared.spectrum
    magnitude = prepared.magnitude
    center_row, center_col = n_rows // 2, n_cols // 2
    col_grid, row_grid = np.meshgrid(
        np.arange(n_cols, dtype=np.float64),
        np.arange(n_rows, dtype=np.float64))

    if center_exclusion_radius is None:
        dc_radius = float(max(8, round(0.06 * min(n_rows, n_cols))))
    else:
        dc_radius = float(center_exclusion_radius)
        if not np.isfinite(dc_radius) or dc_radius < 0:
            raise ValueError("center_exclusion_radius 必须是非负有限数")

    if carrier_cycles is not None:
        cycles = np.asarray(carrier_cycles, dtype=np.float64).reshape(-1)
        if cycles.size != 2 or not np.isfinite(cycles).all():
            raise ValueError("carrier_cycles 必须是有限的 (fx, fy)")
        peak_col = center_col + int(round(float(cycles[0])))
        peak_row = center_row + int(round(float(cycles[1])))
    elif sideband_center is not None:
        peak = np.asarray(sideband_center, dtype=np.float64).reshape(-1)
        if peak.size != 2 or not np.isfinite(peak).all():
            raise ValueError("sideband_center 必须是有限的 (row, col)")
        peak_row, peak_col = int(round(float(peak[0]))), int(round(float(peak[1])))
    else:
        search = magnitude.copy()
        distance = np.hypot(row_grid - center_row, col_grid - center_col)
        search[distance <= dc_radius] = 0.0
        canonical_half = ((col_grid > center_col) |
                          ((col_grid == center_col) & (row_grid < center_row)))
        search[~canonical_half] = 0.0
        edge_margin = max(2, round(0.01 * min(n_rows, n_cols)))
        search[:edge_margin, :] = 0.0
        search[-edge_margin:, :] = 0.0
        search[:, :edge_margin] = 0.0
        search[:, -edge_margin:] = 0.0
        peak_index = int(np.argmax(search))
        if float(search.flat[peak_index]) <= 0.0:
            raise ValueError("未找到可用一级边带，请手动设置载频 fx/fy")
        peak_row, peak_col = np.unravel_index(peak_index, image.shape)

    if not (0 <= peak_row < n_rows and 0 <= peak_col < n_cols):
        raise ValueError("手动一级边带中心超出 FFT 数组范围")
    carrier_distance = float(np.hypot(
        peak_row - center_row, peak_col - center_col))
    if carrier_distance <= max(1.0, dc_radius if carrier_cycles is None and
                               sideband_center is None else 1.0):
        raise ValueError("一级边带距离 DC 太近，无法可靠分离")

    if filter_sigma is None:
        sigma = float(max(3.0, 0.22 * carrier_distance))
    else:
        sigma = float(filter_sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("filter_sigma 必须是正有限数")

    sideband_filter = np.exp(-(
        (row_grid - peak_row) ** 2 + (col_grid - peak_col) ** 2
    ) / (2.0 * sigma ** 2))
    filtered_spectrum = spectrum * sideband_filter
    baseband_spectrum = np.roll(
        filtered_spectrum,
        shift=(center_row - peak_row, center_col - peak_col),
        axis=(0, 1))
    complex_field = np.fft.ifft2(np.fft.ifftshift(baseband_spectrum))
    phase_wrapped = float(phase_sign) * np.angle(complex_field)
    amplitude_full = np.abs(complex_field)
    reference_amplitude = float(np.percentile(amplitude_full[valid], 95))
    if reference_amplitude <= np.finfo(float).eps:
        confidence_full = np.zeros_like(amplitude_full)
    else:
        confidence_full = np.minimum(amplitude_full / reference_amplitude, 1.0)

    amplitude = np.where(aperture, amplitude_full, np.nan)
    confidence = np.where(aperture, confidence_full, np.nan)
    return TakedaFTResult(
        phase_wrapped=phase_wrapped,
        amplitude=amplitude,
        confidence=confidence,
        spectrum_log=prepared.spectrum_log,
        sideband_filter=sideband_filter,
        filtered_spectrum_log=np.log1p(np.abs(filtered_spectrum)),
        sideband_center=(int(peak_row), int(peak_col)),
        carrier_cycles=(int(peak_col - center_col), int(peak_row - center_row)),
        filter_sigma=sigma,
        center_exclusion_radius=dc_radius,
        complex_field=complex_field,
        window=window,
        filter_too_wide=bool(sigma >= 0.48 * carrier_distance),
    )


# ---------------------------------------------------------------- 质量图引导相位展开

def phase_derivative_variance(wrapped: np.ndarray, k: int = 3) -> np.ndarray:
    """
    相位导数方差 (Phase Derivative Variance, PDV) 质量图。

    在 k×k 邻域内统计 x/y 方向包裹相位梯度的方差；值越小代表相位越可靠。
    残差点、噪声、断裂处质量差（值大），将被放到展开顺序的最后。

    参考: Ghiglia & Pritt, "Two-Dimensional Phase Unwrapping", §5.1。
    """
    wrapped = np.asarray(wrapped, dtype=np.float64)
    k = max(3, int(k))
    dx = np.zeros_like(wrapped)
    dy = np.zeros_like(wrapped)
    if wrapped.shape[1] > 1:
        dx[:, :-1] = wrapped[:, 1:] - wrapped[:, :-1]
    if wrapped.shape[0] > 1:
        dy[:-1, :] = wrapped[1:, :] - wrapped[:-1, :]
    # 梯度包裹到 [-pi, pi]
    dx = (dx + np.pi) % (2.0 * np.pi) - np.pi
    dy = (dy + np.pi) % (2.0 * np.pi) - np.pi

    varx = uniform_filter(dx * dx, k) - uniform_filter(dx, k) ** 2
    vary = uniform_filter(dy * dy, k) - uniform_filter(dy, k) ** 2
    q = np.sqrt(np.maximum(varx, 0.0) + np.maximum(vary, 0.0))
    return q


def _qg_core(w_flat, q_flat, m_flat, nr, nc, seed):
    """质量图引导展开的主循环（纯 Python 热路径，已尽量优化）。

    用 in_heap 去重：每个像素最多只入堆一次，堆操作从 ~4n 降到 ~n。
    """
    import heapq
    n = nr * nc
    solved = bytearray(n)          # 0/1，比 list[bool] 更快
    in_heap = bytearray(n)
    out = [0.0] * n
    twopi = 2.0 * math.pi
    heappush = heapq.heappush
    heappop = heapq.heappop

    def push(i):
        if not solved[i] and not in_heap[i] and m_flat[i]:
            in_heap[i] = 1
            heappush(heap, (q_flat[i], i))

    solved[seed] = 1
    out[seed] = w_flat[seed]
    heap = []
    r0, c0 = divmod(seed, nc)
    if c0 > 0:
        push(seed - 1)
    if c0 < nc - 1:
        push(seed + 1)
    if r0 > 0:
        push(seed - nc)
    if r0 < nr - 1:
        push(seed + nc)

    while heap:
        _, i = heappop(heap)
        in_heap[i] = 0
        if solved[i]:
            continue
        r, c = divmod(i, nc)
        # 找一个已展开的邻点作为参考
        if c > 0 and solved[i - 1]:
            base = out[i - 1]
        elif c < nc - 1 and solved[i + 1]:
            base = out[i + 1]
        elif r > 0 and solved[i - nc]:
            base = out[i - nc]
        elif r < nr - 1 and solved[i + nc]:
            base = out[i + nc]
        else:
            continue                    # 理论不会出现，防御
        w = w_flat[i]
        w = w + twopi * round((base - w) / twopi)   # 补 2π 整数倍
        out[i] = w
        solved[i] = 1
        # 四个邻点入堆（质量好的先出）
        if c > 0:
            push(i - 1)
        if c < nc - 1:
            push(i + 1)
        if r > 0:
            push(i - nc)
        if r < nr - 1:
            push(i + nc)
    return out


def qg_dunwrap(wrapped: np.ndarray, mask: np.ndarray,
               quality: Optional[np.ndarray] = None,
               k: int = 3,
               seed: Optional[tuple] = None) -> np.ndarray:
    """
    质量图引导的二维掩膜相位展开。

    原理:
      1. 计算 PDV 质量图（越小越好，可传入自定义质量图）；
      2. 从种子点（默认掩膜内质量最好的像素）出发；
      3. 最大堆(最小质量值优先)始终先展开当前最优质量的邻点，
         相位差超过 ±pi 时补 2π 整数倍；
      4. 只在 mask==1 区域内展开。

    参数:
        wrapped : 包裹相位 (m x n)
        mask    : 二值掩膜 (m x n)，1=有效
        quality : 可选质量图 (m x n)，值越小越先展开；None 时自动计算 PDV
        k       : PDV 邻域尺寸
        seed    : 起始点 (row, col)；None 时取掩膜内质量最好的像素

    返回:
        展开后的相位（掩膜外为 0）。
    """
    wrapped = np.asarray(wrapped, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if wrapped.shape != mask.shape:
        raise ValueError("wrapped 与 mask 尺寸不一致")
    if not mask.any():
        return np.zeros_like(wrapped)

    if quality is None:
        quality = phase_derivative_variance(wrapped, k)
    else:
        quality = np.asarray(quality, dtype=np.float64)
        if quality.shape != wrapped.shape:
            raise ValueError("quality 与 wrapped 尺寸不一致")

    if seed is None:
        q_full = quality.copy()
        q_full[~mask] = np.inf
        seed_flat = int(np.argmin(q_full))          # 质量最好的像素
    else:
        r0, c0 = int(seed[0]), int(seed[1])
        if not (0 <= r0 < wrapped.shape[0] and 0 <= c0 < wrapped.shape[1]):
            raise ValueError("种子点超出图像范围")
        if not mask[r0, c0]:
            # 种子不在掩膜内：取离它最近的掩膜点
            ys, xs = np.nonzero(mask)
            d2 = (ys - r0) ** 2 + (xs - c0) ** 2
            kk = int(np.argmin(d2))
            r0, c0 = int(ys[kk]), int(xs[kk])
        seed_flat = r0 * wrapped.shape[1] + c0

    nr, nc = wrapped.shape
    w_flat = wrapped.reshape(-1).tolist()
    q_flat = quality.reshape(-1).tolist()
    m_flat = mask.reshape(-1).tolist()
    out = _qg_core(w_flat, q_flat, m_flat, nr, nc, seed_flat)
    return np.asarray(out, dtype=np.float64).reshape(nr, nc)


def zStd(rho: np.ndarray, theta: np.ndarray, max_term: int):
    """
    标准 Zernike 多项式（Noll 顺序与归一化，对应 MATLAB zStd.m）。

    归一化: m=0 时 sqrt(n+1)；m!=0 时 sqrt(2)*sqrt(n+1)。
    序号: Z1=piston, Z2=2ρcosθ, Z3=2ρsinθ, Z4=defocus, ...
    同一 n 内按较小的 |m| 优先；偶数 j 使用 cos(mθ)，奇数 j 使用
    sin(mθ)，与 Noll (JOSA 66, 207--211, 1976) 的表 I 一致。

    径向多项式在这里按阶乘系数的有限求和式直接计算，不是递推算法。
    原 MATLAB 文件把归一化公式溯源到《光学车间检测》第 383 页式
    (13.47)，但未记录具体中文版本；可核验的英文第三版对应 Mahajan,
    "Zernike Polynomial and Wavefront Fitting," Optical Shop Testing,
    3rd ed., Chap. 13, pp. 498--546 (2007), doi:10.1002/9780470135976.ch13.

    参数:
        rho      : 归一化半径 (N,)，0<=rho<=1
        theta    : 极角 (N,)
        max_term : 最大项数

    返回:
        zMatrix (N x max_term), nVec, mVec, jVec
    """
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if rho.shape != theta.shape:
        raise ValueError("rho 与 theta 尺寸不一致")
    if int(max_term) < 1:
        raise ValueError("max_term 必须为正整数")
    max_term = int(max_term)
    points = rho.size

    zMatrix = np.zeros((points, max_term), dtype=np.float64)
    nVec = np.zeros(max_term, dtype=int)
    mVec = np.zeros(max_term, dtype=int)
    jVec = np.arange(1, max_term + 1, dtype=int)

    for j in range(1, max_term + 1):
        # ---- 确定径向阶数 n ----
        n = 0
        for nn in range(j):
            if j > nn * (nn + 1) // 2 and j <= (nn + 1) * (nn + 2) // 2:
                n = nn
                break
        # ---- 确定角向频率 m ----
        i = n * (n + 1) // 2
        m = 0
        for mm in range(n + 1):
            if (n - mm) % 2 == 0:
                i += 1
                if i == j:
                    m = mm
                    break
                if mm != 0:
                    i += 1
                    if i == j:
                        m = mm
                        break
        # ---- 径向多项式 R_n^m ----
        radial = np.zeros(points, dtype=np.float64)
        for s in range((n - m) // 2 + 1):
            coef = ((-1) ** s * math.factorial(n - s) /
                    (math.factorial(s) *
                     math.factorial((n + m) // 2 - s) *
                     math.factorial((n - m) // 2 - s)))
            radial += coef * rho ** (n - 2 * s)

        if m == 0:
            zMatrix[:, j - 1] = math.sqrt(n + 1) * radial
        elif j % 2 == 0:
            zMatrix[:, j - 1] = (math.sqrt(2.0) * math.sqrt(n + 1) *
                                 radial * np.cos(m * theta))
        else:
            zMatrix[:, j - 1] = (math.sqrt(2.0) * math.sqrt(n + 1) *
                                 radial * np.sin(m * theta))
        nVec[j - 1] = n
        mVec[j - 1] = m
    return zMatrix, nVec, mVec, jVec


# ---------------------------------------------------------------- 掩膜

@dataclass
class MaskInfo:
    """圆形孔径掩膜的全部信息。"""
    cx: float            # 圆心列坐标 (col)
    cy: float            # 圆心行坐标 (row)
    maskr: int           # 半边长 (像素) = 孔径半径
    rowmin: int = 0
    rowmax: int = 0
    colmin: int = 0
    colmax: int = 0
    mask: np.ndarray = field(default=None)    # 全口径 (R<=1)
    X0: np.ndarray = field(default=None)
    Y0: np.ndarray = field(default=None)
    R0: np.ndarray = field(default=None)


def auto_detect_circle(image: np.ndarray):
    """
    自动检测圆形孔径，返回 (cx, cy, radius)。

    cx=列坐标, cy=行坐标。失败时返回 None。
    """
    if cv2 is None:
        return None
    img = np.asarray(image, dtype=np.float64)
    vmax = img.max()
    if vmax <= 0:
        return None
    img8 = np.uint8(np.clip(img * (255.0 / vmax), 0, 255))

    # 找到"占画面大比例且接近圆形"的轮廓
    blur = cv2.GaussianBlur(img8, (5, 5), 0)
    candidates = []
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, th = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            h, w = img.shape
            if area < 0.05 * h * w:      # 太小的轮廓忽略
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            # 圆度：轮廓面积 / 外接圆面积 接近 1
            circularity = area / (np.pi * r * r)
            if circularity > 0.55:
                candidates.append((circularity, x, y, r))

    if candidates:
        # 取最圆的圆；孔径不应占据整幅图像（过滤误检的全图轮廓）
        candidates.sort(reverse=True)
        for circ, x, y, r in candidates:
            if circ >= 0.70 and r <= 0.65 * max(img.shape):
                return float(x), float(y), float(r)

    # 备用：霍夫圆检测
    circles = cv2.HoughCircles(img8, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=max(img.shape) // 2,
                               param1=100, param2=30,
                               minRadius=max(16, int(0.15 * min(img.shape))),
                               maxRadius=int(0.6 * max(img.shape)))
    if circles is not None:
        x, y, r = circles[0, 0]
        return float(x), float(y), float(r)
    return None


def estimate_fringe_period(image: np.ndarray) -> int:
    """
    用 FFT 自动估计条纹周期（像素）。

    对灰度图去均值后做二维 FFT，分别把功率谱沿行/列方向累加，
    取较强方向上的主峰频率，换算成条纹周期。适合竖直/水平载波条纹。
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image 必须是二维数组")
    img = img - img.mean()
    m, n = img.shape
    power = np.abs(np.fft.fft2(img)) ** 2

    col_prof = power.sum(axis=0)
    row_prof = power.sum(axis=1)
    col_prof[0] = 0.0
    row_prof[0] = 0.0

    def _best(prof, length):
        # 排除 bin0/1（直流与孔径包络的极低频伪峰），
        # 只搜索周期 <= length/2 的载频范围。
        half = length // 2
        if half < 4:
            return None
        idx = int(np.argmax(prof[2:half])) + 2
        return idx, float(prof[idx])

    cb = _best(col_prof, n)
    rb = _best(row_prof, m)
    if cb is None and rb is None:
        raise ValueError("图像太小，无法估计条纹周期")
    if cb is None:
        idx, _ = rb
        return int(round(m / idx))
    if rb is None:
        idx, _ = cb
        return int(round(n / idx))

    # 取功率更强的方向（竖直条纹主峰在列方向，水平条纹在行方向）
    if rb[1] > cb[1]:
        return int(round(m / rb[0]))
    return int(round(n / cb[0]))


def simulate_fizeau_pair(size: int = 720,
                         cx: Optional[float] = None,
                         cy: Optional[float] = None,
                         radius: float = 270.0,
                         period: int = 15,
                         wavelength_nm: float = 635.0,
                         noise: float = 1.5,
                         seed: int = 20231031,
                         ref_aberrations: Optional[dict] = None,
                         test_aberrations: Optional[dict] = None) -> dict:
    """
    仿真干涉图生成器（与 examples/generate_simulation.py 同一物理模型）。

    返回 dict:
        I_ref, I_test : 参考/待测干涉图 (float64, 0~255)
        h_ref, h_test : 两个元件真实面形 (nm)
        h_diff        : 真实相位差面形 = 参考 - 待测 (nm)
        mask          : 圆形孔径掩膜 (bool)
        params        : 全部仿真参数 (含 ref_aberrations / test_aberrations)
    """
    size = int(size)
    if size < 64:
        raise ValueError("图像尺寸过小")
    cx = float(cx) if cx is not None else size / 2.0
    cy = float(cy) if cy is not None else size / 2.0
    radius = float(radius)
    period = int(period)
    if period < 3:
        raise ValueError("条纹周期必须 >= 3")

    ref_aberrations = dict(ref_aberrations or {4: 5.0, 6: 4.0, 8: 3.0})
    test_aberrations = dict(test_aberrations or {
        4: 40.0, 5: 30.0, 6: 25.0, 7: 20.0, 9: 12.0, 11: 15.0})
    maxj = max([*ref_aberrations.keys(), *test_aberrations.keys()] or [4])

    rng = np.random.default_rng(seed)
    xs = np.arange(size, dtype=np.float64)
    X, Y = np.meshgrid(xs, xs)
    u = (X - cx) / radius
    v = (Y - cy) / radius
    rho = np.hypot(u, v)
    theta = np.arctan2(v, u)
    mask = rho <= 1.0

    Zfull, _, _, _ = zStd(np.clip(rho, 0.0, 1.0).ravel(),
                          theta.ravel(), maxj)
    Zfull = Zfull.reshape(size, size, maxj)

    def surface(amps: dict) -> np.ndarray:
        h = np.zeros((size, size), dtype=np.float64)
        for j, a in amps.items():
            h += float(a) * Zfull[:, :, j - 1]
        return h * mask

    h_ref = surface(ref_aberrations)
    h_test = surface(test_aberrations)
    h_diff = (h_ref - h_test) * mask

    # 竖直载波条纹，右边相位小（与软件 s=21 约定一致）
    carrier = -2.0 * np.pi * (X - cx) / period

    def interferogram(h: np.ndarray) -> np.ndarray:
        phase = carrier + 4.0 * np.pi * h / wavelength_nm
        I0 = 50.0 + 100.0 * np.exp(-(rho / 0.75) ** 2)
        I1 = 0.65 * I0 * (1.0 - 0.10 * rho ** 2)
        out = (I0 + I1 * np.cos(phase)) * mask
        if noise > 0:
            out = out + rng.standard_normal(out.shape) * float(noise)
        return out

    return {
        "I_ref": interferogram(h_ref),
        "I_test": interferogram(h_test),
        "h_ref": h_ref,
        "h_test": h_test,
        "h_diff": h_diff,
        "mask": mask,
        "params": {
            "size": size, "cx": cx, "cy": cy, "radius": radius,
            "period": period, "wavelength_nm": float(wavelength_nm),
            "noise": float(noise), "seed": int(seed),
            "ref_aberrations": dict(ref_aberrations),
            "test_aberrations": dict(test_aberrations),
        },
    }


def build_mask(image_shape, cx: float, cy: float, maskr: int) -> MaskInfo:
    """
    由圆心 (cx=列, cy=行) 与半径 maskr 构建 ROI 与归一化掩膜。

    与原 MATLAB 流程一致：
      ROI 为以圆心为中心、边长 2*maskr+1 的正方形；
      内部归一化坐标 [-1,1]，mask = R<=1（完整有效口径）。
    若半径超出图像边界会自动收缩，并给出提示信息。
    """
    rows, cols = int(image_shape[0]), int(image_shape[1])
    cx = float(cx)
    cy = float(cy)
    maskr = int(maskr)
    if maskr < 5:
        raise ValueError("掩膜半径过小 (需 >= 5 像素)")

    # 边界检查并收缩
    limit = int(min(cy, rows - 1 - cy, cx, cols - 1 - cx))
    clipped = False
    if limit < maskr:
        maskr = limit
        clipped = True
    if maskr < 5:
        raise ValueError("圆心太靠近图像边缘，无法建立掩膜")

    rowmin = int(round(cy)) - maskr
    rowmax = int(round(cy)) + maskr
    colmin = int(round(cx)) - maskr
    colmax = int(round(cx)) + maskr

    row = rowmax - rowmin + 1
    column = colmax - colmin + 1
    x0 = np.linspace(-1.0, 1.0, column)
    y0 = np.linspace(-1.0, 1.0, row)
    X0, Y0 = np.meshgrid(x0, y0)
    R0 = np.sqrt(X0 ** 2 + Y0 ** 2)
    mask = (R0 <= 1.0)

    return MaskInfo(cx=cx, cy=cy, maskr=maskr,
                    rowmin=rowmin, rowmax=rowmax,
                    colmin=colmin, colmax=colmax,
                    mask=mask,
                    X0=X0, Y0=Y0, R0=R0)


# ---------------------------------------------------------------- Zernike 名称

_PHASE_METHOD_LABELS = {
    "takeda": "傅里叶变换FT",
    "masked": "空间载波相移SCPS",
    "classic": "经典空间相移 (s_T_shift)",
    "adapt2": "最小二乘空间载波相移LS-SCPS",
    "wft2": "加窗傅里叶变换WFT",
}


def phase_method_label(method: str) -> str:
    return _PHASE_METHOD_LABELS.get(method, method)


ZERNIKE_NAMES = {
    1:  ("Piston",              "活塞（常数项）"),
    2:  ("Tilt X",              "X 倾斜"),
    3:  ("Tip Y",               "Y 倾斜"),
    4:  ("Defocus",             "离焦"),
    5:  ("Astigmatism 45°",     "45°像散"),
    6:  ("Astigmatism 0°",      "0°像散"),
    7:  ("Coma Y",              "Y 彗差"),
    8:  ("Coma X",              "X 彗差"),
    9:  ("Trefoil 30°",         "三叶像差 30°"),
    10: ("Trefoil 0°",          "三叶像差 0°"),
    11: ("Primary spherical",   "初级球差"),
    12: ("Secondary astigmatism 0°",  "二级像散 0°"),
    13: ("Secondary astigmatism 45°", "二级像散 45°"),
    14: ("Tetrafoil 0°",          "四叶像差 0°"),
    15: ("Tetrafoil 22.5°",       "四叶像差 22.5°"),
}


def zernike_name(j: int) -> str:
    if j in ZERNIKE_NAMES:
        return f"{ZERNIKE_NAMES[j][0]} ({ZERNIKE_NAMES[j][1]})"
    return f"Z{j}"


# ---------------------------------------------------------------- 完整流水线

ProgressFn = Callable[[int, str], None]


@dataclass
class ProcessResult:
    """流水线输出的全部结果。"""
    ref_path: str
    test_path: str
    period: int
    wavelength_nm: float
    double_pass: float
    max_term: int
    scale_nm: float                       # 弧度 -> nm 的换算系数
    mask_info: MaskInfo
    truncated_test: np.ndarray            # 待测元件截断相位 (pd)
    truncated_ref: np.ndarray             # 参考元件截断相位 (pc)
    unwrapped_test: np.ndarray            # 待测元件展开相位 (phaseCc)
    unwrapped_ref: np.ndarray             # 参考元件展开相位 (phasedc)
    phase: np.ndarray                     # 相位差 = 参考 - 待测
    coefficients: np.ndarray              # Zernike 系数 (rad)
    coefficients_nm: np.ndarray           # Zernike 系数 (nm)
    nVec: np.ndarray
    mVec: np.ndarray
    jVec: np.ndarray
    n_remove: int = 0
    phase_method: str = "masked"
    residuals: list = field(default_factory=list)  # [{'k','W','rms','pv'}, ...]
    ft_diagnostics: Optional[dict] = None          # {'test','reference'} -> TakedaFTResult
    ft_settings: dict = field(default_factory=dict)
    wft_diagnostics: Optional[dict] = None         # {'test','reference'} -> WFT2Result

    @property
    def global_rms(self) -> float:
        if self.residuals and self.residuals[0]['k'] == 0:
            return self.residuals[0]['rms']
        return float('nan')

    @property
    def global_pv(self) -> float:
        if self.residuals and self.residuals[0]['k'] == 0:
            return self.residuals[0]['pv']
        return float('nan')


def process_fizeau(ref_path: str, test_path: str, period: int,
                   cx: float, cy: float, maskr: int,
                   wavelength_nm: float = 635.0,
                   double_pass: float = 2.0,
                   max_term: int = 80,
                   n_remove: int = 11,
                   phase_method: str = "masked",
                   ft_carrier_cycles: Optional[tuple[float, float]] = None,
                   ft_center_exclusion_radius: Optional[float] = None,
                   ft_filter_sigma: Optional[float] = None,
                   ft_apply_hann: bool = True,
                   ft_phase_sign: int = 1,
                   progress: Optional[ProgressFn] = None) -> ProcessResult:
    """
    完整处理流程（与 MATLAB Main.m 一致）。

    参数:
        ref_path      : 参考元件干涉图路径
        test_path     : 待测元件干涉图路径
        period        : 条纹周期的像素数
        cx, cy        : 掩膜圆心 (列, 行)
        maskr         : 掩膜半径 (像素)
        wavelength_nm : 光源波长 (nm)，默认 635 (HeNe)
        double_pass   : 双程因子，默认 2 (双程反射)
        max_term      : Zernike 拟合项数，默认 80
        n_remove      : 分解项数：依次输出去前 1..n_remove 项残差，默认 11
        phase_method  : 条纹相位提取算法:
                        "takeda"(经典 FT) / "masked"(空间载波相移, 默认)
                        / "classic"(经典 s_T_shift) / "adapt2"(自适应空间相移)
                        / "wft2"(加窗傅里叶滤波)
        ft_carrier_cycles : FT 手动载频 (fx, fy)；None 时自动搜索边带
        ft_center_exclusion_radius : FT 自动搜索 DC 排除半径；None 时自动
        ft_filter_sigma : FT Gaussian 边带宽度；None 时自动
        ft_apply_hann : FT 前是否使用二维 Hann 窗
        ft_phase_sign : FT 相位符号，+1 或 -1
        progress      : 回调 progress(百分比, 消息)

    返回:
        ProcessResult
    """
    max_term = int(max_term)
    n_remove = int(n_remove)
    if n_remove < 0:
        raise ValueError("去项上限不能小于 0")
    if n_remove > max_term:
        raise ValueError(
            f"去项上限 {n_remove} 不能超过 Zernike 拟合项数 {max_term}")
    phase_method = str(phase_method).lower()
    if phase_method not in ("takeda", "masked", "classic", "adapt2", "wft2"):
        raise ValueError(
            "相位算法必须是傅里叶变换FT、空间载波相移SCPS、"
            "最小二乘空间载波相移LS-SCPS或加窗傅里叶变换WFT")

    def report(pct: int, msg: str):
        if progress is not None:
            progress(int(pct), msg)

    report(2, "读取图像…")
    Im_ref = read_image(ref_path)
    Im_test = read_image(test_path)
    if Im_ref.shape != Im_test.shape:
        raise ValueError(
            f"两张干涉图尺寸不一致: 参考 {Im_ref.shape} / 待测 {Im_test.shape}")

    report(8, "建立圆形掩膜…")
    minfo = build_mask(Im_ref.shape, cx, cy, maskr)
    mask = minfo.mask
    rmin, rmax = minfo.rowmin, minfo.rowmax
    cmin, cmax = minfo.colmin, minfo.colmax

    Imaged = Im_test[rmin:rmax + 1, cmin:cmax + 1]   # 待测元件 (原程序 name1)
    Imagec = Im_ref[rmin:rmax + 1, cmin:cmax + 1]    # 参考元件 (原程序 name2)

    ft_diagnostics = None
    ft_settings = {}
    wft_diagnostics = None
    if phase_method == "takeda":
        report(14, "待测元件相位提取（傅里叶变换FT）…")
        ft_test = takeda_ft_phase(
            Imaged, mask=mask, carrier_cycles=ft_carrier_cycles,
            center_exclusion_radius=ft_center_exclusion_radius,
            filter_sigma=ft_filter_sigma, apply_hann=ft_apply_hann,
            phase_sign=ft_phase_sign)
        pd = ft_test.phase_wrapped
        report(32, "参考元件相位提取（傅里叶变换FT）…")
        ft_ref = takeda_ft_phase(
            Imagec, mask=mask, carrier_cycles=ft_carrier_cycles,
            center_exclusion_radius=ft_center_exclusion_radius,
            filter_sigma=ft_filter_sigma, apply_hann=ft_apply_hann,
            phase_sign=ft_phase_sign)
        pc = ft_ref.phase_wrapped
        ft_diagnostics = {"test": ft_test, "reference": ft_ref}
        ft_settings = {
            "carrier_mode": "manual" if ft_carrier_cycles is not None else "auto",
            "requested_carrier_cycles": ft_carrier_cycles,
            "center_exclusion_radius": ft_center_exclusion_radius,
            "filter_sigma": ft_filter_sigma,
            "apply_hann": bool(ft_apply_hann),
            "phase_sign": int(ft_phase_sign),
        }
    elif phase_method == "masked":
        report(14, "待测元件条纹相位提取（空间载波相移SCPS）…")
        pd = masked_s_T_shift(Imaged, mask, int(period), 1, 21)
        report(32, "参考元件条纹相位提取（空间载波相移SCPS）…")
        pc = masked_s_T_shift(Imagec, mask, int(period), 1, 21)
    elif phase_method == "classic":
        report(14, "待测元件条纹相位提取 (经典相移 s_T_shift)…")
        pd = s_T_shift(Imaged, int(period), 1, 21)
        report(32, "参考元件条纹相位提取 (经典相移 s_T_shift)…")
        pc = s_T_shift(Imagec, int(period), 1, 21)
    elif phase_method == "adapt2":
        report(14, "待测元件条纹相位提取（最小二乘空间载波相移LS-SCPS）…")
        pd = adapt2_phase(Imaged, int(period), mask=mask)
        report(32, "参考元件条纹相位提取（最小二乘空间载波相移LS-SCPS）…")
        pc = adapt2_phase(Imagec, int(period), mask=mask)
    else:  # wft2
        # 两幅图共用从参考图估计的载频分支，防止分别选到一对共轭峰。
        carrier = _wft2_carrier(Imagec, mask, int(period))
        report(14, "待测元件条纹相位提取（加窗傅里叶变换WFT）…")
        wft_test = wft2_phase(
            Imaged, int(period), mask=mask, carrier_rad=carrier,
            return_result=True)
        pd = wft_test.phase_wrapped
        report(32, "参考元件条纹相位提取（加窗傅里叶变换WFT）…")
        wft_ref = wft2_phase(
            Imagec, int(period), mask=mask, carrier_rad=carrier,
            return_result=True)
        pc = wft_ref.phase_wrapped
        wft_diagnostics = {"test": wft_test, "reference": wft_ref}

    report(50, "待测元件相位展开 (质量图引导)…")
    # 两幅图共用参考图的质量图和同一种子点，使两次展开走相同路径；
    # 同时按质量从优到劣展开，避免误差穿过低质量区域。
    q_shared = phase_derivative_variance(pc, 3)
    if wft_diagnostics is not None:
        # qg_dunwrap 约定“值越小越可靠”。用两幅图的联合 WFT
        # 置信度放大低调制度/孔径边缘处的 PDV 代价。
        joint_confidence = np.sqrt(
            wft_diagnostics["test"].confidence
            * wft_diagnostics["reference"].confidence)
        q_shared = q_shared / np.maximum(joint_confidence, 0.05)
    seed_pt = (mask.shape[0] // 2, mask.shape[1] // 2)
    qd0 = qg_dunwrap(pd, mask, quality=q_shared, seed=seed_pt)
    phaseCc = qd0 * mask
    report(68, "参考元件相位展开 (质量图引导)…")
    qc0 = qg_dunwrap(pc, mask, quality=q_shared, seed=seed_pt)
    phasedc = qc0 * mask

    phase = phasedc - phaseCc          # 相位差 = 参考 - 待测

    report(80, f"计算前 {max_term} 项 Zernike 多项式…")
    index = np.nonzero(mask.ravel())[0]           # 只拟合掩膜内的点
    rho_m = minfo.R0.ravel()[index]
    theta_m = np.arctan2(minfo.Y0.ravel()[index], minfo.X0.ravel()[index])
    Zp, nVec, mVec, jVec = zStd(rho_m, theta_m, max_term)

    report(90, "最小二乘拟合 Zernike 系数…")
    phase_m = phase.ravel()[index]
    Cc = np.linalg.lstsq(Zp, phase_m, rcond=None)[0]

    scale_nm = float(wavelength_nm) / (2.0 * float(double_pass) * np.pi)

    report(94, "生成去项残差图…")
    M, N = phase.shape
    residuals = []
    for i in range(0, n_remove + 1):
        if i == 0:
            resid_m = phase_m * scale_nm
        else:
            comb_fit = Zp[:, :i] @ Cc[:i]
            resid_m = (phase_m - comb_fit) * scale_nm
        W = np.full((M, N), np.nan, dtype=np.float64)
        Wf = W.ravel()
        Wf[index] = resid_m
        W = Wf.reshape(M, N)
        # 面形统计必须覆盖完整有效口径；不能用内径 80% 降低 RMS/PV。
        W[~mask] = np.nan
        valid = W[mask]
        rms = float(np.std(valid))
        pv = float(valid.max() - valid.min())
        residuals.append({'k': i, 'W': W, 'rms': rms, 'pv': pv})
        report(94 + 6 * (i + 1) // (n_remove + 1), f"残差图 {i}/{n_remove}")

    report(100, "处理完成")
    return ProcessResult(
        ref_path=ref_path, test_path=test_path,
        period=int(period), wavelength_nm=float(wavelength_nm),
        double_pass=float(double_pass), max_term=max_term,
        n_remove=n_remove, phase_method=phase_method,
        scale_nm=scale_nm, mask_info=minfo,
        truncated_test=np.where(mask, pd, np.nan),
        truncated_ref=np.where(mask, pc, np.nan),
        unwrapped_test=np.where(mask, phaseCc, np.nan),
        unwrapped_ref=np.where(mask, phasedc, np.nan),
        phase=np.where(mask, phase, np.nan),
        coefficients=Cc, coefficients_nm=Cc * scale_nm,
        nVec=nVec, mVec=mVec, jVec=jVec,
        residuals=residuals,
        ft_diagnostics=ft_diagnostics, ft_settings=ft_settings,
        wft_diagnostics=wft_diagnostics,
    )


# ---------------------------------------------------------------- 报告文本

def build_report(res: ProcessResult) -> str:
    """生成文字报告（中文，utf-8）。"""
    mi = res.mask_info
    lines = []
    lines.append("=" * 62)
    lines.append("菲索干涉仪数据处理报告")
    lines.append("=" * 62)
    import datetime
    lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("【处理参数】")
    lines.append(f"  参考元件干涉图: {os.path.basename(res.ref_path)}")
    lines.append(f"  待测元件干涉图: {os.path.basename(res.test_path)}")
    lines.append(f"  条纹周期      : {res.period} 像素")
    lines.append(f"  相位提取算法  : {phase_method_label(res.phase_method)}")
    if res.phase_method == "takeda" and res.ft_diagnostics:
        test_ft = res.ft_diagnostics["test"]
        ref_ft = res.ft_diagnostics["reference"]
        mode = res.ft_settings.get("carrier_mode", "auto")
        lines.append(f"  FT 边带模式   : {'自动检测' if mode == 'auto' else '手动载频'}")
        lines.append(
            f"  待测 FT 载频  : fx={test_ft.carrier_cycles[0]}, "
            f"fy={test_ft.carrier_cycles[1]} cycles/image")
        lines.append(
            f"  参考 FT 载频  : fx={ref_ft.carrier_cycles[0]}, "
            f"fy={ref_ft.carrier_cycles[1]} cycles/image")
        lines.append(
            f"  FT Gaussian σ : 待测 {test_ft.filter_sigma:.3f}, "
            f"参考 {ref_ft.filter_sigma:.3f} FFT pixel")
        lines.append(
            f"  FT Hann/符号  : {'开' if res.ft_settings.get('apply_hann', True) else '关'}"
            f" / {res.ft_settings.get('phase_sign', 1):+d}")
    lines.append(f"  光源波长      : {res.wavelength_nm:g} nm")
    lines.append(f"  双程因子      : {res.double_pass:g}")
    lines.append(f"  Zernike 项数  : {res.max_term}")
    lines.append(f"  掩膜圆心      : ({mi.cx:.1f}, {mi.cy:.1f})  半径 {mi.maskr} 像素")
    lines.append("")
    lines.append("【面形结果】(单位: nm)")
    for r in res.residuals:
        k = r['k']
        if k == 0:
            lines.append(f"  全局(未去项)      : RMS = {r['rms']:.3f}  PV = {r['pv']:.3f}")
        else:
            lines.append(f"  去前 {k:2d} 项后      : RMS = {r['rms']:.3f}  PV = {r['pv']:.3f}")
    lines.append("")
    lines.append(f"【前 {min(15, res.max_term)} 项 Zernike 系数】")
    lines.append(f"  {'j':>3} {'n':>2} {'|m|':>3}  {'名称':<30} {'系数(nm)':>12} {'系数(rad)':>12}")
    for i in range(min(15, len(res.coefficients))):
        j = int(res.jVec[i])
        n = int(res.nVec[i])
        m = int(res.mVec[i])
        nm = res.coefficients_nm[i]
        rad = res.coefficients[i]
        lines.append(f"  {j:>3} {n:>2} {m:>3}  {zernike_name(j):<28} {nm:>12.3f} {rad:>12.6f}")
    lines.append("")
    lines.append("说明:")
    lines.append(f"  * 相位差 = 参考元件 - 待测元件; 单位换算系数 λ/(因子·2π) = {res.scale_nm:.3f} nm/rad")
    lines.append("  * RMS/PV 统计区域为完整有效口径 (mask)，包含孔径边缘")
    lines.append("=" * 62)
    return "\n".join(lines)
