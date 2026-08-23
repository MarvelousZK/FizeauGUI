# -*- coding: utf-8 -*-
"""验证 qg_dunwrap 在合成无残差数据上的正确性。"""
import numpy as np
import _paths  # 将项目 src 目录加入可执行测试脚本的模块搜索路径
from fizeau_gui.core import (qg_dunwrap,
                             phase_derivative_variance)

# ---- 1. 合成数据正确性（无残差，应精确还原）----
rng = np.random.default_rng(42)
N = 300
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
R = np.hypot(X, Y)
mask = R <= 1.0
true = (3.0 * X + 2.0 * Y + 1.5 * (2 * R**2 - 1) + 0.6 * np.sin(6 * np.arctan2(Y, X))
        + 0.3 * R**2 * np.cos(4 * np.arctan2(Y, X)))
true = true * mask
wrapped = np.angle(np.exp(1j * (true + 0.0 * rng.standard_normal((N, N)) * mask)))

u_qg = qg_dunwrap(wrapped, mask, quality=phase_derivative_variance(wrapped, 3))

# 对齐全局 2π 常数后与真值比较
def align(a):
    a = a.copy()
    a[mask] -= np.median(a[mask])
    return a
err_qg = np.abs(np.angle(np.exp(1j * (align(u_qg)[mask] - align(true)[mask])))).max()
print(f"合成数据（无残差）质量引导展开误差: {err_qg:.3e} rad")
assert err_qg < 1e-6, "合成数据未精确还原!"
