# -*- coding: utf-8 -*-
"""GUI 冒烟测试：离屏构建界面 → 载图 → 同步跑流水线 → 导出。"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from _paths import DATA_DIR

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
app = QApplication(sys.argv)

# 屏蔽模态对话框
QMessageBox.information = staticmethod(lambda *a, **k: print("[info]", a[2] if len(a) > 2 else ""))
QMessageBox.critical = staticmethod(lambda *a, **k: print("[critical]", a[2] if len(a) > 2 else ""))
QMessageBox.warning = staticmethod(lambda *a, **k: print("[warning]", a[2] if len(a) > 2 else ""))

DATA = str(DATA_DIR)
REF = os.path.join(DATA, "仿真_参考元件.bmp")
TEST = os.path.join(DATA, "仿真_待测元件.bmp")

from fizeau_gui.gui import (FTSidebandDialog, GITHUB_URL, MainWindow,
                            Surface3DDialog, WorkerThread)
from fizeau_gui.core import (build_mask, process_fizeau, s_T_shift,
                             takeda_ft_spectrum)
from fizeau_gui.theme import LIGHT

win = MainWindow()
win.show()
app.processEvents()
assert not win.windowIcon().isNull(), "程序图标未载入"
assert win.btn_github.toolTip() == GITHUB_URL
assert GITHUB_URL == "https://github.com/MarvelousZK/FizeauGUI"
assert win.workflow_tabs.count() == 3, "左侧未建立准备/处理/结果三段式工作流"
assert win.combo_phase.currentData() == "takeda", "Takeda FT 应为默认第一种算法"
assert win.combo_phase.findData("classic") == -1, "学生界面不应再显示经典相移"
assert [win.combo_phase.itemData(i) for i in range(win.combo_phase.count())] == \
    ["takeda", "masked", "adapt2", "wft2"]
assert "Luo" in win.combo_phase.itemText(win.combo_phase.findData("adapt2"))
assert win.result_group.objectName() == "resultGroup"
assert "QGroupBox#resultGroup::title" not in LIGHT["qss"], \
    "结果标题不应用背景色遮挡边框"
assert "margin-top: 22px" in LIGHT["qss"] and "top: -3px" in LIGHT["qss"], \
    "分组标题与边框的垂直间距未生效"
assert win.combo_ft_diag.count() == 5, "FT 诊断应只保留待测图五步主链"
assert win.combo_ft_diag.itemData(4)[1] == "confidence"
assert not win.log_view.isVisible(), "运行日志应默认折叠"
win.btn_log_toggle.setChecked(True)
assert win.log_view.isVisible(), "运行日志无法展开"
win.btn_log_toggle.setChecked(False)
print("1. 窗口构建 OK")

win._load_image(REF, "ref")
win._load_image(TEST, "test")
app.processEvents()
print(f"2. 载图 + 自动检测 OK  圆心=({win.spin_cx.value()},{win.spin_cy.value()}) 半径={win.spin_r.value()}")
assert win.spin_r.value() >= 5
assert win.spin_remove.value() == 11

params = dict(ref_path=REF, test_path=TEST, period=15,
              cx=win.spin_cx.value(), cy=win.spin_cy.value(), maskr=win.spin_r.value(),
              wavelength_nm=635.0, double_pass=2.0, max_term=36, n_remove=11)
w = WorkerThread(params)
w.progress.connect(win.on_progress)
w.done.connect(win.on_done)
w.error.connect(win.on_error)
win._run_revision = win._input_revision
w.run()  # 同步执行
app.processEvents()
assert win.result is not None, "处理未完成"
assert win.result.global_pv < 400.0, "仿真全口径仍存在异常边缘 PV 伪差"
print(f"3. 处理完成 OK  全局 RMS={win.result.global_rms:.2f} nm  PV={win.result.global_pv:.2f} nm")
print(f"   系数表行数: {win.table.rowCount()}  残差下拉项数: {win.combo_resid.count()}")

# Luo 算法必须在完整圆口径统计下抑制旧版的 ±2π 边缘枝错。
luo_result = process_fizeau(**params, phase_method="adapt2")
assert luo_result.global_pv < 400.0, \
    f"Luo adapt2 仍存在严重圆口径边缘误差: PV={luo_result.global_pv:.2f} nm"
print(f"3.0 Luo 圆口径边缘稳定 OK  RMS={luo_result.global_rms:.2f} nm  "
      f"PV={luo_result.global_pv:.2f} nm")

# 切换到经典 Takeda FT，验证专用参数、频谱诊断页和完整流水线。
ft_index = win.combo_phase.findData("takeda")
assert ft_index >= 0, "GUI 未提供 Takeda FT 算法"
win.workflow_tabs.setCurrentIndex(1)
win.combo_phase.setCurrentIndex(ft_index)
app.processEvents()
assert win.ft_options.isVisible(), "选择 Takeda 后 FT 参数面板未显示"
assert not win.ft_advanced_panel.isVisible(), "FT 高级设置应默认收起"
win.btn_ft_advanced.setChecked(True)
assert win.ft_advanced_panel.isVisible(), "FT 高级设置无法展开"
win.btn_ft_advanced.setChecked(False)
assert not win.check_ft_auto.isChecked(), "FT 教学模式应默认手动选边带"
assert win.btn_ft_manual_pick.isEnabled(), "手动频谱选择入口未启用"
assert (win.spin_ft_fx.value(), win.spin_ft_fy.value()) == (0, 0)
assert win.spin_ft_sigma.value() == 8.0
win.on_run()
assert win.worker is None, "未手动选边带时不应启动 FT 处理线程"

# 手动选择器必须显示与核心算法一致的 ROI 频谱，并把点击位置换算成 fx/fy。
mi = build_mask(win.test_img.shape, win.spin_cx.value(),
                win.spin_cy.value(), win.spin_r.value())
roi = win.test_img[mi.rowmin:mi.rowmax + 1, mi.colmin:mi.colmax + 1]
preview = takeda_ft_spectrum(roi, mask=mi.mask, apply_hann=True)
picker = FTSidebandDialog(preview.spectrum_log, sigma=8.0, parent=win)
assert picker.width() <= 800 and picker.height() <= 650, \
    f"FT 选峰弹窗仍然过大: {picker.width()}x{picker.height()}"
picker.selection = (picker.center_row, picker.center_col + 36)
picker._draw()
assert picker.carrier_cycles == (36, 0)
assert picker.filter_sigma == 8.0
assert len(picker.ax.collections) == 1, "所选谱峰应使用单个实心点标记"
assert len(picker.ax.patches) == 1, "Gaussian 1σ 应只绘制一个范围圆"
assert all(text.get_text() != "DC" for text in picker.ax.texts), \
    "频谱图不应再叠加 DC 文字标注"
assert picker.ax.lines[0].get_markersize() <= 9.0, "零频中心标记仍然过大"
assert picker.ax.lines[0].get_markeredgewidth() <= 1.2, "零频中心线仍然过粗"
assert float(picker.ax.collections[0].get_sizes()[0]) <= 20.0, \
    "所选谱峰中心点仍然过大"
assert picker.ax.patches[0].get_linewidth() <= 1.0, \
    "Gaussian 范围圆仍然过粗"
full_x_span = abs(np.subtract(*picker.ax.get_xlim()))
picker._on_scroll(type("ScrollEvt", (), {
    "inaxes": picker.ax, "button": "up",
    "xdata": picker.center_col + 36.0, "ydata": picker.center_row})())
zoomed_limits = (picker.ax.get_xlim(), picker.ax.get_ylim())
assert abs(np.subtract(*zoomed_limits[0])) < full_x_span, \
    "FT 频谱窗鼠标滚轮未放大"
picker.spin_sigma.setValue(9.0)  # 触发重画后仍应保留缩放视野
assert np.allclose(picker.ax.get_xlim(), zoomed_limits[0])
assert np.allclose(picker.ax.get_ylim(), zoomed_limits[1])
picker.close()
print("3.1 FT 默认手动选边带 + 滚轮缩放 + 轻量频谱标记 OK")
params_ft = dict(params, phase_method="takeda")
w_ft = WorkerThread(params_ft)
w_ft.progress.connect(win.on_progress)
w_ft.done.connect(win.on_done)
w_ft.error.connect(win.on_error)
win._run_revision = win._input_revision
w_ft.run()
app.processEvents()
assert win.result is not None and win.result.ft_diagnostics is not None
assert win.tabs.isTabEnabled(win.tabs.indexOf(win.ft_tab))
assert win.result.ft_diagnostics["test"].carrier_cycles == (36, 0)
for i in range(win.combo_ft_diag.count()):
    win.combo_ft_diag.setCurrentIndex(i)
    app.processEvents()
assert "cyc/img" in win.lbl_ft_meta.text()
print(f"3.2 Takeda FT + 频谱诊断 OK  RMS={win.result.global_rms:.2f} nm  "
      f"PV={win.result.global_pv:.2f} nm")

# 切换各显示页
for i in range(win.combo_resid.count()):
    win.combo_resid.setCurrentIndex(i)
    app.processEvents()
win.combo_trunc.setCurrentIndex(1)
win.combo_unwrap.setCurrentIndex(2)
app.processEvents()
print("4. 结果显示页切换 OK")

# 回归检查：colorbar 不应累积
for i in range(win.combo_resid.count()):
    win.combo_resid.setCurrentIndex(i)
    app.processEvents()
n_axes = len(win.canvas_resid.figure.axes)
ax_bounds = tuple(win.canvas_resid.ax.get_position().bounds)
print(f"4.1 残差画布切换 {win.combo_resid.count()} 次后子图数 = {n_axes} (应为 2)")
assert n_axes == 2, f"colorbar 累积 bug: 子图数 {n_axes} != 2"
for i in range(win.combo_resid.count() * 2):
    win.combo_resid.setCurrentIndex(i % win.combo_resid.count())
    app.processEvents()
ax_bounds_after = tuple(win.canvas_resid.ax.get_position().bounds)
assert np.allclose(ax_bounds_after, ax_bounds), \
    f"结果切换后主图布局漂移: {ax_bounds} -> {ax_bounds_after}"
ax_center = ax_bounds_after[0] + ax_bounds_after[2] / 2.0
assert abs(ax_center - 0.5) < 0.02, f"分析主图未居中: center={ax_center:.4f}"
win.canvas_resid.canvas.draw()
renderer = win.canvas_resid.canvas.get_renderer()
figure_right = win.canvas_resid.figure.bbox.x1
tick_right = max(label.get_window_extent(renderer).x1
                 for label in win.canvas_resid.cbar.ax.get_yticklabels()
                 if label.get_visible())
assert tick_right <= figure_right - 2, \
    f"colorbar 刻度仍被右边界裁切: {tick_right:.1f} > {figure_right:.1f}"

# 所有分析结果的有效掩膜外必须为 NaN，绘图时由透明 bad color 显示
mask = win.result.mask_info.mask
for arr in (win.result.truncated_test, win.result.truncated_ref,
            win.result.unwrapped_test, win.result.unwrapped_ref,
            win.result.phase):
    assert np.isnan(arr[~mask]).all(), "全口径 mask 外仍存在非 NaN 数据"
for resid in win.result.residuals:
    assert np.isnan(resid["W"][~mask]).all(), "残差 mask 外仍存在非 NaN 数据"
    assert np.isfinite(resid["W"][mask]).all(), "完整口径内存在未统计区域"
    expected_rms = float(np.std(resid["W"][mask]))
    expected_pv = float(np.ptp(resid["W"][mask]))
    assert np.isclose(resid["rms"], expected_rms), "RMS 未按完整口径统计"
    assert np.isclose(resid["pv"], expected_pv), "PV 未按完整口径统计"
print("4.1.1 分析图居中 + mask 外透明 OK")

# 主题切换回归
win.toggle_theme()
app.processEvents()
win.toggle_theme()
app.processEvents()
print("4.2 主题切换 (深色/浅色) OK")

# 3D 波面必须使用有界的交互网格，避免全分辨率面片拖慢旋转；
# 抽样只影响预览，传入的完整口径结果不能被修改。
surface_source = win.result.residuals[-1]["W"]
surface_source_copy = surface_source.copy()
surface_dlg = Surface3DDialog(surface_source, "3D 性能回归", parent=win)
surface_dlg.show()
app.processEvents()
assert max(surface_dlg._display_grid_shape) <= Surface3DDialog.MAX_GRID_SAMPLES + 2, \
    f"3D 显示网格过大: {surface_dlg._display_grid_shape}"
assert surface_dlg._display_face_count < 5000, \
    f"3D 面片数过多: {surface_dlg._display_face_count}"
assert np.array_equal(surface_source, surface_source_copy, equal_nan=True), \
    "3D 预览修改了完整口径结果"
surface_dlg._on_3d_press(type("Evt", (), {
    "inaxes": surface_dlg.ax, "button": 1})())
assert surface_dlg._fast_interaction and not surface_dlg.ax._axis3don, \
    "3D 拖动时未启用轻量重绘"
surface_dlg._on_3d_release(None)
assert not surface_dlg._fast_interaction and surface_dlg.ax._axis3don, \
    "3D 拖动结束后坐标轴未恢复"
app.processEvents()
surface_dlg.close()
print(f"4.3 3D 波面轻量网格 OK  {surface_dlg._display_grid_shape}, "
      f"面片数={surface_dlg._display_face_count}")

# 导出
outdir = tempfile.mkdtemp(prefix="fizeau_export_")
QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: outdir)
win.on_export()
resdir = None
for d in os.listdir(outdir):
    full = os.path.join(outdir, d)
    if os.path.isdir(full):
        resdir = full
        break
assert resdir, "未生成结果文件夹"
saved = sorted(os.listdir(resdir))
print(f"5. 导出 OK  共 {len(saved)} 个文件:")
for f in saved:
    print("   -", f)
assert any(f.endswith(".png") for f in saved) and any(f.endswith(".txt") for f in saved)

# 输入变更后旧结果必须立即失效，避免导出上一次的数据
win.spin_period.setValue(win.spin_period.value() + 1)
app.processEvents()
assert win.result is None and not win.btn_export.isEnabled(), "输入修改后旧结果仍可导出"
print("5.1 结果失效保护 OK")

# 条纹周期大于处理区域时必须明确报错，不能静默返回全零相位
try:
    s_T_shift(np.zeros((10, 10)), 15)
except ValueError:
    pass
else:
    raise AssertionError("超大条纹周期未被拒绝")
print("5.2 条纹周期边界检查 OK")

# 只先载入待测图时也应自动检测孔径
win2 = MainWindow()
win2._load_image(TEST, "test")
app.processEvents()
assert win2.ref_img is None and win2.spin_r.value() > 5, "待测图先载入时未自动检测孔径"
print("5.3 待测图优先载入自动检测 OK")

# 去项上限随拟合项数约束，并支持只保留全局结果
win2.spin_remove.setValue(11)
win2.spin_zterm.setValue(6)
assert win2.spin_remove.maximum() == 6 and win2.spin_remove.value() == 6
win2.spin_remove.setValue(0)
assert win2.spin_remove.value() == 0
win2.close()
print("5.4 可调去项分析 OK")

win.close()
print("\nSMOKE TEST PASSED")
sys.exit(0)
