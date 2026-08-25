# -*- coding: utf-8 -*-
"""
菲索干涉仪数据处理软件 — PySide6 图形界面
============================================

用法:  python -m fizeau_gui

流程: 载入两张干涉图 → 自动检测圆形孔径掩膜(可拖拽微调)
      → 设置参数 → 开始处理 → 查看结果 → 导出图片与报告
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib import rcParams, font_manager


def _setup_cjk_font():
    """让 matplotlib 使用 Windows 系统中的中文字体。"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun", "KaiTi"):
        for f in font_manager.fontManager.ttflist:
            if f.name == name:
                rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                rcParams["axes.unicode_minus"] = False
                return


_setup_cjk_font()

from matplotlib.backends.backend_agg import FigureCanvasAgg   # 离线保存图片用
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QComboBox, QTableWidget, QTableWidgetItem,
    QPlainTextEdit, QProgressBar, QTabWidget, QSplitter, QScrollArea,
    QFileDialog, QMessageBox, QHeaderView, QStatusBar, QFrame, QDialog,
    QSlider, QCheckBox, QSizePolicy,
)

from scipy.ndimage import map_coordinates

from .core import (
    read_image, auto_detect_circle, build_mask, process_fizeau,
    build_report, zernike_name, estimate_fringe_period, simulate_fizeau_pair,
    takeda_ft_spectrum, zStd,
)
from .theme import LIGHT, DARK

IMAGE_EXTS = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}
GITHUB_URL = "https://github.com/MarvelousZK/FizeauGUI"


def resource_path(name: str) -> str:
    """兼容源码环境与单文件运行环境的资源路径。"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "assets", name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", name)


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def transparent_nan_cmap(name: str):
    """返回独立 colormap，并让 NaN/掩膜区域完全透明。"""
    # 提高连续相位渐变的色阶分辨率，减轻 256 级 LUT 造成的色带。
    cmap = matplotlib.colormaps.get_cmap(name).resampled(2048).copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    return cmap


# ---------------------------------------------------------------- 掩膜圆拖拽

class CircleEditor:
    """让画布上的圆可拖拽：拖动圆心=移动，拖动边缘/滚轮=调整半径。"""

    def __init__(self, ax, canvas, circle_patch, on_change):
        self.ax = ax
        self.canvas = canvas
        self.circle = circle_patch
        self.on_change = on_change
        self._mode = None
        self._press = None
        self.cid_press = canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_motion = canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_scroll = canvas.mpl_connect("scroll_event", self._on_scroll)

    def disconnect(self):
        """断开所有画布事件连接（画布重绘前调用，防止重复响应）。"""
        for cid in (self.cid_press, self.cid_motion, self.cid_release, self.cid_scroll):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._mode = None
        self._press = None

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        cx, cy = self.circle.center
        r = self.circle.radius
        d = np.hypot(event.xdata - cx, event.ydata - cy)
        if d <= max(8.0, r * 0.30):
            self._mode = "move"
        elif abs(d - r) <= max(8.0, r * 0.12):
            self._mode = "resize"
        else:
            self._mode = None
        self._press = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._mode is None or self._press is None:
            return
        if event.inaxes is not self.ax:
            return
        if self._mode == "move":
            self.circle.center = (event.xdata, event.ydata)
        elif self._mode == "resize":
            cx, cy = self.circle.center
            self.circle.radius = np.hypot(event.xdata - cx, event.ydata - cy)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._mode is not None:
            self._mode = None
            self._press = None
            self._emit()

    def _on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        factor = 0.98 if event.button == "up" else 1.02
        self.circle.radius = max(3.0, self.circle.radius * factor)
        self.canvas.draw_idle()
        self._emit()

    def _emit(self):
        cx, cy = self.circle.center
        self.on_change(float(cx), float(cy), float(self.circle.radius))


# ---------------------------------------------------------------- 画布组件

class ImageCanvas(QWidget):
    """单个 matplotlib 画布：显示数组，可叠加可拖拽的圆。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        # 主图保持视觉居中；colorbar 紧邻主图，最右侧保留足够空间给
        # 负号和三位数刻度，避免在窄窗口中被画布边界裁掉。
        grid = self.figure.add_gridspec(
            1, 4, width_ratios=(0.12, 1.0, 0.025, 0.045),
            left=0.035, right=0.91, bottom=0.08, top=0.92, wspace=0.0)
        self.ax = self.figure.add_subplot(grid[0, 1])
        self._cbar_spec = grid[0, 3]
        self._cbar_ax = None
        self.circle_patch = None
        self.editor = None
        self.cbar = None
        self._face = "#FFFFFF"   # 主题色，由 set_theme 更新
        self._text = "#3A3F47"

    def set_theme(self, face: str, text: str):
        """切换浅色/深色主题时同步画布配色。"""
        self._face = face
        self._text = text
        self.figure.set_facecolor(face)
        self._apply_face()
        if self.cbar is not None:
            self.cbar.ax.tick_params(colors=text)
            self.cbar.ax.yaxis.label.set_color(text)
        self.canvas.draw_idle()

    def _apply_face(self):
        self.ax.set_facecolor(self._face)
        self.ax.title.set_color(self._text)
        self.ax.tick_params(colors=self._text)
        for sp in self.ax.spines.values():
            sp.set_color(self._text)

    def _clear_patch(self):
        if self.editor is not None:
            self.editor.disconnect()
            self.editor = None
        self.circle_patch = None
        # 必须通过 Colorbar.remove() 恢复主坐标轴原来的布局。
        # 直接 figure.delaxes(cbar.ax) 只会删掉色条，却会保留主图被挤压后的
        # 位置；反复切换结果时主图就会不断缩小、漂移。
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                # 极端情况下至少清掉残留色条轴；下一次重画仍可继续使用。
                if self.cbar.ax in self.figure.axes:
                    self.figure.delaxes(self.cbar.ax)
            self.cbar = None
            self._cbar_ax = None

    def show_message(self, text: str):
        self._clear_patch()
        self.ax.clear()
        self.ax.set_title(text)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self._apply_face()
        self.canvas.draw_idle()

    def show_array(self, arr, cmap="gray", title="", vmin=None, vmax=None):
        arr = np.ma.masked_invalid(np.asarray(arr, dtype=np.float64))
        H, W = arr.shape
        self._clear_patch()
        self.ax.clear()
        interpolation = "nearest" if cmap == "gray" else "bilinear"
        self.ax.imshow(arr, cmap=transparent_nan_cmap(cmap), vmin=vmin, vmax=vmax,
                       extent=(0, W, H, 0), interpolation=interpolation)
        self.ax.set_title(title)
        self._apply_face()
        self.canvas.draw_idle()

    def set_circle(self, cx, cy, r, on_change=None):
        """叠加圆。on_change(cx, cy, r) 回调时启用拖拽。"""
        if self.circle_patch is None:
            self.circle_patch = Circle((cx, cy), r, fill=False,
                                       edgecolor="#00ff00", linewidth=2)
            self.ax.add_patch(self.circle_patch)
        else:
            self.circle_patch.center = (cx, cy)
            self.circle_patch.radius = r
        if on_change is not None and self.editor is None:
            self.editor = CircleEditor(self.ax, self.canvas,
                                       self.circle_patch, on_change)
        self.canvas.draw_idle()

    def add_colorbar(self):
        if len(self.ax.images) > 0:
            im = self.ax.images[-1]
            try:
                self._cbar_ax = self.figure.add_subplot(self._cbar_spec)
                self.cbar = self.figure.colorbar(im, cax=self._cbar_ax)
                self.cbar.ax.tick_params(colors=self._text)
                self.cbar.ax.yaxis.label.set_color(self._text)
            except Exception:
                pass
            self.canvas.draw_idle()


# ---------------------------------------------------------------- 教学工具

class TwoPointPicker:
    """在 matplotlib 画布上依次点两个点的通用拾取器。"""

    def __init__(self, ax, canvas, on_done, color="#FF5252"):
        self.ax = ax
        self.canvas = canvas
        self.on_done = on_done
        self.color = color
        self.points: list = []
        self.artists: list = []
        self.cid = canvas.mpl_connect("button_press_event", self._on_press)

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if len(self.points) >= 2:
            return
        p = (float(event.xdata), float(event.ydata))
        self.points.append(p)
        self.artists.append(
            self.ax.plot([p[0]], [p[1]], "o", color=self.color,
                         markersize=9, markeredgecolor="white",
                         markeredgewidth=1.2)[0])
        if len(self.points) == 2:
            p1, p2 = self.points
            self.artists.append(
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--",
                             color=self.color, linewidth=1.4)[0])
            self.canvas.draw_idle()
            self.on_done(p1, p2)
        else:
            self.canvas.draw_idle()

    def clear(self):
        for art in self.artists:
            try:
                art.remove()
            except Exception:
                pass
        self.artists = []
        self.points = []
        self.canvas.draw_idle()

    def disconnect(self):
        try:
            self.canvas.mpl_disconnect(self.cid)
        except Exception:
            pass


class FTSidebandDialog(QDialog):
    """让学生在二维频谱上亲自选择一级边带和 Gaussian 窗宽。"""

    def __init__(self, spectrum_log, sigma=8.0, initial_carrier=None,
                 parent=None, face="#FFFFFF", text="#3A3F47"):
        super().__init__(parent)
        self.setWindowTitle("FT 手动选择一级边带")
        self.setMinimumSize(640, 500)
        parent_width = parent.width() if parent is not None else 1280
        parent_height = parent.height() if parent is not None else 800
        screen = parent.screen() if parent is not None else QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        screen_width = available.width() if available is not None else 1280
        screen_height = available.height() if available is not None else 720
        target_width = max(680, min(
            800, int(parent_width * 0.62), int(screen_width * 0.72)))
        target_height = max(520, min(
            650, int(parent_height * 0.72), int(screen_height * 0.78)))
        self.resize(target_width, target_height)

        self.spectrum_log = np.asarray(spectrum_log, dtype=np.float64)
        self.n_rows, self.n_cols = self.spectrum_log.shape
        self.center_row = self.n_rows // 2
        self.center_col = self.n_cols // 2
        self.selection = None
        if initial_carrier is not None and tuple(initial_carrier) != (0, 0):
            fx, fy = (int(initial_carrier[0]), int(initial_carrier[1]))
            row, col = self.center_row + fy, self.center_col + fx
            if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
                self.selection = (row, col)

        layout = QVBoxLayout(self)
        hint = QLabel(
            "先找到中心最亮的零频，再观察关于中心对称的两个一级谱峰。"
            "滚轮可围绕鼠标位置缩放；青色 + 是中心，黄色实心点是所选谱峰，"
            "黄色虚线圆是 Gaussian 1σ。")
        hint.setWordWrap(True)
        hint.setObjectName("hintLabel")
        layout.addWidget(hint)

        self.figure = Figure(figsize=(6.6, 4.8), dpi=100)
        self.figure.set_facecolor(face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)
        self._face = face
        self._text = text

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Gaussian σ:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, max(2.0, min(self.n_rows, self.n_cols) / 2.0))
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(max(1.0, float(sigma)))
        self.spin_sigma.setSuffix(" FFT px")
        self.spin_sigma.setToolTip("窗太宽会混入 DC，太窄会丢失物体相位带宽")
        self.spin_sigma.valueChanged.connect(self._draw)
        controls.addWidget(self.spin_sigma)
        controls.addSpacing(16)
        self.lbl_choice = QLabel("")
        self.lbl_choice.setObjectName("statLabel")
        controls.addWidget(self.lbl_choice, 1)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        btn_clear = QPushButton("清除重选")
        btn_clear.clicked.connect(self._clear_selection)
        btn_cancel = QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        self.btn_accept = QPushButton("确认并使用该边带")
        self.btn_accept.setObjectName("primaryBtn")
        self.btn_accept.clicked.connect(self.accept)
        self.btn_accept.setEnabled(self.selection is not None)
        buttons.addWidget(btn_clear)
        buttons.addStretch(1)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(self.btn_accept)
        layout.addLayout(buttons)

        self._view_limits = None
        self._cid = self.canvas.mpl_connect("button_press_event", self._on_click)
        self._scroll_cid = self.canvas.mpl_connect(
            "scroll_event", self._on_scroll)
        self._draw()

    @property
    def carrier_cycles(self):
        if self.selection is None:
            return None
        row, col = self.selection
        return int(col - self.center_col), int(row - self.center_row)

    @property
    def filter_sigma(self):
        return float(self.spin_sigma.value())

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        col = int(round(float(event.xdata)))
        row = int(round(float(event.ydata)))
        col = min(max(col, 0), self.n_cols - 1)
        row = min(max(row, 0), self.n_rows - 1)
        if np.hypot(row - self.center_row, col - self.center_col) < 2.0:
            QMessageBox.information(self, "请选择一级谱峰",
                                    "这里是零频中心，请点击与它分离的一级边带谱峰。")
            return
        self.selection = (row, col)
        self.btn_accept.setEnabled(True)
        self._draw()

    def _clear_selection(self):
        self.selection = None
        self.btn_accept.setEnabled(False)
        self._draw()

    @staticmethod
    def _clamp_zoom_limits(first, second, lower, upper, minimum_span=12.0):
        """把缩放范围限制在频谱内，同时保留坐标轴原来的正/反方向。"""
        reversed_axis = second < first
        low, high = sorted((float(first), float(second)))
        full_span = float(upper - lower)
        span = min(full_span, max(float(minimum_span), high - low))
        center = 0.5 * (low + high)
        low = center - 0.5 * span
        high = center + 0.5 * span
        if low < lower:
            high += lower - low
            low = lower
        if high > upper:
            low -= high - upper
            high = upper
        low = max(lower, low)
        high = min(upper, high)
        return (high, low) if reversed_axis else (low, high)

    def _on_scroll(self, event):
        """以鼠标所在频点为中心缩放；缩放状态在重画窗圈后仍然保留。"""
        if (event.inaxes is not self.ax or event.xdata is None
                or event.ydata is None):
            return
        if event.button == "up":
            scale = 0.80
        elif event.button == "down":
            scale = 1.25
        else:
            return

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        mouse_x, mouse_y = float(event.xdata), float(event.ydata)
        x_limits = (mouse_x + (x0 - mouse_x) * scale,
                    mouse_x + (x1 - mouse_x) * scale)
        y_limits = (mouse_y + (y0 - mouse_y) * scale,
                    mouse_y + (y1 - mouse_y) * scale)
        x_limits = self._clamp_zoom_limits(
            *x_limits, -0.5, self.n_cols - 0.5)
        y_limits = self._clamp_zoom_limits(
            *y_limits, -0.5, self.n_rows - 0.5)
        self.ax.set_xlim(*x_limits)
        self.ax.set_ylim(*y_limits)
        self._view_limits = (x_limits, y_limits)
        self.canvas.draw_idle()

    def _draw(self, *_):
        self.ax.clear()
        self.ax.imshow(
            self.spectrum_log, cmap=transparent_nan_cmap("magma"),
            extent=(-0.5, self.n_cols - 0.5, self.n_rows - 0.5, -0.5),
            interpolation="nearest")
        self.ax.plot(self.center_col, self.center_row, "+", color="#22D3EE",
                     markersize=9, markeredgewidth=1.2)
        if self.selection is not None:
            row, col = self.selection
            sigma = self.filter_sigma
            self.ax.scatter([col], [row], s=20, color="#FACC15",
                            edgecolors="#1B1E27", linewidths=0.5, zorder=6)
            self.ax.add_patch(Circle((col, row), sigma, fill=False,
                                     edgecolor="#FACC15", linewidth=1.0,
                                     linestyle=(0, (3, 3)), zorder=5))
            fx, fy = self.carrier_cycles
            distance = float(np.hypot(fx, fy))
            self.lbl_choice.setText(
                f"已选: fx={fx}, fy={fy} cyc/img    距中心={distance:.1f} px")
        else:
            self.lbl_choice.setText("尚未选择一级边带")
        self.ax.set_title("二维对数频谱：单击一个一级谱峰")
        self.ax.set_xlabel("FFT 列坐标")
        self.ax.set_ylabel("FFT 行坐标")
        self.ax.set_facecolor(self._face)
        self.ax.title.set_color(self._text)
        self.ax.xaxis.label.set_color(self._text)
        self.ax.yaxis.label.set_color(self._text)
        self.ax.tick_params(colors=self._text)
        for spine in self.ax.spines.values():
            spine.set_color(self._text)
        self.figure.tight_layout()
        if self._view_limits is not None:
            self.ax.set_xlim(*self._view_limits[0])
            self.ax.set_ylim(*self._view_limits[1])
        self.canvas.draw_idle()


def sample_profile(arr, p1, p2, num: int = 400):
    """沿两点连线对二维数组采样，返回 (距离, 值)。"""
    x0, y0 = p1
    x1, y1 = p2
    t = np.linspace(0.0, 1.0, num)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    vals = map_coordinates(np.asarray(arr, dtype=np.float64),
                           np.vstack([ys, xs]), order=1,
                           mode="constant", cval=np.nan)
    dist = np.hypot(xs - x0, ys - y0)
    return dist, vals


class ProfileDialog(QDialog):
    """剖面线查看器：左图显示连线位置，右图显示 1D 剖面。"""

    def __init__(self, arr, p1, p2, title, parent=None,
                 face="#FFFFFF", text="#3A3F47"):
        super().__init__(parent)
        self.setWindowTitle("剖面线")
        self.resize(980, 480)
        self.figure = Figure(figsize=(9, 4.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.canvas)

        self.figure.set_facecolor(face)
        gs = self.figure.add_gridspec(1, 2, width_ratios=(1.15, 1.0),
                                      left=0.06, right=0.97,
                                      bottom=0.10, top=0.90, wspace=0.18)
        ax_img = self.figure.add_subplot(gs[0, 0])
        ax_prf = self.figure.add_subplot(gs[0, 1])

        data = np.ma.masked_invalid(np.asarray(arr, dtype=np.float64))
        H, W = data.shape
        ax_img.imshow(data, cmap=transparent_nan_cmap("coolwarm"),
                      extent=(0, W, H, 0), interpolation="bilinear")
        ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], "--",
                    color="#FF5252", linewidth=2)
        ax_img.plot([p1[0], p2[0]], [p1[1], p2[1]], "o",
                    color="#FF5252", markersize=6)
        ax_img.set_title("剖面位置")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for sp in (ax_img, ax_prf):
            sp.set_facecolor(face)
            sp.title.set_color(text)
            sp.tick_params(colors=text)
            for s in sp.spines.values():
                s.set_color(text)

        dist, vals = sample_profile(data.filled(np.nan), p1, p2)
        ok = np.isfinite(vals)
        ax_prf.plot(dist, vals, color="#4D6BFE", linewidth=1.6)
        if ok.any():
            vmin, vmax = float(vals[ok].min()), float(vals[ok].max())
            rms = float(np.std(vals[ok]))
            pv = vmax - vmin
            ax_prf.scatter([dist[ok][int(np.argmin(vals[ok]))]],
                           [vmin], color="#2E9E4F", zorder=5)
            ax_prf.scatter([dist[ok][int(np.argmax(vals[ok]))]],
                           [vmax], color="#DC2626", zorder=5)
            ax_prf.set_title(f"{title}\nRMS={rms:.2f}  PV={pv:.2f} "
                             f"(绿=min, 红=max)")
        ax_prf.set_xlabel("沿剖面距离 (px)")
        ax_prf.set_ylabel("面形 (nm)")
        ax_prf.grid(alpha=0.25)
        self.canvas.draw_idle()


class Surface3DDialog(QDialog):
    """可旋转的 3D 波面查看器。"""

    # Matplotlib 的 mplot3d 在每次旋转时都要对曲面面片重新做深度排序。
    # 64×64 已足够显示当前最高 36 项 Zernike 面形，同时能把交互面片数
    # 从原先约 3 万降到约 4 千。这里只抽样显示，统计仍使用完整口径数据。
    MAX_GRID_SAMPLES = 64

    def __init__(self, arr, title, zlabel="nm", parent=None,
                 face="#FFFFFF", text="#3A3F47"):
        super().__init__(parent)
        self.setWindowTitle("3D 波面")
        self.resize(820, 740)
        self.figure = Figure(figsize=(6.6, 6.2), dpi=100)
        self.figure.set_facecolor(face)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111, projection="3d")
        data = np.asarray(arr, dtype=np.float64)
        finite_mask = np.isfinite(data)
        finite = data[finite_mask]
        fill = float(finite.min()) if finite.size else 0.0

        # 严格限制两个方向的采样点数。原来的 shape // 160 在 535 px
        # 图像上会得到 step=3，实际仍有 179×179 个点，并未达到上限。
        rows, cols = data.shape
        row_idx = np.linspace(
            0, rows - 1, min(rows, self.MAX_GRID_SAMPLES), dtype=np.intp)
        col_idx = np.linspace(
            0, cols - 1, min(cols, self.MAX_GRID_SAMPLES), dtype=np.intp)

        # 规则抽样可能恰好漏掉 PV 的两个端点；把完整口径的最小/最大点
        # 所在行列补进网格，让 3D 预览仍保留真实的全局高度范围。
        if finite.size:
            flat = data.ravel()
            finite_flat = np.flatnonzero(finite_mask.ravel())
            values = flat[finite_flat]
            extreme_flat = finite_flat[[int(np.argmin(values)),
                                        int(np.argmax(values))]]
            extreme_rows, extreme_cols = np.unravel_index(
                extreme_flat, data.shape)
            row_idx = np.unique(np.concatenate((row_idx, extreme_rows)))
            col_idx = np.unique(np.concatenate((col_idx, extreme_cols)))

        d = data[np.ix_(row_idx, col_idx)]
        # 仅对已经缩小的显示网格填充 NaN，避免复制整幅原始数组。
        d = np.nan_to_num(d, nan=fill, posinf=fill, neginf=fill)
        xs, ys = np.meshgrid(col_idx, row_idx)
        self._display_grid_shape = d.shape
        self._display_face_count = max(0, d.shape[0] - 1) * max(0, d.shape[1] - 1)

        surf = self.ax.plot_surface(xs, ys, d,
                                    cmap="coolwarm", linewidth=0,
                                    antialiased=False, rstride=1, cstride=1,
                                    vmin=float(finite.min()) if finite.size else None,
                                    vmax=float(finite.max()) if finite.size else None)
        self.figure.colorbar(surf, ax=self.ax, shrink=0.62, aspect=12,
                             pad=0.10, label=zlabel)
        self.ax.set_title(title, color=text)
        self.ax.set_xlabel("X (px)")
        self.ax.set_ylabel("Y (px)")
        self.ax.set_zlabel(zlabel)
        self.ax.view_init(elev=34, azim=-62)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.label.set_color(text)
            axis.set_pane_color((0.94, 0.94, 0.96, 0.3))
        self.ax.tick_params(colors=text)

        # 拖动时坐标轴文字、刻度和三块 pane 的重绘成本几乎与曲面相当。
        # 交互期间临时隐藏这些装饰，松开鼠标立即恢复；曲面和彩条不消失。
        self._fast_interaction = False
        self.canvas.mpl_connect("button_press_event", self._on_3d_press)
        self.canvas.mpl_connect("button_release_event", self._on_3d_release)
        self.canvas.draw_idle()

    def _on_3d_press(self, event):
        if event.inaxes is not self.ax or event.button not in (1, 2, 3):
            return
        self._fast_interaction = True
        self.ax.set_axis_off()

    def _on_3d_release(self, _event):
        if not self._fast_interaction:
            return
        self._fast_interaction = False
        self.ax.set_axis_on()
        self.canvas.draw_idle()


class SimulatorDialog(QDialog):
    """教学仿真面形生成器：设定像差 → 生成干涉图并载入软件。"""

    TERM_DEFS = [
        (4, "离焦 Defocus", 40.0),
        (5, "45°像散 Astig 45°", 30.0),
        (6, "0°像散 Astig 0°", 25.0),
        (7, "Y 彗差 Coma Y", 20.0),
        (8, "X 彗差 Coma X", 0.0),
        (9, "30°三叶 Trefoil 30°", 12.0),
        (10, "0°三叶 Trefoil 0°", 0.0),
        (11, "球差 Spherical", 15.0),
    ]

    def __init__(self, on_generated, parent=None):
        super().__init__(parent)
        self.on_generated = on_generated
        self.setWindowTitle("仿真面形生成器")
        self.resize(560, 620)

        v = QVBoxLayout(self)
        tip = QLabel("设定待测元件的 Zernike 像差（nm），生成后软件测量值\n"
                     "可在【真值对照】里和你设定的真值逐项对比。")
        tip.setObjectName("hintLabel")
        v.addWidget(tip)

        grid = QGridLayout()
        self.spin_size = QSpinBox()
        self.spin_size.setRange(360, 1600)
        self.spin_size.setValue(720)
        self.spin_size.setSuffix(" px")
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(100, 700)
        self.spin_radius.setValue(270)
        self.spin_radius.setSuffix(" px")
        self.spin_period = QSpinBox()
        self.spin_period.setRange(3, 200)
        self.spin_period.setValue(15)
        self.spin_period.setSuffix(" px")
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0.0, 20.0)
        self.spin_noise.setValue(1.5)
        self.spin_noise.setSingleStep(0.1)
        self.spin_noise.setSuffix(" 灰度")
        self.spin_wl = QDoubleSpinBox()
        self.spin_wl.setRange(300.0, 2000.0)
        self.spin_wl.setValue(635.0)
        self.spin_wl.setSuffix(" nm")
        grid.addWidget(QLabel("图像尺寸:"), 0, 0)
        grid.addWidget(self.spin_size, 0, 1)
        grid.addWidget(QLabel("孔径半径:"), 0, 2)
        grid.addWidget(self.spin_radius, 0, 3)
        grid.addWidget(QLabel("条纹周期:"), 1, 0)
        grid.addWidget(self.spin_period, 1, 1)
        grid.addWidget(QLabel("噪声:"), 1, 2)
        grid.addWidget(self.spin_noise, 1, 3)
        grid.addWidget(QLabel("波长:"), 2, 0)
        grid.addWidget(self.spin_wl, 2, 1)
        v.addLayout(grid)

        v.addWidget(QLabel("待测元件 Zernike 像差 (nm)："))
        self.ab_spins = {}
        aber_grid = QGridLayout()
        for idx, (j, name, default) in enumerate(self.TERM_DEFS):
            sp = QDoubleSpinBox()
            sp.setRange(-300.0, 300.0)
            sp.setDecimals(1)
            sp.setSingleStep(5.0)
            sp.setValue(default)
            sp.setSuffix(" nm")
            self.ab_spins[j] = sp
            r, c = divmod(idx, 2)
            aber_grid.addWidget(QLabel(f"Z{j} {name}:"), r, c * 2)
            aber_grid.addWidget(sp, r, c * 2 + 1)
        v.addLayout(aber_grid)

        hint = QLabel("参考元件固定为微小残差 (Z4=5, Z6=4, Z8=3 nm)。")
        hint.setObjectName("hintLabel")
        v.addWidget(hint)
        v.addStretch(1)

        row = QHBoxLayout()
        self.btn_make = QPushButton("生成并载入")
        self.btn_make.setObjectName("primaryBtn")
        self.btn_make.clicked.connect(self._make)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_make, 2)
        row.addWidget(btn_close, 1)
        v.addLayout(row)

    def _make(self):
        import tempfile
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(self, "生成失败", "缺少 Pillow 库，无法保存仿真图")
            return
        self.btn_make.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            test_ab = {j: sp.value() for j, sp in self.ab_spins.items()
                       if abs(sp.value()) > 1e-9}
            payload = simulate_fizeau_pair(
                size=self.spin_size.value(),
                radius=self.spin_radius.value(),
                period=self.spin_period.value(),
                wavelength_nm=self.spin_wl.value(),
                noise=self.spin_noise.value(),
                test_aberrations=test_ab or {11: 0.0},
            )
            outdir = os.path.join(tempfile.gettempdir(), "fizeau_sim_teaching")
            os.makedirs(outdir, exist_ok=True)
            ref_path = os.path.join(outdir, "仿真_教学_参考.bmp")
            test_path = os.path.join(outdir, "仿真_教学_待测.bmp")
            for path, key in ((ref_path, "I_ref"), (test_path, "I_test")):
                arr = np.uint8(np.clip(np.round(payload[key]), 0, 255))
                Image.fromarray(arr).save(path)
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_make.setEnabled(True)
        self.on_generated(payload, ref_path, test_path)
        self.accept()


class TruthCompareDialog(QDialog):
    """真值对照：学生设定的像差 vs 软件拟合结果。"""

    def __init__(self, truth, res, parent=None):
        super().__init__(parent)
        self.setWindowTitle("真值对照 — 设定面形 vs 软件测量")
        self.resize(760, 560)
        v = QVBoxLayout(self)

        mi = res.mask_info
        h_true = truth["h_diff"][mi.rowmin:mi.rowmax + 1,
                                 mi.colmin:mi.colmax + 1]
        valid = mi.mask
        W = res.residuals[0]["W"]
        diff = W[valid] - h_true[valid]
        piston = float(np.mean(diff))
        err_rms = float(np.std(diff))
        true_rms = float(np.std(h_true[valid]))
        true_pv = float(h_true[valid].max() - h_true[valid].min())
        meas_rms = res.global_rms
        meas_pv = res.global_pv

        stats = QLabel(
            f"<b>全局 (未去项)</b><br>"
            f"真值:  RMS = {true_rms:.2f} nm　　PV = {true_pv:.2f} nm<br>"
            f"软件:  RMS = {meas_rms:.2f} nm　　PV = {meas_pv:.2f} nm<br>"
            f"恢复误差: <b>{err_rms:.2f} nm RMS</b>"
            f"（已剔除无物理意义的全局平移 {piston:.1f} nm）")
        stats.setWordWrap(True)
        v.addWidget(stats)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["项", "名称", "设定真值 (nm)", "软件拟合 (nm)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        params = truth["params"]
        ref_ab = params["ref_aberrations"]
        test_ab = params["test_aberrations"]
        n = min(15, res.max_term)
        self.table.setRowCount(n)
        for i in range(n):
            j = int(res.jVec[i])
            truth_coef = ref_ab.get(j, 0.0) - test_ab.get(j, 0.0)
            self.table.setItem(i, 0, QTableWidgetItem(f"Z{j}"))
            self.table.setItem(i, 1, QTableWidgetItem(zernike_name(j)))
            self.table.setItem(
                i, 2, QTableWidgetItem(
                    f"{truth_coef:.1f}" if j in ref_ab or j in test_ab else "—"))
            self.table.setItem(
                i, 3, QTableWidgetItem(f"{res.coefficients_nm[i]:.2f}"))
        v.addWidget(self.table, 1)

        note = QLabel("说明：真值面形 = 参考 - 待测；软件测量的是恢复面形，"
                      "逐项系数因拟合/噪声会有小幅偏差，量级和正负应一致。")
        note.setObjectName("hintLabel")
        note.setWordWrap(True)
        v.addWidget(note)
        btn = QPushButton("关闭")
        btn.clicked.connect(self.accept)
        v.addWidget(btn, 0, Qt.AlignRight)


def render_save(arr, path, title, cmap="coolwarm", colorbar=True,
                vmin=None, vmax=None, dpi=150):
    """离屏渲染数组并保存为图片文件。"""
    fig = Figure(figsize=(7, 6), dpi=dpi)
    FigureCanvasAgg(fig)          # 绑定 Agg 画布，无需 GUI
    ax = fig.add_subplot(111)
    data = np.ma.masked_invalid(np.asarray(arr, dtype=np.float64))
    interpolation = "nearest" if cmap == "gray" else "bilinear"
    im = ax.imshow(data, cmap=transparent_nan_cmap(cmap),
                   vmin=vmin, vmax=vmax, interpolation=interpolation)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, bbox_inches="tight")


# ---------------------------------------------------------------- 后台线程

class WorkerThread(QThread):
    progress = Signal(int, str)     # 百分比, 消息
    done = Signal(object)           # ProcessResult
    error = Signal(str)

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            res = process_fizeau(
                **self.params,
                progress=lambda p, m: self.progress.emit(int(p), m),
            )
            self.done.emit(res)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


# ---------------------------------------------------------------- 主窗口

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("菲索干涉仪数据处理系统")
        self.setWindowIcon(QIcon(resource_path("app_icon.ico")))
        self.resize(1420, 880)
        self.setMinimumSize(1120, 700)

        self.ref_img: np.ndarray | None = None
        self.test_img: np.ndarray | None = None
        self.result = None
        self.worker: WorkerThread | None = None
        self.t_start = 0.0
        self._input_revision = 0
        self._run_revision = 0
        self.theme = LIGHT
        self.sim_truth = None             # 仿真生成器保存的真值
        self._sim_payload = None          # 仿真面板当前生成的数据
        self._sim_ref_path = None
        self._sim_test_path = None
        self._ruler = None                # 条纹周期两点拾取器
        self._profile_picker = None       # 剖面线两点拾取器
        self._profile_dlg = None
        self._surface_dlg = None
        self._truth_dlg = None
        self._resid_current = None        # 残差页当前显示的数组/标题

        self._build_ui()
        self.apply_theme()
        self.setAcceptDrops(True)
        self._update_mask_overlay()

    # ---------------- UI 构建 ----------------

    def _build_ui(self):
        central = QSplitter(Qt.Horizontal)
        self.setCentralWidget(central)

        # ===== 左侧三段式实验流程 =====
        # 每次只展示当前阶段，避免把输入、算法和结果全部堆在一条长滚动栏里。
        def workflow_page(*widgets):
            content = QWidget()
            content.setObjectName("sidePanel")
            layout = QVBoxLayout(content)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(8)
            for widget in widgets:
                layout.addWidget(widget)
            layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidget(content)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            return scroll

        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.setObjectName("workflowTabs")
        self.workflow_tabs.addTab(
            workflow_page(self._build_input_group(), self._build_mask_group()),
            "1  准备")
        self.workflow_tabs.addTab(
            workflow_page(self._build_param_group()),
            "2  处理")
        self.workflow_tabs.addTab(
            workflow_page(self._build_result_group()),
            "3  结果")
        workflow_shell = QWidget()
        workflow_shell.setObjectName("workflowShell")
        shell_layout = QVBoxLayout(workflow_shell)
        shell_layout.setContentsMargins(8, 0, 8, 8)
        shell_layout.setSpacing(6)
        shell_layout.addWidget(self.workflow_tabs, 1)
        shell_layout.addWidget(self._build_action_group(), 0)
        workflow_shell.setMinimumWidth(390)
        workflow_shell.setMaximumWidth(560)
        central.addWidget(workflow_shell)

        # ===== 右侧显示区 =====
        self.tabs = QTabWidget()
        self.canvas_img = ImageCanvas()
        self.canvas_ft = ImageCanvas()
        self.canvas_trunc = ImageCanvas()
        self.canvas_unwrap = ImageCanvas()
        self.canvas_resid = ImageCanvas()

        # Tab1: 干涉图与掩膜
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        bar1 = QHBoxLayout()
        bar1.addWidget(QLabel("显示:"))
        self.combo_img = QComboBox()
        self.combo_img.addItems(["参考元件干涉图", "待测元件干涉图"])
        self.combo_img.currentIndexChanged.connect(self.on_img_combo)
        bar1.addWidget(self.combo_img)
        bar1.addStretch(1)
        self.btn_fft = QPushButton("FFT 估周期")
        self.btn_fft.setToolTip("用傅里叶变换自动估计条纹周期并填入参数区")
        self.btn_fft.clicked.connect(self._auto_estimate_period)
        bar1.addWidget(self.btn_fft)
        self.btn_ruler = QPushButton("测条纹周期")
        self.btn_ruler.setCheckable(True)
        self.btn_ruler.setToolTip("在图上依次点两条相邻亮条纹，自动测出像素间距")
        self.btn_ruler.toggled.connect(self._toggle_ruler)
        bar1.addWidget(self.btn_ruler)
        self.lbl_mask_tip = QLabel("拖动绿色圆移动圆心；拖动圆周或滚动滚轮调整半径")
        self.lbl_mask_tip.setObjectName("hintLabel")
        l1.addLayout(bar1)
        l1.addWidget(self.lbl_mask_tip)
        l1.addWidget(self.canvas_img)

        # FT 教学诊断：原始频谱、边带窗、滤波频谱、复场质量
        self.ft_tab = QWidget()
        lft = QVBoxLayout(self.ft_tab)
        bar_ft = QHBoxLayout()
        bar_ft.addWidget(QLabel("显示:"))
        self.combo_ft_diag = QComboBox()
        diagnostic_items = [
            ("待测 · 原始对数频谱", ("test", "spectrum_log", "magma")),
            ("待测 · Gaussian 边带窗", ("test", "sideband_filter", "viridis")),
            ("待测 · 滤波后频谱", ("test", "filtered_spectrum_log", "magma")),
            ("待测 · 复场幅值", ("test", "amplitude", "viridis")),
            ("待测 · 相位置信度", ("test", "confidence", "viridis")),
        ]
        for label, data in diagnostic_items:
            self.combo_ft_diag.addItem(label, data)
        self.combo_ft_diag.currentIndexChanged.connect(self.on_ft_diag_combo)
        bar_ft.addWidget(self.combo_ft_diag)
        bar_ft.addStretch(1)
        self.lbl_ft_meta = QLabel("")
        self.lbl_ft_meta.setObjectName("statLabel")
        bar_ft.addWidget(self.lbl_ft_meta)
        lft.addLayout(bar_ft)
        self.canvas_ft.show_message("在相位算法中选择“经典 Fourier FT/Takeda”并开始处理")
        lft.addWidget(self.canvas_ft)

        # Tab2: 截断相位
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        bar2 = QHBoxLayout()
        bar2.addWidget(QLabel("显示:"))
        self.combo_trunc = QComboBox()
        self.combo_trunc.addItems(["待测元件", "参考元件"])
        self.combo_trunc.currentIndexChanged.connect(self.on_trunc_combo)
        bar2.addWidget(self.combo_trunc)
        bar2.addStretch(1)
        l2.addLayout(bar2)
        l2.addWidget(self.canvas_trunc)

        # Tab3: 展开相位
        t3 = QWidget()
        l3 = QVBoxLayout(t3)
        bar3 = QHBoxLayout()
        bar3.addWidget(QLabel("显示:"))
        self.combo_unwrap = QComboBox()
        self.combo_unwrap.addItems(["待测元件", "参考元件", "相位差(参考-待测)"])
        self.combo_unwrap.currentIndexChanged.connect(self.on_unwrap_combo)
        bar3.addWidget(self.combo_unwrap)
        bar3.addStretch(1)
        l3.addLayout(bar3)
        l3.addWidget(self.canvas_unwrap)

        # Tab4: 面形分解
        t4 = QWidget()
        l4 = QVBoxLayout(t4)
        resid_hint = QLabel("残差 = 实测面形 − 前 N 项 Zernike 拟合面形。"
                            "拖动滑块依次去掉平移/倾斜/离焦等低阶像差，"
                            "看剩下的、这些项解释不了的面形。")
        resid_hint.setObjectName("hintLabel")
        resid_hint.setWordWrap(True)
        l4.addWidget(resid_hint)
        bar4 = QHBoxLayout()
        bar4.addWidget(QLabel("显示:"))
        self.combo_resid = QComboBox()
        self.combo_resid.addItems(["全局 (未去项)"])
        self.combo_resid.currentIndexChanged.connect(self.on_resid_combo)
        bar4.addWidget(self.combo_resid)
        self.btn_3d = QPushButton("3D 波面")
        self.btn_3d.setEnabled(False)
        self.btn_3d.setToolTip("以可旋转 3D 曲面显示当前残差页的面形")
        self.btn_3d.clicked.connect(self._open_surface3d)
        bar4.addWidget(self.btn_3d)
        self.btn_profile = QPushButton("剖面线")
        self.btn_profile.setCheckable(True)
        self.btn_profile.setEnabled(False)
        self.btn_profile.setToolTip("在残差图上点两个点，查看沿该线的 1D 剖面")
        self.btn_profile.toggled.connect(self._toggle_profile)
        bar4.addWidget(self.btn_profile)
        bar4.addStretch(1)
        self.lbl_resid_stat = QLabel("")
        self.lbl_resid_stat.setObjectName("statLabel")
        bar4.addWidget(self.lbl_resid_stat)
        l4.addLayout(bar4)
        # 教学交互：去项分解滑块 + 单项像差视图
        row_k = QHBoxLayout()
        row_k.addWidget(QLabel("去项分解:"))
        self.slider_k = QSlider(Qt.Horizontal)
        self.slider_k.setRange(0, 0)
        self.slider_k.setValue(0)
        self.slider_k.valueChanged.connect(self._on_slider_k)
        row_k.addWidget(self.slider_k, 1)
        self.lbl_k = QLabel("去前 0 项")
        self.lbl_k.setObjectName("statLabel")
        row_k.addWidget(self.lbl_k)
        row_k.addSpacing(12)
        row_k.addWidget(QLabel("单项像差:"))
        self.combo_term = QComboBox()
        self.combo_term.addItem("去项视图", 0)
        self.combo_term.currentIndexChanged.connect(self._refresh_resid_view)
        row_k.addWidget(self.combo_term)
        l4.addLayout(row_k)
        l4.addWidget(self.canvas_resid)

        self.tabs.addTab(t1, "干涉图")
        self.tabs.addTab(self.ft_tab, "FT 诊断")
        self.tabs.addTab(t2, "截断相位")
        self.tabs.addTab(t3, "展开相位")
        self.tabs.addTab(t4, "面形")
        self.tabs.addTab(self._build_sim_tab(), "仿真")
        central.addWidget(self.tabs)
        central.setSizes([410, 1010])           # 左侧收紧，把空间优先留给图像
        central.setStretchFactor(0, 0)          # 左侧不随窗口拉伸
        central.setStretchFactor(1, 1)          # 右侧占满剩余空间
        central.setCollapsible(0, False)        # 左侧不允许折叠隐藏
        central.setHandleWidth(6)               # 加宽拖动把手，方便抓取

        # ===== 底部折叠日志 =====
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setMaximumHeight(104)
        self.log_view.setVisible(False)
        dock = QFrame()
        dock.setObjectName("logPanel")
        dl = QVBoxLayout(dock)
        dl.setContentsMargins(12, 4, 12, 6)
        dl.setSpacing(4)
        log_header = QHBoxLayout()
        self.btn_log_toggle = QPushButton("运行日志  ▸")
        self.btn_log_toggle.setObjectName("logToggleBtn")
        self.btn_log_toggle.setCheckable(True)
        self.btn_log_toggle.setToolTip("展开或收起运行日志")
        self.btn_log_toggle.toggled.connect(self._toggle_log_panel)
        log_header.addWidget(self.btn_log_toggle)
        log_header.addStretch(1)
        self.lbl_log_summary = QLabel("就绪 · 日志已收起")
        self.lbl_log_summary.setObjectName("hintLabel")
        log_header.addWidget(self.lbl_log_summary)
        dl.addLayout(log_header)
        dl.addWidget(self.log_view)

        status = QStatusBar()
        status.addWidget(QLabel(
            "流程：1 准备图像与孔径  →  2 设置算法并运行  →  3 查看与导出结果"))
        self.setStatusBar(status)

        vmain = QWidget()
        vml = QVBoxLayout(vmain)
        vml.setContentsMargins(0, 0, 0, 0)
        vml.setSpacing(0)
        vml.addWidget(self._build_header())
        vml.addWidget(central, 1)
        vml.addWidget(dock)
        self.setCentralWidget(vmain)

        self.canvas_trunc.show_message("请先开始处理")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.ft_tab), False)
        self.canvas_unwrap.show_message("请先开始处理")
        self.canvas_resid.show_message("请先开始处理")

    def _build_header(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("headerBar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 7, 16, 7)

        v = QVBoxLayout()
        v.setSpacing(1)
        title = QLabel("菲索干涉仪数据处理系统")
        title.setObjectName("titleLabel")
        sub = QLabel("Fizeau Interferometer Data Processing")
        sub.setObjectName("subtitleLabel")
        v.addWidget(title)
        v.addWidget(sub)
        lay.addLayout(v)
        lay.addStretch(1)

        self.btn_theme = QPushButton("🌙 深色模式")
        self.btn_theme.setObjectName("ghostBtn")
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.btn_github = QPushButton("GitHub · 拓展阅读 ↗")
        self.btn_github.setObjectName("ghostBtn")
        self.btn_github.setToolTip(GITHUB_URL)
        self.btn_github.clicked.connect(self.open_github)
        lay.addWidget(self.btn_github)
        lay.addWidget(self.btn_theme)
        return bar

    def apply_theme(self):
        """把当前主题应用到整个窗口与所有画布。"""
        QApplication.instance().setStyleSheet(self.theme["qss"])
        self.btn_theme.setText("☀️ 浅色模式" if self.theme is DARK else "🌙 深色模式")
        for canvas in (self.canvas_img, self.canvas_ft, self.canvas_trunc,
                       self.canvas_unwrap, self.canvas_resid,
                       self.canvas_sim_ref, self.canvas_sim_test):
            canvas.set_theme(self.theme["fig_face"], self.theme["fig_text"])

    def toggle_theme(self):
        self.theme = DARK if self.theme is LIGHT else LIGHT
        self.apply_theme()
        self.log(f"已切换到{'深色' if self.theme is DARK else '浅色'}模式")

    def _toggle_log_panel(self, expanded: bool):
        self.log_view.setVisible(expanded)
        self.btn_log_toggle.setText("运行日志  ▾" if expanded else "运行日志  ▸")
        if expanded:
            self.lbl_log_summary.setText("运行记录")
        elif self.log_view.blockCount() <= 1 and not self.log_view.toPlainText():
            self.lbl_log_summary.setText("就绪 · 日志已收起")
        else:
            last = self.log_view.toPlainText().splitlines()[-1]
            self.lbl_log_summary.setText(last[:64])

    def open_github(self):
        """打开作者仓库，供读者继续学习 WFT/Luo 等拓展内容。"""
        if not QDesktopServices.openUrl(QUrl(GITHUB_URL)):
            QMessageBox.warning(self, "无法打开链接", f"请在浏览器中访问:\n{GITHUB_URL}")

    def _build_input_group(self) -> QGroupBox:
        g = QGroupBox("输入干涉图")
        grid = QGridLayout(g)

        self.edit_ref = QLineEdit()
        self.edit_ref.setReadOnly(True)
        self.edit_ref.setPlaceholderText("参考元件干涉图 (20231031-162217.bmp)")
        btn_ref = QPushButton("浏览…")
        btn_ref.clicked.connect(self.on_browse_ref)

        self.edit_test = QLineEdit()
        self.edit_test.setReadOnly(True)
        self.edit_test.setPlaceholderText("待测元件干涉图 (20231031-162300.bmp)")
        btn_test = QPushButton("浏览…")
        btn_test.clicked.connect(self.on_browse_test)

        grid.addWidget(QLabel("参考:"), 0, 0)
        grid.addWidget(self.edit_ref, 0, 1)
        grid.addWidget(btn_ref, 0, 2)
        grid.addWidget(QLabel("待测:"), 1, 0)
        grid.addWidget(self.edit_test, 1, 1)
        grid.addWidget(btn_test, 1, 2)
        tip = QLabel("支持 .bmp/.png/.jpg/.tif，也可直接拖文件到窗口")
        tip.setObjectName("hintLabel")
        grid.addWidget(tip, 2, 0, 1, 3)
        return g

    def _build_mask_group(self) -> QGroupBox:
        g = QGroupBox("有效孔径")
        grid = QGridLayout(g)

        self.spin_cx = QSpinBox()
        self.spin_cx.setRange(0, 100000)
        self.spin_cx.setSuffix(" px")
        self.spin_cy = QSpinBox()
        self.spin_cy.setRange(0, 100000)
        self.spin_cy.setSuffix(" px")
        self.spin_r = QSpinBox()
        self.spin_r.setRange(5, 100000)
        self.spin_r.setSuffix(" px")

        grid.addWidget(QLabel("中心列 X:"), 0, 0)
        grid.addWidget(self.spin_cx, 0, 1)
        grid.addWidget(QLabel("中心行 Y:"), 1, 0)
        grid.addWidget(self.spin_cy, 1, 1)
        grid.addWidget(QLabel("半径 R:"), 2, 0)
        grid.addWidget(self.spin_r, 2, 1)

        btn_auto = QPushButton("自动检测")
        btn_auto.clicked.connect(self.on_auto_detect)
        btn_apply = QPushButton("应用掩膜")
        btn_apply.clicked.connect(self.on_apply_mask)
        actions = QHBoxLayout()
        actions.addWidget(btn_auto)
        actions.addWidget(btn_apply)
        grid.addLayout(actions, 3, 0, 1, 2)

        for sp in (self.spin_cx, self.spin_cy, self.spin_r):
            sp.valueChanged.connect(self.on_mask_spin_changed)
        return g

    def _build_param_group(self) -> QGroupBox:
        g = QGroupBox("相位提取与面形分析")
        grid = QGridLayout(g)

        self.spin_period = QSpinBox()
        self.spin_period.setRange(3, 1000)
        self.spin_period.setValue(15)
        self.spin_period.setSuffix(" 像素")

        self.spin_wl = QDoubleSpinBox()
        self.spin_wl.setRange(300.0, 2000.0)
        self.spin_wl.setValue(635.0)
        self.spin_wl.setDecimals(1)
        self.spin_wl.setSuffix(" nm")

        self.spin_pass = QDoubleSpinBox()
        self.spin_pass.setRange(0.5, 4.0)
        self.spin_pass.setValue(2.0)
        self.spin_pass.setDecimals(1)
        self.spin_pass.setToolTip("双程反射=2，单程=1")

        self.spin_zterm = QSpinBox()
        self.spin_zterm.setRange(1, 100)
        self.spin_zterm.setValue(80)
        self.spin_zterm.setToolTip("Zernike 拟合项数（默认 80，越大越慢）")

        self.spin_remove = QSpinBox()
        self.spin_remove.setRange(0, self.spin_zterm.value())
        self.spin_remove.setValue(11)
        self.spin_remove.setToolTip("依次生成去前 1 项至该项的分解残差图，不能超过拟合项数")
        self.spin_zterm.valueChanged.connect(self._on_zterm_changed)

        self.combo_phase = QComboBox()
        self.combo_phase.addItem("经典 Fourier FT/Takeda", "takeda")
        self.combo_phase.addItem("空间载波相移", "masked")
        self.combo_phase.addItem("Luo 单载频自适应相移 (adapt2)", "adapt2")
        self.combo_phase.addItem("掩膜自适应加窗傅里叶滤波 (Qian WFF)", "wft2")
        self.combo_phase.setToolTip("条纹相位提取算法")

        for control in (self.spin_period, self.spin_wl,
                        self.spin_pass, self.spin_zterm, self.spin_remove):
            control.valueChanged.connect(
                lambda _value: self._invalidate_result("处理参数已修改"))
        self.combo_phase.currentIndexChanged.connect(self._on_phase_method_changed)

        grid.addWidget(QLabel("相位算法:"), 0, 0)
        grid.addWidget(self.combo_phase, 0, 1)
        grid.addWidget(QLabel("条纹周期:"), 1, 0)
        grid.addWidget(self.spin_period, 1, 1)
        grid.addWidget(QLabel("光源波长:"), 2, 0)
        grid.addWidget(self.spin_wl, 2, 1)
        grid.addWidget(QLabel("双程因子:"), 3, 0)
        grid.addWidget(self.spin_pass, 3, 1)
        grid.addWidget(QLabel("拟合项数:"), 4, 0)
        grid.addWidget(self.spin_zterm, 4, 1)
        grid.addWidget(QLabel("分解项数:"), 5, 0)
        grid.addWidget(self.spin_remove, 5, 1)

        self.ft_options = QGroupBox("FT 手动解调")
        ft_grid = QGridLayout(self.ft_options)
        self.btn_ft_manual_pick = QPushButton("① 查看频谱并手动点选一级边带")
        self.btn_ft_manual_pick.setToolTip(
            "显示当前有效孔径 ROI 的二维频谱；点击一个一级谱峰并设置滤波窗宽")
        self.btn_ft_manual_pick.clicked.connect(self._open_ft_sideband_dialog)

        self.check_ft_auto = QCheckBox("自动检测一级边带（仅用于对照）")
        self.check_ft_auto.setChecked(False)
        self.check_ft_auto.setToolTip(
            "教学时建议保持关闭；勾选后软件将代替学生搜索一级谱峰")
        self.spin_ft_fx = QSpinBox()
        self.spin_ft_fx.setRange(-10000, 10000)
        self.spin_ft_fx.setValue(0)
        self.spin_ft_fx.setSuffix(" cyc/img")
        self.spin_ft_fy = QSpinBox()
        self.spin_ft_fy.setRange(-10000, 10000)
        self.spin_ft_fy.setValue(0)
        self.spin_ft_fy.setSuffix(" cyc/img")
        self.spin_ft_dc = QSpinBox()
        self.spin_ft_dc.setRange(0, 5000)
        self.spin_ft_dc.setSpecialValueText("自动")
        self.spin_ft_dc.setSuffix(" px")
        self.spin_ft_sigma = QDoubleSpinBox()
        self.spin_ft_sigma.setRange(1.0, 5000.0)
        self.spin_ft_sigma.setDecimals(2)
        self.spin_ft_sigma.setSingleStep(0.5)
        self.spin_ft_sigma.setValue(8.0)
        self.spin_ft_sigma.setSuffix(" px")
        self.spin_ft_sigma.setToolTip("手动设置一级边带 Gaussian 窗的 σ；过宽会混入 DC")
        self.check_ft_hann = QCheckBox("二维 Hann 窗")
        self.check_ft_hann.setChecked(True)
        self.combo_ft_sign = QComboBox()
        self.combo_ft_sign.addItem("+1", 1)
        self.combo_ft_sign.addItem("−1", -1)

        ft_hint = QLabel("看频谱  →  点一级谱峰  →  调整 σ  →  开始处理")
        ft_hint.setObjectName("hintLabel")
        ft_hint.setWordWrap(True)

        self.btn_ft_advanced = QPushButton("高级设置 · 自动对照 / 相位符号  ▸")
        self.btn_ft_advanced.setObjectName("disclosureBtn")
        self.btn_ft_advanced.setCheckable(True)
        self.btn_ft_advanced.toggled.connect(self._toggle_ft_advanced)
        self.ft_advanced_panel = QFrame()
        self.ft_advanced_panel.setObjectName("subtlePanel")
        advanced_grid = QGridLayout(self.ft_advanced_panel)
        advanced_grid.setContentsMargins(10, 8, 10, 8)
        advanced_grid.addWidget(QLabel("相位符号:"), 0, 0)
        advanced_grid.addWidget(self.combo_ft_sign, 0, 1)
        advanced_grid.addWidget(self.check_ft_auto, 1, 0, 1, 2)
        advanced_grid.addWidget(QLabel("自动模式 DC 排除:"), 2, 0)
        advanced_grid.addWidget(self.spin_ft_dc, 2, 1)
        self.ft_advanced_panel.setVisible(False)

        ft_grid.addWidget(self.btn_ft_manual_pick, 0, 0, 1, 2)
        ft_grid.addWidget(QLabel("已选 fx:"), 1, 0)
        ft_grid.addWidget(self.spin_ft_fx, 1, 1)
        ft_grid.addWidget(QLabel("已选 fy:"), 2, 0)
        ft_grid.addWidget(self.spin_ft_fy, 2, 1)
        ft_grid.addWidget(QLabel("② Gaussian σ:"), 3, 0)
        ft_grid.addWidget(self.spin_ft_sigma, 3, 1)
        ft_grid.addWidget(self.check_ft_hann, 4, 0, 1, 2)
        ft_grid.addWidget(ft_hint, 5, 0, 1, 2)
        ft_grid.addWidget(self.btn_ft_advanced, 6, 0, 1, 2)
        ft_grid.addWidget(self.ft_advanced_panel, 7, 0, 1, 2)
        grid.addWidget(self.ft_options, 6, 0, 1, 2)

        self.check_ft_auto.toggled.connect(self._on_ft_auto_toggled)
        for control in (self.spin_ft_fx, self.spin_ft_fy, self.spin_ft_dc,
                        self.spin_ft_sigma):
            control.valueChanged.connect(
                lambda _value: self._invalidate_result("FT 参数已修改"))
        self.check_ft_hann.toggled.connect(
            lambda _checked: self._invalidate_result("FT 参数已修改"))
        self.combo_ft_sign.currentIndexChanged.connect(
            lambda _index: self._invalidate_result("FT 参数已修改"))
        self._on_ft_auto_toggled(False)
        self.ft_options.setVisible(self.combo_phase.currentData() == "takeda")
        return g

    def _build_action_group(self) -> QGroupBox:
        g = QGroupBox("运行")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_run = QPushButton("▶ 开始处理")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_export = QPushButton("导出结果")
        self.btn_export.setMinimumHeight(34)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export)
        self.btn_help = QPushButton("帮助")
        self.btn_help.clicked.connect(self.on_help)
        row.addWidget(self.btn_run, 2)
        row.addWidget(self.btn_export, 2)
        row.addWidget(self.btn_help, 1)
        v.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        v.addWidget(self.progress)
        self.lbl_status = QLabel("就绪")
        v.addWidget(self.lbl_status)
        return g

    def _build_result_group(self) -> QGroupBox:
        g = QGroupBox("测量结果 · nm")
        g.setObjectName("resultGroup")
        self.result_group = g
        v = QVBoxLayout(g)
        self.lbl_global = QLabel("全局 RMS: —    PV: —")
        self.lbl_global.setObjectName("statLabel")
        v.addWidget(self.lbl_global)

        row_tools = QHBoxLayout()
        self.btn_truth = QPushButton("真值对照")
        self.btn_truth.setEnabled(False)
        self.btn_truth.setToolTip("对比仿真设定的像差与软件测量结果")
        self.btn_truth.clicked.connect(self._open_truth)
        row_tools.addWidget(self.btn_truth)
        row_tools.addStretch(1)
        v.addLayout(row_tools)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["项", "名称", "系数(nm)", "系数(rad)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(160)
        self.table.setMaximumHeight(330)        # 表内滚动，避免面板过长
        v.addWidget(self.table)
        return g

    # ---------------- 仿真生成面板 ----------------

    def _build_sim_tab(self) -> QWidget:
        page = QWidget()
        lay = QHBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8)

        # 左侧参数
        ctrl = QWidget()
        ctrl.setMinimumWidth(350)
        ctrl.setMaximumWidth(440)
        cv = QVBoxLayout(ctrl)
        tip = QLabel("设定待测元件像差与条纹参数，点【生成条纹图】后右侧实时预览；\n"
                     "确认后点【载入到处理流程】，再回右侧「干涉图」页检查孔径。")
        tip.setObjectName("hintLabel")
        tip.setWordWrap(True)
        cv.addWidget(tip)

        grid = QGridLayout()
        self.sim_spin_size = QSpinBox()
        self.sim_spin_size.setRange(360, 1600)
        self.sim_spin_size.setValue(720)
        self.sim_spin_size.setSuffix(" px")
        self.sim_spin_radius = QSpinBox()
        self.sim_spin_radius.setRange(100, 700)
        self.sim_spin_radius.setValue(270)
        self.sim_spin_radius.setSuffix(" px")
        self.sim_spin_period = QSpinBox()
        self.sim_spin_period.setRange(3, 200)
        self.sim_spin_period.setValue(15)
        self.sim_spin_period.setSuffix(" px")
        self.sim_spin_noise = QDoubleSpinBox()
        self.sim_spin_noise.setRange(0.0, 20.0)
        self.sim_spin_noise.setValue(1.5)
        self.sim_spin_noise.setSingleStep(0.1)
        self.sim_spin_noise.setSuffix(" 灰度")
        self.sim_spin_wl = QDoubleSpinBox()
        self.sim_spin_wl.setRange(300.0, 2000.0)
        self.sim_spin_wl.setValue(635.0)
        self.sim_spin_wl.setSuffix(" nm")
        grid.addWidget(QLabel("图像尺寸:"), 0, 0)
        grid.addWidget(self.sim_spin_size, 0, 1)
        grid.addWidget(QLabel("孔径半径:"), 0, 2)
        grid.addWidget(self.sim_spin_radius, 0, 3)
        grid.addWidget(QLabel("条纹周期:"), 1, 0)
        grid.addWidget(self.sim_spin_period, 1, 1)
        grid.addWidget(QLabel("噪声:"), 1, 2)
        grid.addWidget(self.sim_spin_noise, 1, 3)
        grid.addWidget(QLabel("波长:"), 2, 0)
        grid.addWidget(self.sim_spin_wl, 2, 1)
        cv.addLayout(grid)

        cv.addWidget(QLabel("待测元件 Zernike 像差 (nm)："))
        self.sim_ab_spins = {}
        aber_grid = QGridLayout()
        for idx, (j, name, default) in enumerate(SimulatorDialog.TERM_DEFS):
            sp = QDoubleSpinBox()
            sp.setRange(-300.0, 300.0)
            sp.setDecimals(1)
            sp.setSingleStep(5.0)
            sp.setValue(default)
            sp.setSuffix(" nm")
            self.sim_ab_spins[j] = sp
            r, c = divmod(idx, 2)
            aber_grid.addWidget(QLabel(f"Z{j} {name}:"), r, c * 2)
            aber_grid.addWidget(sp, r, c * 2 + 1)
        cv.addLayout(aber_grid)

        hint = QLabel("参考元件固定为微小残差 (Z4=5, Z6=4, Z8=3 nm)。")
        hint.setObjectName("hintLabel")
        cv.addWidget(hint)
        cv.addStretch(1)

        self.btn_sim_generate = QPushButton("生成条纹图")
        self.btn_sim_generate.setObjectName("primaryBtn")
        self.btn_sim_generate.clicked.connect(self._generate_sim_images)
        cv.addWidget(self.btn_sim_generate)
        self.btn_sim_load = QPushButton("载入到处理流程")
        self.btn_sim_load.setEnabled(False)
        self.btn_sim_load.setToolTip("把预览的这对仿真图载入主流程并记录真值")
        self.btn_sim_load.clicked.connect(self._load_sim_to_pipeline)
        cv.addWidget(self.btn_sim_load)

        # 右侧预览
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.canvas_sim_ref = ImageCanvas()
        self.canvas_sim_test = ImageCanvas()
        self.canvas_sim_ref.show_message("点【生成条纹图】后显示参考元件干涉图")
        self.canvas_sim_test.show_message("点【生成条纹图】后显示待测元件干涉图")
        rv.addWidget(self.canvas_sim_ref)
        rv.addWidget(self.canvas_sim_test)

        lay.addWidget(ctrl)
        lay.addWidget(right, 1)
        return page

    def _generate_sim_images(self):
        import tempfile
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.critical(self, "生成失败", "缺少 Pillow 库，无法保存仿真图")
            return
        self.btn_sim_generate.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            test_ab = {j: sp.value() for j, sp in self.sim_ab_spins.items()
                       if abs(sp.value()) > 1e-9}
            payload = simulate_fizeau_pair(
                size=self.sim_spin_size.value(),
                radius=self.sim_spin_radius.value(),
                period=self.sim_spin_period.value(),
                wavelength_nm=self.sim_spin_wl.value(),
                noise=self.sim_spin_noise.value(),
                test_aberrations=test_ab or {11: 0.0},
            )
            outdir = os.path.join(tempfile.gettempdir(), "fizeau_sim_teaching")
            os.makedirs(outdir, exist_ok=True)
            ref_path = os.path.join(outdir, "仿真_教学_参考.bmp")
            test_path = os.path.join(outdir, "仿真_教学_待测.bmp")
            for path, key in ((ref_path, "I_ref"), (test_path, "I_test")):
                arr = np.uint8(np.clip(np.round(payload[key]), 0, 255))
                Image.fromarray(arr).save(path)
        except Exception as e:
            QMessageBox.critical(self, "生成失败", str(e))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_sim_generate.setEnabled(True)

        self._sim_payload = payload
        self._sim_ref_path = ref_path
        self._sim_test_path = test_path
        self.canvas_sim_ref.show_array(
            payload["I_ref"], cmap="gray", title="仿真参考元件干涉图")
        self.canvas_sim_test.show_array(
            payload["I_test"], cmap="gray", title="仿真待测元件干涉图")
        self.btn_sim_load.setEnabled(True)
        p = payload["params"]
        terms = ", ".join(f"Z{j}={a:g}" for j, a in p["test_aberrations"].items())
        self.log(f"仿真条纹图已生成: {p['size']}px, 周期={p['period']}px, "
                 f"噪声={p['noise']} 灰度; 待测像差(nm): {terms}")
        self.log("预览满意后点【载入到处理流程】，再回右侧「干涉图」页检查孔径")

    # ---------------- 输入/掩膜 ----------------

    def log(self, msg: str):
        self.log_view.appendPlainText(msg)
        if not self.btn_log_toggle.isChecked():
            first_line = str(msg).splitlines()[-1]
            self.lbl_log_summary.setText(first_line[:64])

    def _invalidate_result(self, reason: str = "输入已修改"):
        """标记输入版本已变化，并清除可能被误导出的旧结果。"""
        self._input_revision += 1
        self.tabs.setTabEnabled(self.tabs.indexOf(self.ft_tab), False)
        self.canvas_ft.show_message("FT 参数或输入已修改，请重新处理")
        self.lbl_ft_meta.setText("")
        if self.result is None:
            return

        self.result = None
        self.btn_export.setEnabled(False)
        self.btn_3d.setEnabled(False)
        self.btn_profile.setChecked(False)
        self.btn_profile.setEnabled(False)
        self.btn_truth.setEnabled(False)
        self._resid_current = None
        if self._profile_picker is not None:
            self._profile_picker.clear()
            self._profile_picker.disconnect()
            self._profile_picker = None
        self.lbl_global.setText("全局 RMS: —    PV: —")
        self.lbl_resid_stat.setText("")
        self.table.setRowCount(0)

        self.combo_resid.blockSignals(True)
        self.combo_resid.clear()
        self.combo_resid.addItem("全局 (未去项)")
        self.combo_resid.blockSignals(False)
        self.slider_k.blockSignals(True)
        self.slider_k.setRange(0, 0)
        self.slider_k.setValue(0)
        self.slider_k.blockSignals(False)
        self.lbl_k.setText("去前 0 项")

        self.canvas_trunc.show_message("输入已修改，请重新处理")
        self.canvas_unwrap.show_message("输入已修改，请重新处理")
        self.canvas_resid.show_message("输入已修改，请重新处理")
        self.log(f"{reason}，旧结果已清除，请重新处理")

    def _on_zterm_changed(self, value: int):
        """去项分析不能超过实际参与拟合的 Zernike 项数。"""
        self.spin_remove.setMaximum(int(value))

    def _on_phase_method_changed(self, _index: int):
        is_ft = self.combo_phase.currentData() == "takeda"
        self.ft_options.setVisible(is_ft)
        self._invalidate_result("相位算法已修改")

    def _on_ft_auto_toggled(self, checked: bool):
        self.spin_ft_fx.setEnabled(not checked)
        self.spin_ft_fy.setEnabled(not checked)
        self.btn_ft_manual_pick.setEnabled(not checked)
        self.spin_ft_dc.setEnabled(checked)
        if hasattr(self, "btn_export"):
            self._invalidate_result("FT 边带模式已修改")

    def _toggle_ft_advanced(self, expanded: bool):
        self.ft_advanced_panel.setVisible(expanded)
        self.btn_ft_advanced.setText(
            "高级设置 · 自动对照 / 相位符号  ▾" if expanded
            else "高级设置 · 自动对照 / 相位符号  ▸")

    def _open_ft_sideband_dialog(self):
        """从当前 ROI 计算频谱，让学生点击选择一级边带。"""
        image = self.test_img if self.test_img is not None else self.ref_img
        source_name = "待测" if self.test_img is not None else "参考"
        if image is None:
            QMessageBox.information(self, "提示", "请先载入至少一张干涉图")
            return
        try:
            info = build_mask(image.shape, self.spin_cx.value(),
                              self.spin_cy.value(), self.spin_r.value())
            roi = image[info.rowmin:info.rowmax + 1,
                        info.colmin:info.colmax + 1]
        except Exception as exc:
            QMessageBox.warning(self, "无法生成 FT 频谱", str(exc))
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            preview = takeda_ft_spectrum(
                roi, mask=info.mask, apply_hann=self.check_ft_hann.isChecked())
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "无法生成 FT 频谱", str(exc))
            return
        QApplication.restoreOverrideCursor()

        current = (self.spin_ft_fx.value(), self.spin_ft_fy.value())
        dialog = FTSidebandDialog(
            preview.spectrum_log,
            sigma=self.spin_ft_sigma.value(),
            initial_carrier=current,
            parent=self,
            face=self.theme["fig_face"], text=self.theme["fig_text"])
        if dialog.exec() != QDialog.Accepted:
            return
        carrier = dialog.carrier_cycles
        if carrier is None:
            return
        for control, value in ((self.spin_ft_fx, carrier[0]),
                               (self.spin_ft_fy, carrier[1]),
                               (self.spin_ft_sigma, dialog.filter_sigma)):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._invalidate_result("FT 手动边带已选择")
        distance = float(np.hypot(*carrier))
        self.log(
            f"FT 手动选边带（{source_name}图 ROI）: fx={carrier[0]}, "
            f"fy={carrier[1]} cyc/img, 距离DC={distance:.1f}px, "
            f"Gaussian σ={dialog.filter_sigma:.1f}px")

    # ---------------- 教学工具 ----------------

    def _auto_estimate_period(self):
        img = self.ref_img if self.ref_img is not None else self.test_img
        if img is None:
            QMessageBox.information(self, "提示", "请先载入干涉图")
            return
        try:
            p = estimate_fringe_period(img)
        except Exception as e:
            QMessageBox.warning(self, "估计失败", str(e))
            return
        p = max(3, min(1000, int(round(p))))
        self.spin_period.setValue(p)
        self.log(f"FFT 自动估计条纹周期 ≈ {p} 像素，已填入参数区")

    def _toggle_ruler(self, checked: bool):
        if checked:
            if self._ruler is None:
                self._ruler = TwoPointPicker(
                    self.canvas_img.ax, self.canvas_img.canvas,
                    self._on_ruler_done)
            self.lbl_mask_tip.setText("在干涉图上依次点击两条相邻亮条纹")
            self.log("请依次点击两条相邻亮条纹（或暗条纹）完成测距")
        else:
            if self._ruler is not None:
                self._ruler.clear()
            self.lbl_mask_tip.setText(
                "拖动绿色圆移动圆心；拖动圆周或滚动滚轮调整半径")

    def _on_ruler_done(self, p1, p2):
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if d < 2.0:
            self.log("两点太近，请重新测量")
            return
        p = max(3, min(1000, int(round(d))))
        self.spin_period.setValue(p)
        self.log(f"两点间距 {d:.1f} px → 条纹周期已设为 {p} 像素")
        self.btn_ruler.setChecked(False)

    def _toggle_profile(self, checked: bool):
        if checked:
            if self.result is None:
                self.btn_profile.setChecked(False)
                return
            if self._profile_picker is not None:
                self._profile_picker.disconnect()
                self._profile_picker = None
            self._profile_picker = TwoPointPicker(
                self.canvas_resid.ax, self.canvas_resid.canvas,
                self._on_profile_done)
            self.log("请在残差图上点击两个点确定剖面线")
        else:
            if self._profile_picker is not None:
                self._profile_picker.clear()
                self._profile_picker.disconnect()
                self._profile_picker = None

    def _on_profile_done(self, p1, p2):
        if self._resid_current is None:
            return
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if d < 2.0:
            self.log("两点太近，请重新选择剖面")
            return
        arr, title = self._resid_current
        self._profile_dlg = ProfileDialog(
            arr, p1, p2, title, self,
            face=self.theme["fig_face"], text=self.theme["fig_text"])
        self._profile_dlg.show()
        self.btn_profile.setChecked(False)

    def _open_surface3d(self):
        if self._resid_current is None:
            QMessageBox.information(self, "提示", "请先完成处理")
            return
        arr, title = self._resid_current
        # 同一时间只保留一个 3D 画布，避免重复点击后多个 mplot3d
        # 窗口同时驻留并参与 Qt 重绘，造成越来越卡。
        if self._surface_dlg is not None:
            try:
                self._surface_dlg.close()
                self._surface_dlg.deleteLater()
            except RuntimeError:
                pass
            self._surface_dlg = None
        self._surface_dlg = Surface3DDialog(
            arr, title, "nm", self,
            face=self.theme["fig_face"], text=self.theme["fig_text"])
        self._surface_dlg.show()

    def _open_truth(self):
        if self.sim_truth is None or self.result is None:
            QMessageBox.information(
                self, "提示", "请先用【仿真面形生成器】生成数据并完成处理")
            return
        self._truth_dlg = TruthCompareDialog(self.sim_truth, self.result, self)
        self._truth_dlg.show()

    def _load_sim_to_pipeline(self):
        if self._sim_payload is None:
            QMessageBox.information(self, "提示", "请先生成条纹图")
            return
        self._on_sim_generated(self._sim_payload,
                               self._sim_ref_path, self._sim_test_path)
        self.tabs.setCurrentIndex(0)   # 回到干涉图页，方便确认掩膜

    def _on_sim_generated(self, payload, ref_path, test_path):
        self._invalidate_result("已载入新的仿真数据")
        self.sim_truth = payload
        try:
            self.ref_img = read_image(ref_path)
            self.test_img = read_image(test_path)
        except Exception as e:
            QMessageBox.critical(self, "载入失败", str(e))
            return
        self.edit_ref.setText(ref_path)
        self.edit_test.setText(test_path)
        p = payload["params"]
        terms = ", ".join(f"Z{j}={a:g}" for j, a in
                          p["test_aberrations"].items())
        self.log(f"仿真数据已载入: {p['size']}px, 周期={p['period']}px, "
                 f"噪声={p['noise']} 灰度")
        self.log(f"待测元件设定像差(nm): {terms}")
        self.log("设置好参数后点【开始处理】，完成后点【真值对照】对比答案")
        self.on_img_combo()
        self.on_auto_detect(silent_fail=True)

    # ---------------- Zernike 教学交互 ----------------

    def _on_slider_k(self, value: int):
        self.lbl_k.setText(f"去前 {value} 项")
        if self.result is None:
            return
        self.combo_resid.blockSignals(True)
        self.combo_resid.setCurrentIndex(value)
        self.combo_resid.blockSignals(False)
        self._refresh_resid_view()

    def _term_array(self, j: int):
        """把第 j 项 Zernike 单项像差展开为与残差图同尺寸的数组。"""
        res = self.result
        mi = res.mask_info
        index = np.nonzero(mi.mask.ravel())[0]
        rho = mi.R0.ravel()[index]
        theta = np.arctan2(mi.Y0.ravel()[index], mi.X0.ravel()[index])
        Zj = zStd(rho, theta, j)[0][:, -1]
        W = np.full(mi.mask.shape, np.nan, dtype=np.float64)
        Wf = W.ravel()
        Wf[index] = res.coefficients_nm[j - 1] * Zj
        return Wf.reshape(mi.mask.shape), res.coefficients_nm[j - 1]

    def _refresh_resid_view(self, *_):
        if self.result is None:
            return
        term = int(self.combo_term.currentData())
        if term == 0:
            idx = self.combo_resid.currentIndex()
            r = self.result.residuals[idx]
            k = r['k']
            title = (f"Full Wavefront (全局)" if k == 0
                     else f"Remove {k} terms (去前{k}项)")
            self.canvas_resid.show_array(r['W'], cmap="coolwarm", title=title)
            self.canvas_resid.add_colorbar()
            self.lbl_resid_stat.setText(
                f"RMS = {r['rms']:.3f} nm    PV = {r['pv']:.3f} nm")
            self._resid_current = (r['W'], title)
        else:
            W, coeff = self._term_array(term)
            name = zernike_name(term)
            title = f"Z{term} {name}  系数 = {coeff:.3f} nm"
            self.canvas_resid.show_array(W, cmap="coolwarm", title=title)
            self.canvas_resid.add_colorbar()
            self.lbl_resid_stat.setText(f"Z{term} 单项像差: {coeff:.3f} nm")
            self._resid_current = (W, title)

    def on_browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择参考元件干涉图", "", "图像文件 (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self._load_image(path, "ref")

    def on_browse_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择待测元件干涉图", "", "图像文件 (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if path:
            self._load_image(path, "test")

    def _load_image(self, path: str, which: str):
        try:
            img = read_image(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))
            return
        should_detect = (self.ref_img is None and self.test_img is None) or which == "ref"
        self._invalidate_result("输入图像已修改")
        if which == "ref":
            self.ref_img = img
            self.edit_ref.setText(path)
        else:
            self.test_img = img
            self.edit_test.setText(path)
        self.log(f"已载入 {'参考' if which == 'ref' else '待测'}元件干涉图: "
                 f"{os.path.basename(path)}  ({img.shape[0]}x{img.shape[1]})")
        self.on_img_combo()
        # 第一张图无论是参考还是待测都自动检测；参考图更新后再以参考图为准。
        if should_detect:
            self.on_auto_detect(silent_fail=True)

    def on_img_combo(self):
        img = self.ref_img if self.combo_img.currentIndex() == 0 else self.test_img
        if img is None:
            self.canvas_img.show_message("请先载入干涉图")
            return
        self.canvas_img.show_array(img, cmap="gray",
                                   title="参考元件干涉图" if self.combo_img.currentIndex() == 0
                                   else "待测元件干涉图")
        self._update_mask_overlay()

    def on_auto_detect(self, silent_fail: bool = False):
        img = self.ref_img if self.ref_img is not None else self.test_img
        if img is None:
            QMessageBox.information(self, "提示", "请先载入干涉图")
            return
        res = auto_detect_circle(img)
        if res is None:
            if not silent_fail:
                QMessageBox.warning(
                    self, "自动检测失败",
                    "未能自动识别圆形孔径。\n请在【1 准备】和右侧“干涉图”页手动调整。")
            self.log("自动检测孔径失败，请手动调整绿色圆")
            return
        cx, cy, r = res
        for sp, val in ((self.spin_cx, round(cx)), (self.spin_cy, round(cy)),
                        (self.spin_r, round(r))):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self._invalidate_result("孔径掩膜已修改")
        self.log(f"自动检测孔径: 中心({cx:.0f}, {cy:.0f})  半径 {r:.0f} 像素")
        self._update_mask_overlay()

    def on_apply_mask(self):
        img = self.ref_img if self.ref_img is not None else self.test_img
        if img is None:
            QMessageBox.information(self, "提示", "请先载入干涉图")
            return
        try:
            info = build_mask(img.shape, self.spin_cx.value(),
                              self.spin_cy.value(), self.spin_r.value())
        except ValueError as e:
            QMessageBox.warning(self, "掩膜无效", str(e))
            return
        self.log(f"掩膜已应用: 圆心({info.cx:.0f},{info.cy:.0f}) 半径 {info.maskr} 像素"
                 f"  ROI {info.rowmin}~{info.rowmax} 行, {info.colmin}~{info.colmax} 列")
        self._update_mask_overlay()

    def on_mask_spin_changed(self):
        self._update_mask_overlay()
        self._invalidate_result("孔径掩膜已修改")

    def _update_mask_overlay(self):
        if self.canvas_img.circle_patch is None and self.ref_img is None and self.test_img is None:
            return
        img = self.ref_img if self.combo_img.currentIndex() == 0 else self.test_img
        if img is None:
            return
        self.canvas_img.set_circle(self.spin_cx.value(), self.spin_cy.value(),
                                   self.spin_r.value(), on_change=self._on_circle_dragged)

    def _on_circle_dragged(self, cx, cy, r):
        for sp, val in ((self.spin_cx, round(cx)), (self.spin_cy, round(cy)),
                        (self.spin_r, round(r))):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self._invalidate_result("孔径掩膜已修改")

    # ---------------- 处理 ----------------

    def on_run(self):
        if self.ref_img is None or self.test_img is None:
            QMessageBox.information(self, "提示", "请先载入参考图和待测图各一张")
            return
        try:
            info = build_mask(self.ref_img.shape, self.spin_cx.value(),
                              self.spin_cy.value(), self.spin_r.value())
        except ValueError as e:
            QMessageBox.warning(self, "掩膜无效", str(e))
            return

        if (self.combo_phase.currentData() == "takeda" and
                not self.check_ft_auto.isChecked() and
                self.spin_ft_fx.value() == 0 and self.spin_ft_fy.value() == 0):
            QMessageBox.information(
                self, "请先手动选择一级边带",
                "FT 教学模式不会自动替你寻找谱峰。\n\n"
                "请点击【① 查看频谱并手动点选一级边带】，在二维频谱中"
                "选择一个与中心 DC 分离的一级谱峰，然后再开始处理。")
            return

        params = dict(
            ref_path=self.edit_ref.text(),
            test_path=self.edit_test.text(),
            period=self.spin_period.value(),
            phase_method=self.combo_phase.currentData(),
            cx=self.spin_cx.value(),
            cy=self.spin_cy.value(),
            maskr=info.maskr,
            wavelength_nm=self.spin_wl.value(),
            double_pass=self.spin_pass.value(),
            max_term=self.spin_zterm.value(),
            n_remove=self.spin_remove.value(),
            ft_carrier_cycles=(
                None if self.check_ft_auto.isChecked() else
                (self.spin_ft_fx.value(), self.spin_ft_fy.value())),
            ft_center_exclusion_radius=(
                None if self.spin_ft_dc.value() == 0 else self.spin_ft_dc.value()),
            ft_filter_sigma=(
                None if self.spin_ft_sigma.value() == 0 else self.spin_ft_sigma.value()),
            ft_apply_hann=self.check_ft_hann.isChecked(),
            ft_phase_sign=self.combo_ft_sign.currentData(),
        )
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setText("处理中…")
        self.t_start = time.time()
        self._run_revision = self._input_revision
        self.log("=" * 50)
        self.log(f"开始处理: 参考={os.path.basename(params['ref_path'])}  "
                 f"待测={os.path.basename(params['test_path'])}")
        self.log(f"参数: 周期={params['period']}px  波长={params['wavelength_nm']}nm  "
                 f"双程={params['double_pass']}  相位算法={params['phase_method']}  "
                 f"Zernike拟合={params['max_term']}  分解项数={params['n_remove']}")
        if params["phase_method"] == "takeda":
            carrier = ("自动" if params["ft_carrier_cycles"] is None else
                       str(params["ft_carrier_cycles"]))
            sigma = "自动" if params["ft_filter_sigma"] is None else params["ft_filter_sigma"]
            dc = ("自动" if params["ft_center_exclusion_radius"] is None else
                  params["ft_center_exclusion_radius"])
            self.log(f"FT 参数: 载频={carrier}, Gaussian σ={sigma}, "
                     f"DC排除={dc}, Hann={params['ft_apply_hann']}, "
                     f"符号={params['ft_phase_sign']:+d}")

        self.worker = WorkerThread(params, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        self.lbl_status.setText(msg)

    def on_done(self, res):
        if self._run_revision != self._input_revision:
            self.btn_run.setEnabled(True)
            self.btn_export.setEnabled(False)
            self.progress.setValue(0)
            self.lbl_status.setText("输入已变化，请重新处理")
            self.log("处理期间输入发生变化，本次结果已丢弃，请重新处理")
            QMessageBox.warning(
                self, "结果已失效", "处理期间输入或参数发生了变化，请重新点击【开始处理】。")
            return
        self.result = res
        dt = time.time() - self.t_start
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress.setValue(100)
        self.lbl_status.setText(f"完成 (耗时 {dt:.1f} 秒)")
        self.workflow_tabs.setCurrentIndex(2)

        # 结果概览
        self.lbl_global.setText(
            f"全局 RMS: {res.global_rms:.3f} nm     PV: {res.global_pv:.3f} nm")
        self._fill_table(res)

        # 残差下拉框 + 教学交互控件
        self.combo_resid.blockSignals(True)
        self.combo_resid.clear()
        self.combo_resid.addItems(["全局 (未去项)"] +
                                  [f"去前 {r['k']} 项" for r in res.residuals if r['k'] > 0])
        self.combo_resid.setCurrentIndex(0)
        self.combo_resid.blockSignals(False)

        self.slider_k.blockSignals(True)
        self.slider_k.setRange(0, len(res.residuals) - 1)
        self.slider_k.setValue(0)
        self.slider_k.blockSignals(False)
        self.lbl_k.setText("去前 0 项")

        self.combo_term.blockSignals(True)
        self.combo_term.clear()
        self.combo_term.addItem("去项视图", 0)
        for j in range(1, min(36, res.max_term) + 1):
            self.combo_term.addItem(f"单项 Z{j} {zernike_name(j)}", j)
        self.combo_term.setCurrentIndex(0)
        self.combo_term.blockSignals(False)

        self.btn_3d.setEnabled(True)
        self.btn_profile.setEnabled(True)
        self.btn_truth.setEnabled(self.sim_truth is not None)
        if self.sim_truth is not None:
            self.log("本次数据来自仿真生成器，点击【真值对照】对比设定值与测量值")

        has_ft = bool(res.phase_method == "takeda" and res.ft_diagnostics)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.ft_tab), has_ft)
        if has_ft:
            self.on_ft_diag_combo()
            test_ft = res.ft_diagnostics["test"]
            ref_ft = res.ft_diagnostics["reference"]
            self.log(
                f"FT 检测载频: 待测 {test_ft.carrier_cycles}, "
                f"参考 {ref_ft.carrier_cycles} cycles/image; "
                f"Gaussian σ={test_ft.filter_sigma:.2f}/{ref_ft.filter_sigma:.2f}")
            if test_ft.filter_too_wide or ref_ft.filter_too_wide:
                self.log("警告: FT 滤波窗相对载频过宽，可能混入 DC")
        else:
            self.canvas_ft.show_message("当前结果不是 Takeda FT；请选择 FT 算法后重新处理")
            self.lbl_ft_meta.setText("")

        self.on_trunc_combo()
        self.on_unwrap_combo()
        self.on_resid_combo()
        self.log(f"处理完成 (耗时 {dt:.1f} 秒)，点击【导出结果】保存图片与报告")

    def on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("处理失败")
        self.progress.setValue(0)
        self.log(f"处理出错: {msg}")
        self.btn_log_toggle.setChecked(True)
        QMessageBox.critical(self, "处理失败", msg)

    def _fill_table(self, res):
        n = min(15, len(res.coefficients))
        self.table.setRowCount(n)
        for i in range(n):
            j = int(res.jVec[i])
            self.table.setItem(i, 0, QTableWidgetItem(
                f"Z{j} (n={int(res.nVec[i])},|m|={int(res.mVec[i])})"))
            self.table.setItem(i, 1, QTableWidgetItem(zernike_name(j)))
            self.table.setItem(i, 2, QTableWidgetItem(f"{res.coefficients_nm[i]:.3f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{res.coefficients[i]:.6f}"))

    # ---------------- 结果显示 ----------------

    def on_ft_diag_combo(self, *_):
        if self.result is None or not self.result.ft_diagnostics:
            self.canvas_ft.show_message("请先使用经典 Takeda FT 完成处理")
            self.lbl_ft_meta.setText("")
            return
        data = self.combo_ft_diag.currentData()
        if not data:
            return
        source, field_name, cmap = data
        ft = self.result.ft_diagnostics[source]
        arr = getattr(ft, field_name)
        source_label = "待测" if source == "test" else "参考"
        field_labels = {
            "spectrum_log": "原始对数频谱",
            "sideband_filter": "Gaussian 一级边带窗",
            "filtered_spectrum_log": "滤波后频谱",
            "amplitude": "IFFT 复场幅值",
            "confidence": "相位置信度",
            "window": "二维 Hann 窗",
        }
        self.canvas_ft.show_array(
            arr, cmap=cmap, title=f"{source_label} · {field_labels[field_name]}")
        self.canvas_ft.add_colorbar()
        row, col = ft.sideband_center
        fx, fy = ft.carrier_cycles
        self.lbl_ft_meta.setText(
            f"边带(row,col)=({row + 1},{col + 1})   "
            f"f=({fx},{fy}) cyc/img   σ={ft.filter_sigma:.2f}px")

    def on_trunc_combo(self):
        if self.result is None:
            return
        if self.combo_trunc.currentIndex() == 0:
            arr = self.result.truncated_test
            title = "Truncated Phase - Test (待测元件截断相位)"
        else:
            arr = self.result.truncated_ref
            title = "Truncated Phase - Reference (参考元件截断相位)"
        self.canvas_trunc.show_array(arr, cmap="coolwarm", title=title)
        self.canvas_trunc.add_colorbar()

    def on_unwrap_combo(self):
        if self.result is None:
            return
        idx = self.combo_unwrap.currentIndex()
        if idx == 0:
            arr, title = self.result.unwrapped_test, "Unwrapped Phase - Test (待测展开相位)"
        elif idx == 1:
            arr, title = self.result.unwrapped_ref, "Unwrapped Phase - Reference (参考展开相位)"
        else:
            arr, title = self.result.phase, "Phase Difference (参考 - 待测)"
        self.canvas_unwrap.show_array(arr, cmap="coolwarm", title=title)
        self.canvas_unwrap.add_colorbar()

    def on_resid_combo(self):
        if self.result is None:
            return
        idx = self.combo_resid.currentIndex()
        self.slider_k.blockSignals(True)
        self.slider_k.setValue(idx)
        self.slider_k.blockSignals(False)
        self.lbl_k.setText(f"去前 {idx} 项")
        self._refresh_resid_view()

    # ---------------- 导出 ----------------

    def on_export(self):
        res = self.result
        if res is None:
            QMessageBox.information(self, "提示", "请先完成处理")
            return
        base = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not base:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(base, f"菲索干涉仪结果_{stamp}")
        os.makedirs(outdir, exist_ok=True)
        try:
            files = []

            def save(arr, name, title, cmap="coolwarm"):
                p = os.path.join(outdir, name)
                render_save(arr, p, title, cmap=cmap)
                files.append(p)

            save(res.truncated_test, "待测元件截断相位图.png",
                 "Truncated Phase - Test")
            save(res.truncated_ref, "参考元件截断相位图.png",
                 "Truncated Phase - Reference")
            save(res.unwrapped_test, "待测元件展开相位图.png",
                 "Unwrapped Phase - Test")
            save(res.unwrapped_ref, "参考元件展开相位图.png",
                 "Unwrapped Phase - Reference")
            save(res.phase, "Zernike多项式拟合相位分布图.png",
                 "Phase Difference (Zernike fitting input)")
            for r in res.residuals:
                if r['k'] == 0:
                    continue
                save(r['W'], f"Zernike多项式去前{r['k']}项后相位分布图.png",
                     f"Remove {r['k']} terms, RMS = {r['rms']:.3f} nm, PV = {r['pv']:.3f} nm")

            # Takeda FT 教学诊断：频谱、滤波窗、复场幅值与置信度
            if res.ft_diagnostics:
                for source, source_cn in (("test", "待测"), ("reference", "参考")):
                    ft = res.ft_diagnostics[source]
                    save(ft.spectrum_log, f"FT_{source_cn}_原始对数频谱.png",
                         f"Takeda FT Spectrum - {source_cn}", cmap="magma")
                    save(ft.sideband_filter, f"FT_{source_cn}_Gaussian边带窗.png",
                         f"Takeda FT Sideband Filter - {source_cn}", cmap="viridis")
                    save(ft.filtered_spectrum_log, f"FT_{source_cn}_滤波后频谱.png",
                         f"Takeda FT Filtered Spectrum - {source_cn}", cmap="magma")
                    save(ft.amplitude, f"FT_{source_cn}_复场幅值.png",
                         f"Takeda FT Complex Amplitude - {source_cn}", cmap="viridis")
                    save(ft.confidence, f"FT_{source_cn}_相位置信度.png",
                         f"Takeda FT Phase Confidence - {source_cn}", cmap="viridis")
                    save(ft.window, f"FT_{source_cn}_Hann窗.png",
                         f"Takeda FT Hann Window - {source_cn}", cmap="viridis")

            # 掩膜叠加图
            img = self.ref_img
            fig = Figure(figsize=(7, 6), dpi=150)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            H, W = img.shape
            ax.imshow(img, cmap="gray", extent=(0, W, H, 0))
            ax.add_patch(Circle((res.mask_info.cx, res.mask_info.cy),
                                res.mask_info.maskr, fill=False,
                                edgecolor="#00ff00", linewidth=2))
            ax.set_title("Interferogram and Mask")
            p = os.path.join(outdir, "干涉图与掩膜.png")
            fig.savefig(p, bbox_inches="tight")
            files.append(p)

            # 报告
            p = os.path.join(outdir, "数据处理报告.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(build_report(res))
            files.append(p)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(e))
            return

        self.log(f"已导出 {len(files)} 个文件到:\n{outdir}")
        QMessageBox.information(self, "导出完成",
                                f"已导出 {len(files)} 个文件:\n{outdir}")

    # ---------------- 帮助 ----------------

    def on_help(self):
        QMessageBox.information(
            self, "使用说明",
            "① 左侧选择【1 准备】，载入参考元件和待测元件干涉图；\n"
            "② 检查自动检测的绿色孔径圆；拖动圆心、圆周或滚轮可微调，"
            "也可在“有效孔径”中输入数值；\n"
            "③ 切换到【2 处理】并检查参数：条纹周期是相邻两条纹的像素间距；\n"
            "④ FT 教学：选择【经典 Fourier FT/Takeda】，点击"
            "【查看频谱并手动点选一级边带】；先辨认中心 DC 和成对一级谱峰，"
            "再亲自点击其中一个谱峰，并调整 Gaussian σ；\n"
            "⑤ 点击左栏底部固定的【开始处理】，在【FT 诊断】页检查所选边带、"
            "滤波窗、复场幅值和置信度；\n"
            "⑥ 完成后左栏自动进入【3 结果】，右侧可查看截断相位、展开相位和面形；\n"
            "⑦ 自动边带仅用于课后对照；【导出结果】保存全部图片和报告。"
            "底部【运行日志】按需展开，拓展算法见顶部 GitHub。\n\n"
            "结果含义: 残差图 RMS/PV 越小代表面形越接近对应 Zernike 前 N 项拟合。")

    # ---------------- 拖放支持 ----------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if is_image_file(url.toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not is_image_file(path):
                continue
            if not self.edit_ref.text():
                self._load_image(path, "ref")
            else:
                self._load_image(path, "test")
        event.acceptProposedAction()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("菲索干涉仪数据处理系统")
    app.setWindowIcon(QIcon(resource_path("app_icon.ico")))
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(LIGHT["qss"])   # 启动即用浅色主题，避免闪白
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
