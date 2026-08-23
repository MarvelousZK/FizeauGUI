# -*- coding: utf-8 -*-
"""
设计系统主题（浅色 / 深色）
==========================
参考 Apple HIG 与 Microsoft Fluent 2 的通用设计原则：
留白克制、8pt 间距体系、圆角分层、清晰的状态反馈（hover/pressed/
focus/disabled）、统一的控件尺寸与排版层级。

设计 token（QSS 中以 @token@ 占位，make_qss 统一填充）：
    间距:  space_xs=4px  space_sm=8px  space_md=12px
           space_lg=16px space_xl=24px
    圆角:  radius_sm=6px radius_md=10px radius_lg=14px
    字号:  font_caption=11px font_body=13px font_subhead=14px font_title=17px
    控件:  输入类/普通按钮最小高度 36px，主按钮 38px（接近 HIG 触控建议）

品牌主色 token 为 primary 系列；若想换成 Apple 系统蓝，
把两套配色里的 primary 改成 #0A84FF 即可。

字体使用系统自带栈（Segoe UI Variable / Microsoft YaHei UI / PingFang SC），
项目不包含 Apple 专有的 SF Pro / SF Symbols。
"""

import re

# 与配色无关的尺寸/字号 token，浅色深色共用
DESIGN_SCALE = {
    # 间距
    "space_xs": "4px",
    "space_sm": "8px",
    "space_md": "12px",
    "space_lg": "16px",
    "space_xl": "24px",
    # 圆角
    "radius_sm": "6px",
    "radius_md": "10px",
    "radius_lg": "14px",
    # 字号
    "font_caption": "11px",
    "font_body": "13px",
    "font_subhead": "14px",
    "font_title": "17px",
}

_QSS = """
/* ===== 全局 ===== */
QWidget {
    font-family: "Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI",
                 "Microsoft YaHei", "PingFang SC", sans-serif;
    font-size: @font_body@;
    color: @text@;
    selection-background-color: @primary@;
    selection-color: #FFFFFF;
}
QMainWindow, QDialog {
    background: @bg@;
}
QToolTip {
    background: @card@;
    color: @text@;
    border: 1px solid @border@;
    border-radius: @radius_sm@;
    padding: @space_xs@ @space_sm@;
}

/* ===== 菜单 ===== */
QMenu {
    background: @card@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
    padding: @space_xs@;
}
QMenu::item {
    background: transparent;
    color: @text@;
    padding: @space_xs@ @space_lg@ @space_xs@ @space_md@;
    border-radius: @radius_sm@;
    min-width: 96px;
}
QMenu::item:selected {
    background: @primary_soft@;
    color: @primary@;
}
QMenu::separator {
    height: 1px;
    background: @border@;
    margin: @space_xs@ @space_sm@;
}

/* ===== 顶栏 ===== */
QFrame#headerBar {
    background: @card@;
    border-bottom: 1px solid @border@;
}
QLabel#titleLabel {
    font-size: @font_title@;
    font-weight: 700;
    color: @text@;
}
QLabel#subtitleLabel {
    font-size: @font_caption@;
    color: @muted@;
}

/* ===== 左侧三段式工作流 ===== */
QWidget#workflowShell {
    background: transparent;
}
QTabWidget#workflowTabs::pane {
    border: none;
    background: transparent;
}
QTabWidget#workflowTabs QTabBar {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 6px 2px 2px 2px;
}
QTabWidget#workflowTabs QTabBar::tab {
    min-width: 92px;
    padding: 9px 10px;
    margin: 0 3px 0 0;
    background: @card2@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
    color: @muted@;
}
QTabWidget#workflowTabs QTabBar::tab:selected {
    background: @primary@;
    color: #FFFFFF;
    border-color: @primary@;
}
QTabWidget#workflowTabs QTabBar::tab:hover:!selected {
    background: @primary_soft@;
    color: @primary@;
    border-color: @primary@;
}
QWidget#workflowShell QGroupBox {
    border-radius: @radius_md@;
    padding: @space_md@ @space_sm@ @space_sm@ @space_sm@;
}

/* ===== 卡片分组 ===== */
QGroupBox {
    background: @card@;
    border: 1px solid @border@;
    border-radius: @radius_lg@;
    /* 标题完全放在边框上方：不用底色遮线，浅/深色主题都保持干净。 */
    margin-top: 22px;
    padding: @space_lg@ @space_md@ @space_md@ @space_md@;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: @space_md@;
    top: -3px;
    padding: 0 @space_xs@;
    color: @primary@;
    background: transparent;
    font-weight: 600;
}

/* ===== 按钮 ===== */
QPushButton {
    background: @card@;
    border: 1px solid @border2@;
    border-radius: @radius_md@;
    padding: 0 @space_lg@;
    min-height: 36px;
    color: @text@;
}
QPushButton:hover {
    background: @primary_soft@;
    border-color: @primary@;
    color: @primary@;
}
QPushButton:pressed {
    background: @primary_soft@;
    border-color: @primary_pressed@;
    color: @primary_pressed@;
}
QPushButton:focus {
    border-color: @primary@;
}
QPushButton:disabled {
    background: @card2@;
    color: @disabled_text@;
    border-color: @border@;
}
QPushButton:default {
    border-color: @primary@;
}
QPushButton#primaryBtn {
    background: @primary@;
    color: #FFFFFF;
    border: 1px solid @primary@;
    font-weight: 600;
    min-height: 38px;
}
QPushButton#primaryBtn:hover {
    background: @primary_hover@;
    border-color: @primary_hover@;
    color: #FFFFFF;
}
QPushButton#primaryBtn:pressed {
    background: @primary_pressed@;
    border-color: @primary_pressed@;
    color: #FFFFFF;
}
QPushButton#primaryBtn:disabled {
    background: @primary_disabled@;
    border-color: @primary_disabled@;
    color: @on_primary_disabled@;
}
QPushButton#ghostBtn {
    background: transparent;
    border: 1px solid @border@;
    color: @muted@;
    min-height: 36px;
}
QPushButton#ghostBtn:hover {
    border-color: @primary@;
    color: @primary@;
    background: @primary_soft@;
}
QPushButton#ghostBtn:pressed {
    color: @primary_pressed@;
    border-color: @primary_pressed@;
}
QPushButton#disclosureBtn {
    background: transparent;
    border: none;
    border-top: 1px solid @border@;
    border-radius: 0;
    color: @muted@;
    min-height: 30px;
    padding: 4px 2px 0 2px;
    text-align: left;
}
QPushButton#disclosureBtn:hover,
QPushButton#disclosureBtn:checked {
    background: transparent;
    color: @primary@;
    border-top-color: @border@;
}
QFrame#subtlePanel {
    background: @card2@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: @radius_sm@;
    padding: @space_xs@;
    min-width: 28px;
    min-height: 28px;
    color: @muted@;
}
QToolButton:hover {
    background: @card2@;
    color: @text@;
}
QToolButton:checked {
    background: @primary_soft@;
    color: @primary@;
}
QToolButton:disabled {
    color: @disabled_text@;
}
QDialogButtonBox QPushButton, QMessageBox QPushButton {
    min-width: 72px;
}

/* ===== 输入控件 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: @card@;
    border: 1px solid @border2@;
    border-radius: @radius_md@;
    padding: 0 @space_sm@;
    min-height: 36px;
    selection-background-color: @primary@;
    selection-color: #FFFFFF;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border: 1px solid @control_hover@;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid @primary@;
    background: @card@;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: @disabled_text@;
    background: @card2@;
    border-color: @border@;
}
QLineEdit[readOnly="true"] {
    background: @card2@;
    color: @muted@;
    border-color: @border@;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid @muted@;
    margin-right: 10px;
}
QComboBox QAbstractItemView {
    background: @card@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
    padding: @space_xs@;
    outline: 0;
    selection-background-color: @primary_soft@;
    selection-color: @text@;
}
QComboBox QAbstractItemView::item {
    min-height: 30px;
    padding: @space_xs@ @space_sm@;
    border-radius: @radius_sm@;
    color: @text@;
}
QComboBox QAbstractItemView::item:hover {
    background: @card2@;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: transparent;
    border: none;
    width: 20px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: @card2@;
    border-radius: @radius_sm@;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid @muted@;
    margin-top: 3px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid @muted@;
    margin-bottom: 3px;
}

/* ===== 标签页（分段控件式） ===== */
QTabWidget::pane {
    border: none;
    background: transparent;
}
QTabBar {
    background: @tab_bg@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
    padding: @space_xs@;
}
QTabBar::tab {
    background: transparent;
    color: @muted@;
    padding: @space_sm@ @space_lg@;
    margin: 2px;
    border: 1px solid transparent;
    border-radius: @radius_sm@;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: @tab_hover_bg@;
    color: @text@;
}
QTabBar::tab:selected {
    background: @tab_selected_bg@;
    color: @primary@;
    border: 1px solid @border2@;
    font-weight: 700;
}
QTabBar::tab:disabled {
    color: @disabled_text@;
}

/* ===== 进度条 ===== */
QProgressBar {
    background: @progress_bg@;
    border: none;
    border-radius: @radius_sm@;
    min-height: 12px;
    max-height: 12px;
}
QProgressBar::chunk {
    background: @primary@;
    border-radius: @radius_sm@;
}

/* ===== 表格 ===== */
QTableWidget, QTableView {
    background: @card@;
    alternate-background-color: @card2@;
    border: 1px solid @border@;
    border-radius: @radius_md@;
    gridline-color: @border@;
}
QTableWidget::item, QTableView::item {
    padding: @space_xs@ @space_sm@;
    border: none;
}
QTableWidget::item:selected, QTableView::item:selected {
    background: @primary_soft@;
    color: @text@;
}
QHeaderView::section {
    background: @card2@;
    color: @muted@;
    border: none;
    border-bottom: 1px solid @border@;
    padding: @space_sm@;
    font-weight: 600;
}
QHeaderView::section:hover {
    background: @tab_hover_bg@;
    color: @text@;
}
QTableCornerButton::section {
    background: @card2@;
    border: none;
}

/* ===== 日志区（控制台样式） ===== */
QPlainTextEdit#logView {
    background: @log_bg@;
    color: @log_text@;
    border: 1px solid @log_border@;
    border-radius: @radius_md@;
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei", monospace;
    font-size: 12px;
    padding: @space_sm@;
    selection-background-color: @primary@;
    selection-color: #FFFFFF;
}
QFrame#logPanel {
    background: @card@;
    border-top: 1px solid @border@;
}
QPushButton#logToggleBtn {
    background: transparent;
    border: none;
    border-radius: @radius_sm@;
    color: @muted@;
    font-weight: 600;
    min-height: 28px;
    padding: 0 @space_sm@;
}
QPushButton#logToggleBtn:hover,
QPushButton#logToggleBtn:checked {
    background: @primary_soft@;
    color: @primary@;
}

/* ===== 滚动条 ===== */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: @scroll_handle@;
    border-radius: @radius_sm@;
    min-height: 32px;
}
QScrollBar::handle:vertical:hover {
    background: @scroll_handle_hover@;
}
QScrollBar::handle:vertical:pressed {
    background: @scroll_handle_hover@;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}
QScrollBar::handle:horizontal {
    background: @scroll_handle@;
    border-radius: @radius_sm@;
    min-width: 32px;
}
QScrollBar::handle:horizontal:hover {
    background: @scroll_handle_hover@;
}
QScrollBar::handle:horizontal:pressed {
    background: @scroll_handle_hover@;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}

/* ===== 分割条 ===== */
QSplitter::handle {
    background: transparent;
}
QSplitter::handle:horizontal {
    width: 6px;
}
QSplitter::handle:vertical {
    height: 6px;
}
QSplitter::handle:hover {
    background: @primary_soft@;
}

/* ===== 滚动区面板 ===== */
QScrollArea {
    border: none;
    background: transparent;
}
QScrollArea > QWidget {
    background: transparent;
}
QWidget#sidePanel {
    background: transparent;
}

/* ===== 复选/单选 ===== */
QCheckBox, QRadioButton {
    background: transparent;
    color: @text@;
    spacing: @space_sm@;
}
QCheckBox:hover, QRadioButton:hover {
    color: @primary@;
}
QCheckBox:disabled, QRadioButton:disabled {
    color: @disabled_text@;
}

/* ===== 滑块 ===== */
QSlider::groove:horizontal {
    height: 4px;
    background: @progress_bg@;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: @primary@;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: @card@;
    border: 2px solid @primary@;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    border-color: @primary_hover@;
}
QSlider::groove:vertical {
    width: 4px;
    background: @progress_bg@;
    border-radius: 2px;
}
QSlider::sub-page:vertical {
    background: @primary@;
    border-radius: 2px;
}
QSlider::handle:vertical {
    width: 14px;
    height: 14px;
    margin: 0 -5px;
    background: @card@;
    border: 2px solid @primary@;
    border-radius: 7px;
}
QSlider::handle:vertical:hover {
    border-color: @primary_hover@;
}

/* ===== 文字提示 / 状态 ===== */
QLabel#hintLabel {
    color: @muted@;
    font-size: @font_caption@;
}
QLabel#statLabel {
    font-weight: 700;
    color: @primary@;
    font-size: @font_subhead@;
}
QStatusBar {
    background: transparent;
    color: @muted@;
    font-size: @font_caption@;
}
QStatusBar::item {
    border: none;
}
QStatusBar QLabel {
    color: @muted@;
}
"""


def make_qss(palette: dict) -> str:
    """用 palette 填充 QSS 中的 @token@ 占位符，并校验无遗漏。"""
    s = _QSS
    for k, v in palette.items():
        s = s.replace("@" + k + "@", v)
    missing = sorted(set(re.findall(r"@(\w+)@", s)))
    if missing:
        raise ValueError(f"主题缺少 token: {', '.join(missing)}")
    return s


def _build(name: str, fig_face: str, fig_text: str, colors: dict) -> dict:
    return {
        "name": name,
        "fig_face": fig_face,
        "fig_text": fig_text,
        "qss": make_qss({**DESIGN_SCALE, **colors}),
    }


LIGHT = _build(
    name="light",
    fig_face="#FFFFFF",
    fig_text="#3A3F47",
    colors=dict(
        # 中性色
        bg="#F5F6FA",
        card="#FFFFFF",
        card2="#F2F4F8",
        border="#E5E8EF",
        border2="#D5DAE6",
        control_hover="#B9C1D1",
        text="#1B1E27",
        muted="#697182",
        disabled_text="#A9B0BD",
        # 品牌主色（换成 #0A84FF 即 Apple 系统蓝）
        primary="#4D6BFE",
        primary_hover="#3F5BF0",
        primary_pressed="#3550E3",
        primary_disabled="#B4BEFF",
        primary_soft="#EEF1FE",
        on_primary_disabled="#EDF0FF",
        # 语义色
        success="#2E9E4F",
        warning="#D97706",
        danger="#DC2626",
        # 组件色
        progress_bg="#E8ECF4",
        tab_bg="#ECEEF5",
        tab_hover_bg="#E3E7F0",
        tab_selected_bg="#FFFFFF",
        log_bg="#F8F9FB",
        log_text="#4A5261",
        log_border="#E5E8EF",
        scroll_handle="#CBD2DE",
        scroll_handle_hover="#ADB6C6",
    ),
)

DARK = _build(
    name="dark",
    fig_face="#1B1D23",
    fig_text="#C9CDD4",
    colors=dict(
        # 中性色
        bg="#14151A",
        card="#1E2026",
        card2="#282B33",
        border="#343842",
        border2="#3F4450",
        control_hover="#4E5462",
        text="#E9EBF0",
        muted="#9AA1AD",
        disabled_text="#585E69",
        # 品牌主色
        primary="#5D7BFF",
        primary_hover="#728DFF",
        primary_pressed="#4C67F0",
        primary_disabled="#313A63",
        primary_soft="#262E52",
        on_primary_disabled="#7C87C4",
        # 语义色
        success="#2EA65C",
        warning="#F0A020",
        danger="#E5484D",
        # 组件色
        progress_bg="#2A2D36",
        tab_bg="#22242C",
        tab_hover_bg="#2B2E38",
        tab_selected_bg="#30333E",
        log_bg="#0F1013",
        log_text="#C6CAD1",
        log_border="#2A2D35",
        scroll_handle="#3A3F4A",
        scroll_handle_hover="#4A505C",
    ),
)
