# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


project_root = Path(SPECPATH).parent
source_root = project_root / "src"
entry_point = source_root / "fizeau_gui" / "__main__.py"
icon_file = source_root / "fizeau_gui" / "assets" / "app_icon.ico"

a = Analysis(
    [str(entry_point)],
    pathex=[str(source_root)],
    binaries=[],
    datas=[(str(icon_file), "assets")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick", "PySide6.QtQml", "PySide6.QtQuick",
        "PySide6.QtMultimedia", "PySide6.Qt3DCore", "tkinter",
    ],
    noarchive=False,
    optimize=0,
)

# Codex/Poppler may place its private ICU 78 DLLs on PATH. Qt 6 on Windows
# imports the unversioned system ICU API, so bundling those suffixed exports
# beside the executable makes QtCore fail with WinError 127 at startup.
_foreign_icu = {"icuuc.dll", "icudt78.dll"}
a.binaries = [
    item for item in a.binaries
    if Path(item[0]).name.lower() not in _foreign_icu
]
a.datas = [
    item for item in a.datas
    if Path(item[0]).name.lower() not in _foreign_icu
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="菲索干涉仪数据处理系统",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(icon_file)],
)
