"""Generate reproducible screenshots of the main result views.

The script uses the deterministic interferograms in ``examples/simulated_data``
and runs the same processing code as the desktop application.  It does not
depend on a previously exported result folder.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# On Windows the native Qt platform is required for system Chinese fonts.
# A CI runner may still opt into offscreen mode by setting QT_QPA_PLATFORM
# before launching this script.
os.environ.setdefault("QT_SCALE_FACTOR", "1")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fizeau_gui.core import process_fizeau
from fizeau_gui.gui import MainWindow, ProfileDialog, Surface3DDialog


OUTPUT_DIR = ROOT / "docs" / "screenshots"
REFERENCE_IMAGE = ROOT / "examples" / "simulated_data" / "仿真_参考元件.bmp"
TEST_IMAGE = ROOT / "examples" / "simulated_data" / "仿真_待测元件.bmp"


def _render(widget, output_name: str, app: QApplication) -> None:
    """Finish pending Qt/Matplotlib paints and save a widget screenshot."""
    widget.show()
    for _ in range(4):
        app.processEvents()
    pixmap = widget.grab()
    output_path = OUTPUT_DIR / output_name
    if not pixmap.save(str(output_path), "PNG"):
        raise RuntimeError(f"无法保存截图: {output_path}")
    print(f"saved {output_path.relative_to(ROOT)}  {pixmap.width()}x{pixmap.height()}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_IMAGE.exists() or not TEST_IMAGE.exists():
        raise FileNotFoundError("缺少 examples/simulated_data 中的仿真干涉图")

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication.instance() or QApplication(sys.argv)

    window = MainWindow()
    window.resize(1500, 920)
    window._load_image(str(REFERENCE_IMAGE), "ref")
    window._load_image(str(TEST_IMAGE), "test")

    # The simulated-data description fixes these values.  Explicit assignment
    # keeps the documentation screenshots independent of circle auto-detection.
    window.spin_cx.setValue(360)
    window.spin_cy.setValue(360)
    window.spin_r.setValue(270)

    processing_started = time.time()
    result = process_fizeau(
        ref_path=str(REFERENCE_IMAGE),
        test_path=str(TEST_IMAGE),
        period=15,
        cx=360,
        cy=360,
        maskr=270,
        wavelength_nm=635.0,
        double_pass=2.0,
        max_term=36,
        n_remove=11,
        phase_method="wft2",
    )

    window._run_revision = window._input_revision
    window.t_start = processing_started
    window.on_done(result)
    window.workflow_tabs.setCurrentIndex(2)

    # Main-window result pages.
    window.tabs.setCurrentIndex(2)
    window.combo_trunc.setCurrentIndex(0)
    window.on_trunc_combo()
    _render(window, "01_truncated_phase.png", app)

    window.tabs.setCurrentIndex(4)
    # Removing piston, tilt and defocus gives a more instructive view of the
    # remaining astigmatism/coma than the piston-dominated global map.
    window.combo_resid.setCurrentIndex(4)
    window.on_resid_combo()
    _render(window, "02_surface_analysis.png", app)

    # Stand-alone interactive viewers use the exact same array as the current
    # surface-analysis page.
    surface, title = window._resid_current
    face = window.theme["fig_face"]
    text = window.theme["fig_text"]

    dialog_3d = Surface3DDialog(
        surface,
        title,
        zlabel="nm",
        face=face,
        text=text,
    )
    dialog_3d.resize(900, 760)
    _render(dialog_3d, "03_wavefront_3d.png", app)

    height, width = surface.shape
    p1 = (0.18 * width, 0.55 * height)
    p2 = (0.82 * width, 0.55 * height)
    dialog_profile = ProfileDialog(
        surface,
        p1,
        p2,
        title,
        face=face,
        text=text,
    )
    dialog_profile.resize(1100, 520)
    _render(dialog_profile, "04_profile_line.png", app)

    dialog_profile.close()
    dialog_3d.close()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
