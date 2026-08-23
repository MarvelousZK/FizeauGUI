# -*- coding: utf-8 -*-
"""
Fourier fringe analysis - publication-style figure generator (Nature style)
===========================================================================
Generates every figure of the pipeline from fringe formation to phase
unwrapping, sized and typeset following the open-source SciencePlots
nature.mplstyle conventions (https://github.com/garrettj403/SciencePlots):

  - single column width 3.5 in, full width 7.2 in (183 mm)
  - all text 8 pt, panel labels 9 pt bold
  - sans-serif fonts, muted colorblind-friendly palette
  - thin 0.5 pt axes, inward ticks, tight bounding boxes
  - 300 dpi PNG for slides/Word + vector PDF for print

Pipeline:
    surface phase + carrier -> interferogram -> 2-D FFT
    -> spectrum (DC / +-1 orders) -> bandpass around one sideband
    -> shift sideband to baseband -> inverse FFT gives complex field g
    -> wrapped phase -> phase unwrapping -> recovery vs truth,
    plus reference subtraction.

Usage:
    python make_fft_fringe_figures.py

Output (this directory):
    F00_fft_fringe_pipeline_overview.png/.pdf
    F01_carrier_and_surface_make_fringes.png/.pdf
    F02_interferogram_profile.png/.pdf
    F03_fft_spectrum_three_orders.png/.pdf
    F04_bandpass_single_sideband.png/.pdf
    F05_shift_to_baseband.png/.pdf
    F06_inverse_complex_field.png/.pdf
    F07_wrapped_phase.png/.pdf
    F08_phase_unwrapping.png/.pdf
    F09_recovery_vs_truth.png/.pdf
    F10_reference_subtraction.png/.pdf
    figure_captions.md

Dependencies: numpy / matplotlib / scipy.
"""

from __future__ import annotations

import os
from cycler import cycler

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless rendering
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- global setup
# The script lives inside the figures folder: images are written next to it.
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = HERE
os.makedirs(OUTDIR, exist_ok=True)

# Nature-style rcParams, adapted from SciencePlots nature.mplstyle
# (usetex kept off so no LaTeX installation is required).
plt.rcParams.update({
    # figure size / dpi
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # typography: 8 pt text, 9 pt bold panel labels (slightly larger
    # than the strict 7 pt Nature minimum, for on-screen teaching use)
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    # thin lines, inward ticks
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    # SciencePlots muted color cycle
    "axes.prop_cycle": cycler("color", [
        "#0C5DA5", "#00B945", "#FF9500", "#FF2C00",
        "#845B97", "#474747", "#9E9E9E"]),
})

# Nature widths: single column 89 mm, full two-column 183 mm
W_SINGLE = 3.5
W_FULL = 7.2

# ---------------------------------------------------------------- simulation
N = 640                     # image side length (px)
CX, CY = N / 2, N / 2       # aperture center
R = 240                     # aperture radius (px)
F0 = 1.0 / 16.0             # spatial carrier frequency (cyc/px), period 16 px
SHIFT = int(round(F0 * N))  # +1-order offset in the spectrum (bins)
SIGMA = 12.0                # Gaussian bandpass sigma (bins)
NOISE = 0.012               # Gaussian noise amplitude

X, Y = np.meshgrid(np.arange(N, dtype=np.float64),
                   np.arange(N, dtype=np.float64))
U = (X - CX) / R
V = (Y - CY) / R
RHO = np.hypot(U, V)
MASK = RHO <= 1.0

RNG = np.random.default_rng(20231031)


def surface_phase(u, v, rho, kind: str) -> np.ndarray:
    """Synthetic smooth surface phase (rad). kind: 'test' or 'ref'."""
    if kind == "test":
        phi = (0.8 * u + 0.5 * v
               + 5.0 * (rho ** 2 - 0.5)
               + 2.5 * (u ** 2 - v ** 2))
    else:
        phi = (0.20 * u + 0.15 * v
               + 0.8 * (rho ** 2 - 0.5)
               + 0.3 * (u ** 2 - v ** 2))
    return phi * (rho <= 1.0)


def interferogram(phi: np.ndarray, f0: float = F0,
                  noise: float = NOISE) -> np.ndarray:
    """I = a + m*a*cos(2*pi*f0*x + phi), Gaussian illumination."""
    a = 0.62 + 0.38 * np.exp(-(RHO / 0.85) ** 2)
    m = 0.50 * (1.0 - 0.08 * RHO ** 2)
    I = a + m * a * np.cos(2.0 * np.pi * f0 * X + phi)
    I = I * MASK
    if noise > 0:
        I = I + RNG.standard_normal(I.shape) * noise
    return I


# ---------------------------------------------------------------- algorithm
def fft_spectrum(I: np.ndarray, window: np.ndarray) -> np.ndarray:
    """2-D FFT with the zero frequency shifted to the center."""
    return np.fft.fftshift(np.fft.fft2(I * window))


def bandpass_mask(shift: int, sigma: float = SIGMA) -> np.ndarray:
    """2-D Gaussian bandpass window around the +1 order (fx = +f0)."""
    ky, kx = np.indices((N, N), dtype=np.float64)
    return np.exp(-(((kx - (N / 2 + shift)) ** 2
                     + (ky - N / 2) ** 2) / (2.0 * sigma ** 2)))


def shift_to_baseband(F: np.ndarray, shift: int) -> np.ndarray:
    """Roll the +1 order to the zero-frequency position (carrier removal)."""
    return np.roll(F, -shift, axis=1)


def inverse_fft(F_base: np.ndarray) -> np.ndarray:
    """Inverse FFT of the baseband spectrum -> complex field g."""
    return np.fft.ifft2(np.fft.ifftshift(F_base))


def takeda_phase(I: np.ndarray, window: np.ndarray,
                 shift: int = SHIFT, sigma: float = SIGMA):
    """Takeda front half; returns every intermediate array and wrapped phase."""
    F = fft_spectrum(I, window)
    G = bandpass_mask(shift, sigma)
    F_filt = F * G
    F_base = shift_to_baseband(F_filt, shift)
    g = inverse_fft(F_base)
    wrapped = np.angle(g)
    return F, G, F_filt, F_base, g, wrapped


def unwrap2d(wrapped: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ideal-data unwrapping used only by the FT teaching figure."""
    out = np.unwrap(np.unwrap(wrapped, axis=1), axis=0)
    return np.where(mask, out, 0.0)


def remove_plane(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove piston and x/y tilt so the result can be compared with truth."""
    idx = mask.ravel()
    A = np.column_stack([np.ones(idx.sum()),
                         X.ravel()[idx], Y.ravel()[idx]])
    coef, *_ = np.linalg.lstsq(A, arr.ravel()[idx], rcond=None)
    return arr - (coef[0] + coef[1] * X + coef[2] * Y)


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Set pixels outside the aperture to NaN for display."""
    return np.where(mask, arr, np.nan)


# ---------------------------------------------------------------- plotting
def save(fig, name: str):
    """Save both 300-dpi PNG and vector PDF with a tight bounding box."""
    for ext in (".png", ".pdf"):
        path = os.path.join(OUTDIR, name.replace(".png", ext))
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  generated {name.replace('.png', '.png/.pdf')}")


def panel(ax, letter: str):
    """Bold 8 pt panel label at the top-left corner (Nature convention)."""
    ax.text(0.02, 0.98, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="top")


def imshow(ax, arr, title, cmap="gray", vmin=None, vmax=None,
           cbar=False, cbar_label=""):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="upper", interpolation="nearest")
    ax.set_title(title, pad=3)
    ax.set_xticks([0, N // 2, N])
    ax.set_yticks([0, N // 2, N])
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.tick_params(top=False, right=False)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=cbar_label)
    return im


def style_xy(ax):
    """Nature-style x-y panel: no top/right spines, inward ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.grid(alpha=0.3, linewidth=0.4)


def spectrum_2d(F: np.ndarray):
    """log(1+|F|) magnitude spectrum for display."""
    return np.log10(1.0 + np.abs(F))


# ---------------------------------------------------------------- main
def main():
    print("Generating publication-style teaching figures ...")
    phi_test = surface_phase(U, V, RHO, "test")
    phi_ref = surface_phase(U, V, RHO, "ref")
    I_test = interferogram(phi_test)
    I_ref = interferogram(phi_ref)

    # Full pipeline for the test surface (hard circular aperture)
    WINDOW = MASK.astype(np.float64)
    F, G, F_filt, F_base, g, wrapped = takeda_phase(I_test, WINDOW)
    unwrapped = unwrap2d(wrapped, MASK)
    recovered = remove_plane(unwrapped, MASK)
    truth = remove_plane(phi_test, MASK)
    error = recovered - truth
    err_rms = float(np.sqrt(np.mean(error[MASK] ** 2)))
    spec = spectrum_2d(F)
    spec_filt = spectrum_2d(F_filt)
    spec_base = spectrum_2d(F_base)
    amp = np.abs(g)

    row = int(CY)
    xpix = np.arange(N)
    profile = I_test[row, :]
    fx = (np.arange(N) - N / 2) / N        # cyc/px
    spec_slice = spec[N // 2, :]
    wrapped_slice = wrapped[row, :]
    unwrapped_slice = unwrapped[row, :]

    # =============== F00: seven-step overview ===============
    fig, axes = plt.subplots(1, 7, figsize=(W_FULL, 1.85))
    panels = [
        ("1 Interferogram", I_test, "gray", None, None),
        ("2 FFT spectrum", spec, "cividis", None, None),
        ("3 Bandpass", spec_filt, "cividis", None, None),
        ("4 Baseband", spec_base, "cividis", None, None),
        ("5 |g|", masked(amp, MASK), "viridis", None, None),
        ("6 Wrapped phase", masked(wrapped, MASK), "coolwarm", None, None),
        ("7 Unwrapped", masked(recovered, MASK), "coolwarm", None, None),
    ]
    for ax, (title, arr, cmap, _, _) in zip(axes, panels):
        imshow(ax, arr, title, cmap=cmap)
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.4)
    save(fig, "F00_fft_fringe_pipeline_overview.png")

    # =============== F01: carrier + surface -> fringes ===============
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.35))
    carrier = np.cos(2 * np.pi * F0 * X) * MASK
    imshow(axes[0], carrier, "Carrier fringes", cmap="gray")
    imshow(axes[1], masked(phi_test, MASK), "Surface phase",
           cmap="coolwarm", cbar=True, cbar_label="rad")
    imshow(axes[2], I_test, "Interferogram", cmap="gray")
    for letter, ax in zip("abc", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F01_carrier_and_surface_make_fringes.png")

    # =============== F02: interferogram + 1-D profile ===============
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.55),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    imshow(axes[0], I_test, "Fringe pattern", cmap="gray")
    axes[1].plot(xpix, profile, color="#0C5DA5", linewidth=1.0)
    p = profile
    peaks = np.nonzero((p[1:-1] > p[0:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    peaks = peaks[(peaks > CX - R + 20) & (peaks < CX + R - 20)]
    if len(peaks) >= 2:
        x1, x2 = int(peaks[0]), int(peaks[1])
        axes[1].annotate("", xy=(x2, p[x2]), xytext=(x1, p[x1]),
                         arrowprops=dict(arrowstyle="<->", color="#FF2C00"))
        axes[1].text((x1 + x2) / 2, p[x1:x2].min() - 0.06,
                     f"period = {x2 - x1} px", ha="center",
                     color="#FF2C00", fontsize=8)
        axes[1].plot([x1, x2], [p[x1], p[x2]], "o", color="#FF2C00",
                     markersize=3)
    axes[1].set_title("Central-row profile", pad=3)
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("Intensity I")
    axes[1].set_xlim(CX - R, CX + R)
    style_xy(axes[1])
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F02_interferogram_profile.png")

    # =============== F03: spectrum with DC / +-1 orders ===============
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.55),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    imshow(axes[0], spec, "Log magnitude spectrum", cmap="cividis",
           cbar=True, cbar_label="log(1+|F|)")
    half = N // 2
    for kx, color in ((half, "#00B945"),
                      (half + SHIFT, "#FF2C00"),
                      (half - SHIFT, "#FF2C00")):
        axes[0].plot([kx], [half], "o", color=color, markersize=3.5)
    axes[0].annotate("DC", (half, half - 22), color="#00B945",
                     fontsize=8, fontweight="bold", ha="center")
    axes[0].annotate("+1", (half + SHIFT + 8, half - 3), color="#FF2C00",
                     fontsize=8, fontweight="bold", ha="left")
    axes[0].annotate("-1", (half - SHIFT - 8, half - 3), color="#FF2C00",
                     fontsize=8, fontweight="bold", ha="right")
    axes[1].plot(fx, spec_slice, color="#845B97", linewidth=1.0)
    axes[1].set_title("Slice along fx", pad=3)
    axes[1].set_xlabel("fx (cyc/px)")
    axes[1].set_ylabel("log(1+|F|)")
    for fx0, name, color in ((0, "DC", "#00B945"),
                             (F0, "+f0", "#FF2C00"),
                             (-F0, "-f0", "#FF2C00")):
        axes[1].axvline(fx0, color=color, linestyle="--", linewidth=0.6)
        axes[1].text(fx0, spec_slice[half] + 0.12, name, ha="center",
                     color=color, fontsize=8)
    axes[1].set_xlim(-0.18, 0.18)
    style_xy(axes[1])
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F03_fft_spectrum_three_orders.png")

    # =============== F04: bandpass around +1 order ===============
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.35))
    imshow(axes[0], spec, "Original spectrum", cmap="cividis")
    imshow(axes[1], G, "Gaussian bandpass", cmap="gray",
           cbar=True, cbar_label="weight")
    imshow(axes[2], spec_filt, "After bandpass", cmap="cividis")
    for ax in axes:
        ax.set_xlim(N // 2 - 70, N // 2 + 70)
        ax.set_ylim(N // 2 - 70, N // 2 + 70)
    for letter, ax in zip("abc", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F04_bandpass_single_sideband.png")

    # =============== F05: shift to baseband ===============
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.45))
    imshow(axes[0], spec_filt, "Before shift", cmap="cividis")
    imshow(axes[1], spec_base, "After shift to zero frequency",
           cmap="cividis")
    for ax in axes:
        ax.set_xlim(N // 2 - 70, N // 2 + 70)
        ax.set_ylim(N // 2 - 70, N // 2 + 70)
        ax.plot([N // 2], [N // 2], "+", color="#00B945",
                markersize=6, markeredgewidth=1)
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F05_shift_to_baseband.png")

    # =============== F06: inverse FFT -> complex field g ===============
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.55),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    imshow(axes[0], masked(amp, MASK), "Complex-field magnitude |g|",
           cmap="viridis", cbar=True, cbar_label="|g|")
    axes[1].plot(xpix, amp[row, :], color="#00B945", linewidth=1.0)
    axes[1].set_title("Central-row |g| profile", pad=3)
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("|g|")
    axes[1].set_xlim(CX - R, CX + R)
    style_xy(axes[1])
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F06_inverse_complex_field.png")

    # =============== F07: wrapped phase ===============
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 2.55),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    imshow(axes[0], masked(wrapped, MASK), "Wrapped phase",
           cmap="coolwarm", vmin=-np.pi, vmax=np.pi, cbar=True,
           cbar_label="rad")
    axes[1].plot(xpix, wrapped_slice, color="#0C5DA5", linewidth=1.0)
    axes[1].axhline(np.pi, color="#FF2C00", linestyle="--", linewidth=0.6)
    axes[1].axhline(-np.pi, color="#FF2C00", linestyle="--", linewidth=0.6)
    axes[1].set_title("Central-row wrapped phase", pad=3)
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].set_ylim(-4.2, 4.2)
    axes[1].set_xlim(CX - R, CX + R)
    style_xy(axes[1])
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F07_wrapped_phase.png")

    # =============== F08: phase unwrapping ===============
    fig = plt.figure(figsize=(W_FULL, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1.plot(xpix, wrapped_slice, color="#0C5DA5", linewidth=1.0)
    ax1.set_title("Before unwrapping", pad=3)
    ax1.set_xlabel("x (px)")
    ax1.set_ylabel("rad")
    ax1.set_ylim(-4.2, 4.2)
    ax1.set_xlim(CX - R, CX + R)
    ax2.plot(xpix, unwrapped_slice, color="#FF9500", linewidth=1.0)
    ax2.set_title("After unwrapping", pad=3)
    ax2.set_xlabel("x (px)")
    ax2.set_ylabel("rad")
    ax2.set_xlim(CX - R, CX + R)
    imshow(ax3, masked(unwrapped, MASK), "Unwrapped phase (2-D)",
           cmap="coolwarm", cbar=True, cbar_label="rad")
    for ax in (ax1, ax2):
        style_xy(ax)
    for letter, ax in zip("abc", (ax1, ax2, ax3)):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F08_phase_unwrapping.png")

    # =============== F09: recovery vs truth ===============
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 2.35))
    vmin = min(float(np.nanmin(masked(truth, MASK))),
               float(np.nanmin(masked(recovered, MASK))))
    vmax = max(float(np.nanmax(masked(truth, MASK))),
               float(np.nanmax(masked(recovered, MASK))))
    imshow(axes[0], masked(truth, MASK), "True surface phase",
           cmap="coolwarm", vmin=vmin, vmax=vmax)
    imshow(axes[1], masked(recovered, MASK), "Recovered phase",
           cmap="coolwarm", vmin=vmin, vmax=vmax, cbar=True, cbar_label="rad")
    imshow(axes[2], masked(error, MASK),
           f"Error (recovered - truth), RMS = {err_rms:.2f} rad",
           cmap="coolwarm")
    for letter, ax in zip("abc", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "F09_recovery_vs_truth.png")

    # =============== F10: reference subtraction ===============
    _, _, _, _, g_ref, _ = takeda_phase(I_ref, WINDOW)
    delta_wrap = np.angle(g * np.conj(g_ref))
    delta_unwrapped = unwrap2d(delta_wrap, MASK)
    delta_rec = remove_plane(delta_unwrapped, MASK)
    delta_true = remove_plane(phi_test - phi_ref, MASK)
    delta_err = delta_rec - delta_true
    delta_rms = float(np.sqrt(np.mean(delta_err[MASK] ** 2)))
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, 1.95))
    imshow(axes[0], masked(delta_wrap, MASK), "Wrapped phase difference",
           cmap="coolwarm", vmin=-np.pi, vmax=np.pi)
    vmin2 = min(float(np.nanmin(masked(delta_true, MASK))),
                float(np.nanmin(masked(delta_rec, MASK))))
    vmax2 = max(float(np.nanmax(masked(delta_true, MASK))),
                float(np.nanmax(masked(delta_rec, MASK))))
    imshow(axes[1], masked(delta_unwrapped, MASK), "Unwrapped difference",
           cmap="coolwarm", vmin=vmin2, vmax=vmax2, cbar=True, cbar_label="rad")
    imshow(axes[2], masked(delta_true, MASK), "True difference",
           cmap="coolwarm", vmin=vmin2, vmax=vmax2)
    imshow(axes[3], masked(delta_err, MASK),
           f"Error, RMS = {delta_rms:.2f} rad", cmap="coolwarm")
    for letter, ax in zip("abcd", axes):
        panel(ax, letter)
    fig.tight_layout(pad=0.2, w_pad=0.5)
    save(fig, "F10_reference_subtraction.png")

    # =============== captions ===============
    captions = [
        ("F00_fft_fringe_pipeline_overview",
         "Seven-step pipeline on one page.",
         "Lecture front page / review slide"),
        ("F01_carrier_and_surface_make_fringes",
         "(a) straight carrier fringes, (b) surface phase, "
         "(c) the final interferogram.",
         "Draft Figure 2"),
        ("F02_interferogram_profile",
         "(a) 2-D fringe pattern; (b) central-row profile with the fringe "
         "period marked in red.",
         "Where the fringe period / carrier frequency is defined"),
        ("F03_fft_spectrum_three_orders",
         "(a) 2-D log spectrum with DC and +-1 orders; (b) 1-D slice along "
         "fx showing the three peaks.",
         "Draft Figure 3"),
        ("F04_bandpass_single_sideband",
         "(a) original spectrum; (b) Gaussian bandpass centered at +f0; "
         "(c) filtered spectrum keeping only the +1 order.",
         "Draft step 4"),
        ("F05_shift_to_baseband",
         "(a) +1 order at f0; (b) shifted to the zero-frequency center "
         "(carrier removal).",
         "Draft step 5"),
        ("F06_inverse_complex_field",
         "(a) magnitude |g| of the inverse-FFT complex field; "
         "(b) central-row profile.",
         "Draft step 6"),
        ("F07_wrapped_phase",
         "(a) wrapped phase 2-D map; (b) central-row sawtooth with +-pi "
         "lines.",
         "Draft step 7"),
        ("F08_phase_unwrapping",
         "(a) wrapped 1-D profile; (b) unwrapped 1-D profile; "
         "(c) unwrapped 2-D phase.",
         "New section: phase unwrapping (missing in the draft)"),
        ("F09_recovery_vs_truth",
         "(a) true surface phase; (b) recovered phase; (c) error map with "
         "RMS.",
         "Closing validation slide"),
        ("F10_reference_subtraction",
         "(a) wrapped delta-phi = angle(g_test * conj(g_ref)); (b) unwrapped "
         "difference; (c) true difference; (d) error map.",
         "Draft step 8"),
    ]
    lines = ["# Fourier Fringe Analysis - Figure List",
             "",
             "Generated by `make_fft_fringe_figures.py` "
             "(Nature-style typography adapted from "
             "https://github.com/garrettj403/SciencePlots).",
             "",
             "| File | Content | Suggested place |",
             "| --- | --- | --- |"]
    for name, desc, where in captions:
        lines.append(f"| `{name}.png/.pdf` | {desc} | {where} |")
    lines += [
        "",
        "## Layout & typography",
        "",
        "- Full-width figures: 7.2 in (183 mm); panel labels: 9 pt bold; "
        "all other text: 8 pt.",
        "- Sans-serif font (DejaVu Sans / Arial), 0.5 pt axes, inward ticks, "
        "diverging maps use the coolwarm colormap.",
        "- PNG is 300 dpi for slides/Word; PDF is vector for print.",
        "",
        "## Simulation parameters (edit at the top of the script)",
        "",
        f"- Image {N} x {N} px, aperture radius {R} px",
        f"- Carrier f0 = 1/{int(1/F0)} cyc/px (fringe period {int(1/F0)} px)",
        f"- Gaussian bandpass sigma = {SIGMA:g} bins, noise amplitude {NOISE}",
        "- Surfaces: tilt + defocus + 0-degree astigmatism (test); "
        "small residual (reference)",
        "",
        "Regenerate everything: `python make_fft_fringe_figures.py`",
    ]
    with open(os.path.join(OUTDIR, "figure_captions.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  generated figure_captions.md")
    print(f"\nDone. Output directory: {OUTDIR}")


if __name__ == "__main__":
    main()
