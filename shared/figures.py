"""Comparison figures.

Every project produces the same three kinds of picture, so they are written once
here: an N-up grid of method outputs, a before/after pair, and an error heatmap.

All functions take **RGB uint8** images (see :mod:`shared.io`) and write a PNG.
Matplotlib is driven through the non-interactive ``Agg`` backend so figures
render identically in CI, where there is no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import; CI has no display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .io import ensure_rgb, to_float  # noqa: E402

_TITLE_SIZE = 10
_DPI = 130


def _show(ax, img: np.ndarray, title: str = "") -> None:
    img = np.asarray(img)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255 if img.dtype == np.uint8 else 1.0)
    else:
        ax.imshow(ensure_rgb(img))
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.axis("off")


def grid(
    panels: list[tuple[str, np.ndarray]],
    out_path: str | Path,
    ncols: int = 3,
    suptitle: str = "",
    figsize_scale: float = 3.6,
) -> Path:
    """Write an N-up comparison grid of ``(title, image)`` panels."""
    n = len(panels)
    if n == 0:
        raise ValueError("grid() needs at least one panel")
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_scale, nrows * figsize_scale)
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, img) in zip(axes, panels):
        _show(ax, img, title)
    for ax in axes[n:]:  # blank any unused cell
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=_TITLE_SIZE + 3, y=0.995)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def pair(
    before: np.ndarray,
    after: np.ndarray,
    out_path: str | Path,
    titles: tuple[str, str] = ("Input", "Output"),
    suptitle: str = "",
) -> Path:
    """Write a side-by-side before/after figure."""
    return grid(
        [(titles[0], before), (titles[1], after)],
        out_path,
        ncols=2,
        suptitle=suptitle,
        figsize_scale=4.2,
    )


def error_heatmap(
    pred: np.ndarray,
    truth: np.ndarray,
    out_path: str | Path,
    title: str = "Absolute error",
) -> Path:
    """Write a heatmap of |pred - truth|, with a colour bar in 0-255 units.

    Worth producing for every restoration project: the error map shows *where* a
    method fails, which a single PSNR number cannot.
    """
    diff = np.abs(to_float(pred) - to_float(truth))
    if diff.ndim == 3:
        diff = diff.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(diff * 255.0, cmap="inferno")
    ax.set_title(title, fontsize=_TITLE_SIZE)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="abs error (0-255)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def lines(
    x: list[float],
    series: dict[str, list[float]],
    out_path: str | Path,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    invert_x: bool = False,
    dashed: set[str] | None = None,
    vlines: dict[str, float] | None = None,
) -> Path:
    """Write a multi-series line plot — the right figure for a parameter sweep.

    A sweep answers "where does this method stop working", which a bar chart of
    one operating point cannot show.

    ``dashed`` names series to draw as dashed lines, which is how a reference or
    oracle curve should be distinguished from a real method. ``vlines`` draws
    labelled vertical markers, for thresholds worth naming on the axis.
    """
    dashed = dashed or set()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (label, ys) in enumerate(series.items()):
        ax.plot(
            x,
            ys,
            marker=markers[i % len(markers)],
            linewidth=1.9,
            markersize=5,
            label=label,
            linestyle="--" if label in dashed else "-",
            color="#444444" if label in dashed else None,
        )

    for label, xv in (vlines or {}).items():
        ax.axvline(xv, color="#b03030", linestyle=":", linewidth=1.4)
        ax.annotate(
            label,
            xy=(xv, 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=_TITLE_SIZE - 2,
            color="#b03030",
            rotation=90,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel(xlabel, fontsize=_TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=_TITLE_SIZE)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE + 1)
    if invert_x:
        ax.invert_xaxis()
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(fontsize=_TITLE_SIZE - 1, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def metric_bars(
    labels: list[str],
    values: list[float],
    out_path: str | Path,
    ylabel: str = "",
    title: str = "",
    highlight_best: str = "max",
) -> Path:
    """Write a bar chart of one metric across methods, best bar highlighted."""
    if len(labels) != len(values):
        raise ValueError("labels and values must be the same length")

    best = int(np.argmax(values) if highlight_best == "max" else np.argmin(values))
    colours = ["#9aa5b1"] * len(values)
    colours[best] = "#2f6f4f"

    fig, ax = plt.subplots(figsize=(max(5.0, 1.15 * len(labels)), 3.8))
    ax.bar(labels, values, color=colours)
    ax.set_ylabel(ylabel, fontsize=_TITLE_SIZE)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE + 1)
    ax.tick_params(axis="x", labelrotation=30, labelsize=_TITLE_SIZE - 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
