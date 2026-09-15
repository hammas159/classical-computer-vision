"""Reusable interactive components for the Streamlit apps.

Every project's app needs the same three things beyond its own pictures: a pixel
distribution, a readable grid of raw pixel values, and a comparative
methods-x-metrics matrix. They are built once here.

**This module deliberately does not import streamlit.** It returns matplotlib
figures and pandas Stylers, and the app calls ``st.pyplot`` / ``st.dataframe`` on
them. That keeps the shared layer importable and testable without a Streamlit
runtime, which matters because CI runs the tests headless.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .figures import merge_close_marks  # noqa: E402

Column = tuple[str, str, bool]  # (header, key, higher_is_better)


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #


def histogram_figure(
    series: dict[str, np.ndarray],
    vlines: dict[str, float] | None = None,
    bins: int = 96,
    value_range: tuple[float, float] = (0, 255),
    xlabel: str = "pixel value",
    ylabel: str = "fraction of pixels",
    title: str = "",
    figsize: tuple[float, float] = (7.0, 3.4),
):
    """A matplotlib Figure of overlaid pixel distributions with threshold markers.

    Returns the Figure rather than saving it, so an app can hand it straight to
    ``st.pyplot``. Callers should close it afterwards (``plt.close(fig)``) or a
    long-lived app accumulates figures until matplotlib warns.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#2f6f9f", "#c2632c", "#2f6f4f", "#8b4a8b", "#7a7a7a"]

    for i, (label, values) in enumerate(series.items()):
        v = np.asarray(values).ravel()
        if v.size == 0:
            continue
        counts, edges = np.histogram(v, bins=bins, range=value_range)
        counts = counts / max(counts.sum(), 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        c = colours[i % len(colours)]
        ax.plot(centres, counts, color=c, linewidth=1.7, label=label)
        ax.fill_between(centres, counts, color=c, alpha=0.18)

    for label, xv in merge_close_marks(vlines):
        ax.axvline(xv, color="#b03030", linestyle="--", linewidth=1.5)
        ax.annotate(
            label,
            xy=(xv, 0.97),
            xycoords=("data", "axes fraction"),
            fontsize=8,
            color="#b03030",
            rotation=90,
            ha="right",
            va="top",
        )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    # only draw a legend if something was actually plotted; calling it on an
    # empty axes emits a UserWarning on every rerun of a live app
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.2, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def lines_figure(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    logx: bool = False,
):
    """One line per method over a swept parameter, computed live in the app.

    The static counterpart is :func:`shared.figures.lines`, which writes a PNG for
    the README. This returns the figure instead, so a slider can move and the
    curve can be recomputed for *the user's* image rather than the repo's.
    """
    if not series:
        raise ValueError("lines_figure() needs at least one series")

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    for label, ys in series.items():
        ax.plot(x, ys, marker="o", markersize=4, linewidth=1.8, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# matrices
# --------------------------------------------------------------------------- #


def pixel_grid(patch: np.ndarray, cmap: str = "gray", fmt: str = "{:.0f}"):
    """A pandas Styler showing a small patch's raw values, colour-coded.

    An image *is* a matrix, and for a thresholding question the numbers are the
    argument: "shadowed paper reads 74 and the global cut was 100" is settled by
    reading two cells.

    Keep the patch small — roughly 16x16 or less — or the table stops being
    readable and becomes a picture with extra steps.
    """
    p = np.asarray(patch)
    if p.ndim != 2:
        raise ValueError(f"pixel_grid needs a 2-D patch, got shape {p.shape}")
    df = pd.DataFrame(p)
    df.columns = [f"x{c}" for c in range(p.shape[1])]
    df.index = [f"y{r}" for r in range(p.shape[0])]
    return df.style.background_gradient(cmap=cmap, vmin=0, vmax=255).format(fmt)


def comparison_table(
    rows: Sequence[dict],
    columns: Sequence[Column],
    row_key: str = "method",
    cmap: str = "RdYlGn",
):
    """A pandas Styler of the methods-x-metrics matrix, one colour ramp per column.

    Each column is coloured on its own scale because milliseconds, IoU and
    decibels do not share an axis. ``higher_is_better`` flips the ramp so green
    always means better, whether the column is an accuracy or an error.

    Returns ``(styler, dataframe)`` — the frame is handed back so the caller can
    sort, filter or download it without recomputing anything.
    """
    if not rows or not columns:
        raise ValueError("comparison_table() needs rows and columns")

    data = {}
    for header, key, _ in columns:
        data[header] = [r.get(key) for r in rows]
    df = pd.DataFrame(data, index=[str(r.get(row_key, "?")) for r in rows])

    styler = df.style.format(precision=3, na_rep="n/a")
    for header, _, higher_better in columns:
        col = pd.to_numeric(df[header], errors="coerce")
        if col.notna().sum() < 2 or col.nunique(dropna=True) < 2:
            continue
        styler = styler.background_gradient(
            cmap=cmap if higher_better else f"{cmap}_r", subset=[header]
        )
    return styler, df


def region_breakdown(
    per_method: dict[str, dict[str, float]], cmap: str = "RdYlGn"
):
    """A methods-x-regions matrix, e.g. how much of body / hair / background was got right.

    Kept separate from :func:`comparison_table` because every cell here is the
    same kind of quantity (a fraction in [0, 1]), so one shared colour scale is
    correct and makes the rows directly comparable — which is the whole point of
    splitting a score by region.
    """
    df = pd.DataFrame(per_method).T
    return df.style.background_gradient(cmap=cmap, vmin=0, vmax=1).format("{:.3f}"), df


def confusion_frame(
    matrix: np.ndarray, row_labels: list[str], col_labels: list[str], normalise: str = "row"
):
    """A confusion matrix as a styled frame showing counts and row percentages."""
    m = np.asarray(matrix, dtype=np.float64)
    if normalise == "row":
        shown = m / np.maximum(m.sum(axis=1, keepdims=True), 1e-9)
    elif normalise == "all":
        shown = m / max(m.sum(), 1e-9)
    else:
        shown = m

    counts = pd.DataFrame(m.astype(np.int64), index=row_labels, columns=col_labels)
    pct = pd.DataFrame(shown, index=row_labels, columns=col_labels)
    labelled = counts.astype(str) + pct.map(lambda v: f"\n({v * 100:.1f}%)")
    styler = pct.style.background_gradient(cmap="Blues", vmin=0, vmax=1).format("{:.1%}")
    return styler, counts, labelled
