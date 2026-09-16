"""Comparison figures.

Every project produces the same three kinds of picture, so they are written once
here: an N-up grid of method outputs, a before/after pair, and an error heatmap.

All functions take **RGB uint8** images (see :mod:`shared.io`) and write a PNG.
Matplotlib is driven through the non-interactive ``Agg`` backend so figures
render identically in CI, where there is no display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import; CI has no display

import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .io import ensure_rgb, to_float  # noqa: E402

_TITLE_SIZE = 10
_DPI = 130

#: Widest a comparison gallery is allowed to be, in pixels. GitHub renders a
#: README image at roughly 900 px, so this keeps ~2.5x of headroom for opening
#: the file full size while stopping a nine-column figure reaching 4406 px.
GALLERY_MAX_PX = 2400


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


def merge_close_marks(
    vlines: dict[str, float] | None, tol: float = 4.0
) -> list[tuple[str, float]]:
    """Group threshold markers that nearly coincide into one labelled line.

    Two thresholds landing within a few grey levels of each other is not an edge
    case — it is what happens when a method is performing *well*, which is
    exactly when you least want the figure to look broken. Rotated labels
    stacked on top of each other are illegible, so near-coincident marks are
    drawn once and labelled together.
    """
    if not vlines:
        return []
    items = sorted(vlines.items(), key=lambda kv: kv[1])
    groups: list[list[tuple[str, float]]] = [[items[0]]]
    for label, x in items[1:]:
        if abs(x - groups[-1][-1][1]) <= tol:
            groups[-1].append((label, x))
        else:
            groups.append([(label, x)])

    out = []
    for g in groups:
        x = float(np.mean([v for _, v in g]))
        if len(g) == 1:
            out.append((f"{g[0][0]} = {g[0][1]:.0f}", x))
        else:
            out.append((" = ".join(f"{lbl} {v:.0f}" for lbl, v in g), x))
    return out


def histogram(
    series: dict[str, np.ndarray],
    out_path: str | Path,
    bins: int = 128,
    value_range: tuple[float, float] = (0, 255),
    xlabel: str = "pixel value",
    ylabel: str = "fraction of pixels",
    title: str = "",
    vlines: dict[str, float] | None = None,
    fill: bool = True,
) -> Path:
    """Overlaid pixel-value distributions, with optional labelled thresholds.

    This is the figure that turns "the histogram stopped being bimodal" from an
    assertion into evidence. A thresholding method picks a single number on this
    axis; drawing that number on the distribution shows immediately whether it
    landed in the valley between two classes or somewhere arbitrary.

    Densities are normalised to fractions so distributions over different-sized
    regions can be compared on one pair of axes.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = ["#2f6f9f", "#c2632c", "#2f6f4f", "#8b4a8b", "#7a7a7a"]

    for i, (label, values) in enumerate(series.items()):
        v = np.asarray(values).ravel()
        counts, edges = np.histogram(v, bins=bins, range=value_range)
        counts = counts / max(counts.sum(), 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        c = colours[i % len(colours)]
        ax.plot(centres, counts, color=c, linewidth=1.7, label=label)
        if fill:
            ax.fill_between(centres, counts, color=c, alpha=0.18)

    for label, xv in merge_close_marks(vlines):
        ax.axvline(xv, color="#b03030", linestyle="--", linewidth=1.5)
        ax.annotate(
            label,
            xy=(xv, 0.97),
            xycoords=("data", "axes fraction"),
            fontsize=_TITLE_SIZE - 2,
            color="#b03030",
            rotation=90,
            ha="right",
            va="top",
        )

    ax.set_xlabel(xlabel, fontsize=_TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=_TITLE_SIZE)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE + 1)
    ax.legend(fontsize=_TITLE_SIZE - 1, frameon=False)
    ax.grid(alpha=0.2, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def confusion_matrix(
    matrix: np.ndarray,
    out_path: str | Path,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    normalise: str = "row",
    cmap: str = "Blues",
) -> Path:
    """An annotated confusion matrix showing both counts and percentages.

    Both are shown deliberately. Counts alone hide how unbalanced the classes
    are; percentages alone hide that a whole row may rest on a handful of pixels.
    For a subject/background split the background is usually the large majority,
    so a 99% background score and a 5% hair score can sit in the same table and
    only the counts explain why the overall number still looks good.

    ``normalise`` is "row" (recall per true class), "all", or "none".
    """
    m = np.asarray(matrix, dtype=np.float64)
    if normalise == "row":
        denom = m.sum(axis=1, keepdims=True)
        shown = np.divide(m, np.maximum(denom, 1e-9))
    elif normalise == "all":
        shown = m / max(m.sum(), 1e-9)
    else:
        shown = m

    fig, ax = plt.subplots(figsize=(1.9 * len(col_labels) + 2.2, 1.5 * len(row_labels) + 1.9))
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=shown.max() if shown.max() > 0 else 1)

    ax.set_xticks(range(len(col_labels)), col_labels, fontsize=_TITLE_SIZE - 1)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=_TITLE_SIZE - 1)
    ax.set_xlabel("predicted", fontsize=_TITLE_SIZE)
    ax.set_ylabel("actual", fontsize=_TITLE_SIZE)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE + 1)

    threshold = (shown.max() if shown.max() > 0 else 1) * 0.55
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            colour = "white" if shown[i, j] > threshold else "#222222"
            count = m[i, j]
            label = f"{count / 1000:.1f}k" if count >= 10_000 else f"{count:,.0f}"
            ax.text(
                j, i - 0.12, label, ha="center", va="center", fontsize=_TITLE_SIZE, color=colour
            )
            ax.text(
                j,
                i + 0.20,
                f"{shown[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=_TITLE_SIZE - 2,
                color=colour,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def value_matrix(
    panels: list[tuple[str, np.ndarray]],
    out_path: str | Path,
    cmap: str = "gray",
    vmin: float = 0,
    vmax: float = 255,
    fmt: str = "{:.0f}",
    title: str = "",
) -> Path:
    """Render small 2-D arrays as colour-coded grids with the numbers printed.

    An image is a matrix, and at some point it is worth showing it as one. A
    12x12 patch of a page printed as raw values makes concrete what "the shadow
    pushed the paper below the ink" actually means, in a way no rendered picture
    can: the reader can read the two numbers and compare them.
    """
    n = len(panels)
    if n == 0:
        raise ValueError("value_matrix() needs at least one panel")

    rows, cols = panels[0][1].shape
    fig, axes = plt.subplots(1, n, figsize=(n * (cols * 0.42 + 0.9), rows * 0.42 + 1.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, patch) in zip(axes, panels):
        p = np.asarray(patch)
        ax.imshow(p, cmap=cmap, vmin=vmin, vmax=vmax)
        mid = (vmin + vmax) / 2
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                ax.text(
                    j,
                    i,
                    fmt.format(p[i, j]),
                    ha="center",
                    va="center",
                    fontsize=max(4.5, 9 - 0.22 * p.shape[1]),
                    color="#111111" if p[i, j] > mid else "#f2f2f2",
                )
        ax.set_title(label, fontsize=_TITLE_SIZE)
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=_TITLE_SIZE + 2, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def comparison_matrix(
    rows: list[dict],
    columns: list[tuple[str, str, bool]],
    out_path: str | Path,
    row_key: str = "method",
    title: str = "",
    cmap: str = "RdYlGn",
    scale: str = "rank",
) -> Path:
    """A methods x metrics heatmap with the real values printed in each cell.

    ``columns`` is a list of ``(header, key, higher_is_better)``. Each column is
    normalised **independently** to [0, 1] for colour only — the printed number is
    always the true value. Independent scaling is what makes the grid readable at
    all: milliseconds, IoU and decibels do not share an axis, and colouring them
    on one scale would make every cell the same shade.

    ``higher_is_better`` flips the colour ramp per column, so green always means
    "better" whether the metric is an accuracy or an error.

    This is the figure that answers "which method should I use" in one look, and
    it is also the figure that exposes when different columns disagree about the
    winner — which happens more often than method comparisons usually admit.
    """
    if not rows or not columns:
        raise ValueError("comparison_matrix() needs rows and columns")

    labels = [str(r.get(row_key, "?")) for r in rows]
    raw = np.full((len(rows), len(columns)), np.nan)
    for j, (_, key, _) in enumerate(columns):
        for i, r in enumerate(rows):
            v = r.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                raw[i, j] = float(v)

    shaded = np.full_like(raw, np.nan)
    for j, (_, _, higher_better) in enumerate(columns):
        col = raw[:, j]
        idx = np.flatnonzero(np.isfinite(col))
        if idx.size == 0:
            continue
        vals = col[idx]

        if idx.size == 1 or float(vals.max() - vals.min()) < 1e-12:
            shaded[idx, j] = 0.5
            continue

        if scale == "rank":
            # Dense rank, not min-max. One catastrophic outlier (a baseline 20x
            # worse than everything else) compresses a linear scale until every
            # real method is the same shade of green. Ranking keeps the ordering
            # visible; the exact value is printed in the cell anyway.
            #
            # np.unique gives the *dense* rank, which matters: ordinal ranking
            # would hand five methods tied at 100% five different shades and
            # invent a ranking that the numbers do not support.
            uniq, inv = np.unique(vals, return_inverse=True)
            if uniq.size == 1:
                shaded[idx, j] = 0.5
                continue
            norm = inv / (uniq.size - 1)
        else:
            lo, hi = float(vals.min()), float(vals.max())
            norm = (vals - lo) / (hi - lo)

        shaded[idx, j] = norm if higher_better else 1.0 - norm

    fig, ax = plt.subplots(
        figsize=(1.55 * len(columns) + 3.4, 0.62 * len(rows) + 1.9)
    )
    ax.imshow(np.ma.masked_invalid(shaded), cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(columns)), [c[0] for c in columns], fontsize=_TITLE_SIZE - 1)
    ax.set_yticks(range(len(labels)), labels, fontsize=_TITLE_SIZE - 1)
    ax.tick_params(axis="x", labelrotation=22)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=_TITLE_SIZE - 1,
                        color="#777777")
                continue
            text = f"{v:,.0f}" if abs(v) >= 100 else (f"{v:.3g}" if abs(v) >= 0.01 else f"{v:.2e}")
            ax.text(j, i, text, ha="center", va="center", fontsize=_TITLE_SIZE - 1,
                    color="#141414")

    for j in range(len(columns) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)
    for i in range(len(labels) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)

    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE + 1, pad=12)
    ax.set_xlabel("green = better in that column", fontsize=_TITLE_SIZE - 2, labelpad=8)

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


def gallery(
    columns: Sequence[str],
    rows: Sequence[tuple[str, Sequence[np.ndarray]]],
    out_path: str | Path,
    suptitle: str = "",
    figsize_scale: float = 3.4,
    cell_notes: Sequence[Sequence[str]] | None = None,
    numbered: bool = True,
) -> Path:
    """A samples-across, stages-down gallery: one column per image, one row per stage.

    ``columns`` names the images; ``rows`` is ``[(stage_label, [img_per_column])]``.

    This exists because a single before/after picture is not evidence. Every
    project in this repo aggregates its *numbers* over four to six images and
    then illustrated them with **one**, which lets the reader assume the one was
    representative — and in several projects it is not. Project 10's advantage is
    +20 points on `chelsea` and −1 on `coffee`; project 01's best detector fails
    on a real photograph. Showing four columns makes the variance part of the
    figure instead of a caveat in the text.
    """
    if not columns or not rows:
        raise ValueError("gallery() needs at least one column and one row")
    ncols, nrows = len(columns), len(rows)

    # Letterbox every panel onto one common canvas. Source images have different
    # aspect ratios, and without this the rows do not line up and the stage
    # labels on the left point at nothing in particular.
    # Per ROW, not globally: a pipeline's stages legitimately change shape (a
    # photo is landscape, the page rectified out of it is portrait), and padding
    # everything to one canvas wastes most of the figure on black bars. Within a
    # row the panels share a size, so the columns still line up.
    def fit_row(imgs):
        box_h = max(im.shape[0] for im in imgs)
        box_w = max(im.shape[1] for im in imgs)
        out = []
        for im in imgs:
            h, w = im.shape[:2]
            if (h, w) == (box_h, box_w):
                out.append(im)
                continue
            scale = min(box_h / h, box_w / w)
            new = cv2.resize(
                im,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.zeros((box_h, box_w) + im.shape[2:], im.dtype)
            y0 = (box_h - new.shape[0]) // 2
            x0 = (box_w - new.shape[1]) // 2
            canvas[y0 : y0 + new.shape[0], x0 : x0 + new.shape[1]] = new
            out.append(canvas)
        return out

    rows = [(label, fit_row(list(imgs))) for label, imgs in rows]
    if numbered:
        # "Sr 1", "Sr 2", ... down the left margin. This figure is a comparison
        # table made of pictures, and a table's rows are numbered so a reader can
        # point at one -- "row 3 is where Hough breaks" -- instead of describing
        # it.
        rows = [
            ("Sr " + str(i) + "\n" + label, imgs)
            for i, (label, imgs) in enumerate(rows, start=1)
        ]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * figsize_scale, nrows * figsize_scale), squeeze=False
    )
    for r, (label, images) in enumerate(rows):
        if len(images) != ncols:
            raise ValueError(f"row {label!r} has {len(images)} images, expected {ncols}")
        for c, img in enumerate(images):
            ax = axes[r][c]
            # the column name on the top row, and a per-cell score underneath it
            # wherever the caller supplied one — a comparison grid without the
            # numbers in it makes the reader estimate by eye what the code
            # already measured
            head = columns[c] if r == 0 else ""
            note = cell_notes[r][c] if cell_notes is not None else ""
            title = "\n".join(t for t in (head, note) if t)
            _show(ax, img, title)
            if c == 0:
                # the stage label goes on the left margin, once per row, rather
                # than being repeated in every panel title
                ax.set_ylabel(label, fontsize=_TITLE_SIZE, rotation=90, labelpad=8)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=_TITLE_SIZE + 3, y=0.998)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # A gallery is the widest figure in the repo. Nine columns at the standard
    # 130 dpi came out 4406 px wide and 12.8 MB, and GitHub renders it at about
    # 900 px. Fifty-eight projects at that size is gigabytes of repository for
    # detail nobody can see. Scale the dpi so a wide gallery lands near
    # GALLERY_MAX_PX -- still roughly 2.5x what a README displays, so it stays
    # sharp when opened full size -- and leave narrow ones at full resolution.
    width_in = ncols * figsize_scale
    dpi = min(_DPI, max(60, int(GALLERY_MAX_PX / max(width_in, 1))))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
