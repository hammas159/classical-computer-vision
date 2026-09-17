"""Choose a project's candidate photographs from the cache, on a stated axis.

    python tools/select_images.py --axis tone --n 12 --sheet out.png
    python tools/select_images.py --axis detail --n 12 --pool sipi

The problem this solves
-----------------------
Each project needs 10-12 candidates that **no other project uses** and that
**span whatever that project actually measures**. Those are different
requirements and both are easy to get wrong:

* picking at random gives four pictures of the same kind of thing, and a
  comparison figure whose rows differ only by subject is a comparison of
  subjects;
* picking by eye does not scale to 58 projects, and smuggles in the answer --
  choosing the images where the method you like wins is not an experiment.

So candidates are chosen by a **measured** property, named on the command line,
and the selection spreads them evenly across that property's range. The image
pool is then a stated, reproducible consequence of the axis rather than a taste
judgement, and the axis is printed into the project's README.

The axes are deliberately cheap statistics, not a quality score. Which end of
an axis is "good" is exactly what the project is there to find out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.fetch_images import CACHE_DIR  # noqa: E402

MEASURE_CACHE = CACHE_DIR / "measures.json"


# --------------------------------------------------------------------------- #
# the axes
# --------------------------------------------------------------------------- #


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def m_detail(img: np.ndarray) -> float:
    """Mean gradient magnitude x1000 -- how much fine structure is present.

    The axis for sharpening, denoising, edges and compression: a method that
    preserves detail cannot be judged on images that have none.
    """
    g = _gray(img).astype(np.float32) / 255.0
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(dx, dy)) * 1000)


def m_brightness(img: np.ndarray) -> float:
    """Mean luminance, 0-255."""
    return float(_gray(img).mean())


def m_tone(img: np.ndarray) -> float:
    """How much of the 0-255 range the histogram actually uses, as a percentage.

    The axis for histogram equalisation and tone mapping. An image already
    filling the range has nothing for equalisation to give it, and one crammed
    into a third of it is the case the method exists for. Measured between the
    1st and 99th percentiles so a handful of specular pixels do not report a
    full range that the picture does not have.
    """
    g = _gray(img)
    lo, hi = np.percentile(g, (1, 99))
    return float(hi - lo) / 255.0 * 100


def m_entropy(img: np.ndarray) -> float:
    """Shannon entropy of the luminance histogram, in bits."""
    h = np.bincount(_gray(img).ravel(), minlength=256).astype(np.float64)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def m_colour(img: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness -- the axis for white balance and colour work."""
    if img.ndim == 2:
        return 0.0
    b, g, r = (img[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def m_edge_density(img: np.ndarray) -> float:
    """Percentage of pixels Canny calls an edge, at its own Otsu-derived thresholds."""
    g = _gray(img)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((cv2.Canny(g, 0.5 * t, t) > 0).mean() * 100)


def m_texture(img: np.ndarray) -> float:
    """Fraction of spectral energy above a quarter Nyquist -- high-frequency content.

    Distinguishes a picture of a textured *surface* from a picture of objects
    with smooth faces, which a gradient measure alone does not.
    """
    g = cv2.resize(_gray(img), (256, 256)).astype(np.float32) / 255.0
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    y, x = np.ogrid[:256, :256]
    rad = np.hypot(y - 128, x - 128)
    total = f.sum()
    return float(f[rad > 32].sum() / total * 100) if total > 0 else 0.0


AXES = {
    "detail": m_detail,
    "brightness": m_brightness,
    "tone": m_tone,
    "entropy": m_entropy,
    "colour": m_colour,
    "edges": m_edge_density,
    "texture": m_texture,
}


# --------------------------------------------------------------------------- #
# measuring and choosing
# --------------------------------------------------------------------------- #


def measure_pool(pool: str, refresh: bool = False) -> dict[str, dict[str, float]]:
    """Measure every cached image on every axis, once, and remember the result."""
    cache = json.loads(MEASURE_CACHE.read_text()) if MEASURE_CACHE.exists() else {}
    files = sorted((CACHE_DIR / pool).glob("*"))
    todo = [f for f in files if refresh or f.stem not in cache]
    if todo:
        print(f"measuring {len(todo)} {pool} images on {len(AXES)} axes ...")
        for i, f in enumerate(todo, start=1):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cache[f.stem] = {"pool": pool, "file": f.name}
            cache[f.stem] |= {name: round(fn(img), 4) for name, fn in AXES.items()}
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ...")
        MEASURE_CACHE.write_text(json.dumps(cache, indent=0))
    return {k: v for k, v in cache.items() if v.get("pool") == pool}


def already_used() -> set[str]:
    """Every cache id already committed to `assets/real/`.

    The no-reuse rule is enforced against what is actually in the repository
    rather than against a list someone maintains by hand, because the list is
    what goes stale.

    Matching by *filename* is not enough: images are renamed to something
    descriptive on the way in, so the twelve BSDS photographs already used by
    project 07 look like new candidates to a name-based check. The manifest
    written by `tools/check_image_reuse.py` maps each committed file back to the
    cache entry it came from by perceptual hash, and that is what is excluded
    here. Run that tool after adding images or this set goes stale too.
    """
    manifest = REPO / "assets" / "real" / "manifest.json"
    used = {p.stem for p in (REPO / "assets" / "real").glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    if manifest.exists():
        for entry in json.loads(manifest.read_text()).values():
            if entry.get("source"):
                used.add(entry["source"].split("/", 1)[1])
    return used


def spread(rows: list[dict], axis: str, n: int) -> list[dict]:
    """Take ``n`` images spread evenly across the axis, not the ``n`` most extreme.

    Both tails and the middle are wanted: a project needs to show the case its
    method is for *and* the case it is not for, and an all-extremes pool cannot
    show where the crossover is.
    """
    ranked = sorted(rows, key=lambda r: r[axis])
    if len(ranked) <= n:
        return ranked
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return [ranked[i] for i in dict.fromkeys(idx.tolist())]


def contact_sheet(rows: list[dict], axis: str, out: Path, cols: int = 6) -> None:
    """A labelled grid of the candidates, so the semantic naming is done by eye once.

    The *selection* is computed; what each picture is **of** is not something a
    statistic knows, and four rows that differ by measurement can still be four
    photographs of a bird. This sheet is the step where that is checked.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    r = (n + cols - 1) // cols
    fig, axs = plt.subplots(r, cols, figsize=(cols * 2.7, r * 2.5))
    for ax, row in zip(np.ravel(axs), rows):
        img = cv2.imread(str(CACHE_DIR / row["pool"] / row["file"]), cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['id']}\n{axis} {row[axis]:.1f}", fontsize=7)
        ax.axis("off")
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    fig.suptitle(f"{n} candidates spread across '{axis}'", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", required=True, choices=list(AXES))
    ap.add_argument("--pool", default="bsds", choices=["bsds", "sipi"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sheet", default=None, help="write a labelled contact sheet here")
    ap.add_argument("--min", type=float, default=None, help="reject below this axis value")
    ap.add_argument("--max", type=float, default=None)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    measures = measure_pool(args.pool, refresh=args.refresh)
    used = already_used()
    rows = [dict(v, id=k) for k, v in measures.items() if k not in used]
    print(f"{len(measures)} in pool, {len(measures) - len(rows)} already used by a project")

    if args.min is not None:
        rows = [r for r in rows if r[args.axis] >= args.min]
    if args.max is not None:
        rows = [r for r in rows if r[args.axis] <= args.max]

    chosen = spread(rows, args.axis, args.n)
    print(f"\n{len(chosen)} candidates spread across '{args.axis}':")
    other = [a for a in AXES if a != args.axis]
    print(f"{'id':12s} {args.axis:>10s} | " + " ".join(f"{a[:7]:>7s}" for a in other))
    for r in chosen:
        print(f"{r['id']:12s} {r[args.axis]:10.2f} | "
              + " ".join(f"{r[a]:7.1f}" for a in other))

    if args.sheet:
        contact_sheet(chosen, args.axis, Path(args.sheet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
