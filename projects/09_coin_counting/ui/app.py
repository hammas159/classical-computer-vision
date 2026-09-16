"""Streamlit UI for project 09 — coin counting and measurement.

    streamlit run ui/app.py

Count touching objects, then measure them in millimetres. The count is
objectively right or wrong — no metric choice required — and the measurement is
only as good as one assumed reference, which the app lets you get wrong on
purpose to see what that costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import coins as cn  # noqa: E402
from shared import io as shared_io, theme  # noqa: E402
from shared import ui  # noqa: E402
from shared.bench import timeit  # noqa: E402

st.set_page_config(
    page_title="Coin counting — Classical CV", layout="wide", initial_sidebar_state="collapsed",
    page_icon="🪙",
)

#: Project 09's identity: "Brass on felt". Each of the 58 apps has its own
#: palette, face and corner radius, so a screenshot says which project it
#: came from before the title is read. See shared/theme.py.
PALETTE = theme.apply(9)

st.title("Coin counting and measurement")
st.caption(
    "Counting touching objects is the classic watershed demo, and it usually "
    "stops at \"it found the coins\". Two harder questions: how many does each "
    "method actually get right, and can you then measure them in millimetres? "
    "No training, no network, no GPU."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["The coins plate", "Upload your own"],
        horizontal=True,
        help="The sample image has a hand-counted ground truth of 24 coins.",
    )
    uploaded = None
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Photo of objects on a plain background",
                                    type=["png", "jpg", "jpeg", "bmp", "webp"])
    flatten = st.checkbox(
        "Flatten the illumination first (top-hat)", value=True,
        help="The sample plate is lit unevenly. Turn this off to watch a lighting "
             "artefact destroy one seeding rule and leave the other untouched.",
    )
    reference_mm = st.number_input(
        "Reference: diameter of the LARGEST object (mm)",
        min_value=1.0, max_value=200.0, value=float(cn.REFERENCE_DIAMETER_MM), step=0.25,
        help="Every other measurement is scaled from this one number. Change it "
             "and watch the whole table move with it.",
    )

with right:
    method_name = st.selectbox("Method", list(cn.METHODS), index=3)
    st.caption(
        {
            "Otsu + components": "Threshold, then label connected components. The "
            "naive baseline, and it must undercount: two touching coins are one "
            "component. The size of that undercount measures how much of the "
            "problem is separation rather than thresholding.",
            "Adaptive + components": "A local threshold instead of a global one — "
            "robust to uneven lighting, and still blind to touching.",
            "Watershed (global seed)": "The OpenCV tutorial's version. Seeds are "
            "pixels above a fraction of the GLOBAL distance maximum, which is a "
            "sensible rule only if every object is the same size and already "
            "separated. Turn off illumination flattening to see it return 1 coin.",
            "Watershed (local maxima)": "Seeds are per-object local maxima of the "
            "distance transform. No global threshold anywhere, so a merged blob "
            "elsewhere in the image cannot suppress this coin's seed.",
            "Hough circles": "A shape PRIOR, not a region method. It fits circles, "
            "so it separates touching coins effortlessly and would find nothing at "
            "all on objects that are not round. Here that restriction is exactly "
            "what makes it the only method that measures every coin plausibly.",
        }.get(method_name, "")
    )
    min_distance = st.slider(
        "Seed neighbourhood radius (px)", 4, 30, 12,
        help="Local-maxima seeding only. A peak must dominate a disc this big.",
    )

st.divider()

truth = None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a photo above, or switch to the coins plate.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(img.shape[:2]) > 1000:
        s = 1000 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    img = shared_io.sample("coins")
    truth = cn.TRUE_COIN_COUNT


def run(name: str):
    fn = cn.METHODS[name]
    if name.startswith("Watershed (local"):
        return timeit(lambda: fn(img, min_distance=min_distance, flatten=flatten), runs=1, warmup=0)
    if name.startswith("Watershed (global"):
        return timeit(lambda: fn(img, flatten=flatten), runs=1, warmup=0)
    return timeit(lambda: fn(img), runs=1, warmup=0)


labels, timing = run(method_name)
props = cn.region_properties(labels)
mm_per_px = cn.calibrate_mm_per_px(props, reference_mm)
mm = cn.measure_mm(props, mm_per_px)


def overlay(lab, pr):
    rng = np.random.default_rng(0)
    colour = np.zeros_like(img)
    for i in range(1, int(lab.max()) + 1):
        colour[lab == i] = rng.integers(70, 255, 3)
    blend = cv2.addWeighted(img, 0.55, colour, 0.45, 0)
    for j, p in enumerate(sorted(pr, key=lambda q: q["centroid"][1]), start=1):
        cx, cy = map(int, p["centroid"])
        cv2.putText(blend, str(j), (cx - 8, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return blend


cols = st.columns(3)
cols[0].image(img, caption="1 · Input", width="stretch")
cols[1].image(cn._foreground_mask(img, flatten=flatten), caption="2 · Foreground mask", width="stretch")
cols[2].image(overlay(labels, props), caption=f"3 · {method_name} — {len(props)} objects", width="stretch")


st.subheader("Measurements")
m = st.columns(5)
m[0].metric("Objects found", len(props), delta=(None if truth is None else f"{len(props) - truth:+d} vs {truth}"))
m[1].metric("Time", f"{timing.median_ms:.1f} ms")
m[2].metric("Scale", f"{mm_per_px:.4f} mm/px")
if mm:
    floor = cn.PLAUSIBLE_MIN_FRACTION * reference_mm
    implausible = sum(1 for d in mm if d < floor)
    m[3].metric("Diameter range", f"{min(mm):.1f}–{max(mm):.1f} mm")
    m[4].metric(
        "Implausible regions", implausible,
        help=f"Regions measuring under {floor:.1f} mm — too small to be an object "
             "next to the reference, so a broken region rather than a small coin.",
    )

    if truth is not None and len(props) == truth and implausible > 0:
        st.warning(
            f"**The count is exactly right and {implausible} of the measurements are not.** "
            f"`{method_name}` found all {truth} coins, and the smallest region it "
            f"produced implies a **{min(mm):.2f} mm** coin against a {reference_mm:.2f} mm "
            "reference. A watershed boundary can squeeze a basin to a fraction of "
            "its coin without losing the count — so \"it counted correctly\" is not "
            "evidence that it segmented correctly. Try `Hough circles`, which fits "
            "a shape and therefore cannot produce a sliver."
        )
    elif truth is not None and len(props) == truth and implausible == 0:
        st.success(
            f"All {truth} coins found, and every measurement is plausible "
            f"({min(mm):.1f}–{max(mm):.1f} mm, CV {np.std(mm) / np.mean(mm):.3f})."
        )
    elif truth is not None:
        st.error(
            f"Counted **{len(props)}** against a hand-counted truth of **{truth}**. "
            + (
                "Touching coins merge into one connected component — that is the "
                "entire reason watershed exists."
                if "components" in method_name
                else "Check the foreground mask: if a band of background was "
                "classified as foreground, it has merged with the objects touching it."
            )
        )
else:
    st.info("No objects found.")

if truth is None:
    st.caption(
        "There is no ground-truth count for an image you supplied, so the count "
        "is reported without an error. The millimetre column is only as good as "
        "the reference you entered — see the calibration tab."
    )

st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4, t5 = st.tabs(
    ["Per-object measurements", "Method comparison", "The seeding ablation",
     "The tutorial's knob", "Calibration sensitivity"]
)

with t1:
    st.caption(
        "The output a measuring tool actually produces. Sorted by size, so a "
        "sliver stands out at the bottom."
    )
    rows = sorted(
        (
            {
                "rank": i,
                "area_px": p["area_px"],
                "diameter_px": round(p["diameter_px"], 2),
                "diameter_mm": round(p["diameter_px"] * mm_per_px, 2),
            }
            for i, p in enumerate(sorted(props, key=lambda q: -q["area_px"]), start=1)
        ),
        key=lambda r: r["rank"],
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if mm:
        fig = ui.histogram_figure(
            {method_name: np.asarray(mm)},
            bins=20, value_range=(0, max(mm) * 1.15),
            xlabel="measured diameter (mm)",
            title="Measured diameter distribution",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)

with t2:
    st.caption("Every method on **this** image, counting *and* measuring.")
    rows = []
    for name in cn.METHODS:
        lab, tm = run(name)
        pr = cn.region_properties(lab)
        scale = cn.calibrate_mm_per_px(pr, reference_mm)
        d = cn.measure_mm(pr, scale)
        row = {
            "method": name,
            "count": len(pr),
            "cv": round(float(np.std(d) / np.mean(d)), 4) if d else None,
            "smallest_mm": round(float(np.min(d)), 2) if d else None,
            "implausible": int(sum(1 for v in d if v < cn.PLAUSIBLE_MIN_FRACTION * reference_mm)),
            "ms": round(tm.median_ms, 1),
        }
        if truth is not None:
            row["count_error"] = abs(len(pr) - truth)
        rows.append(row)
    columns = [
        ("Count", "count", True),
        ("Smallest (mm)", "smallest_mm", True),
        ("Diameter CV", "cv", False),
        ("Implausible", "implausible", False),
        ("Time (ms)", "ms", False),
    ]
    if truth is not None:
        columns.insert(1, ("Count error", "count_error", False))
    styler, frame = ui.comparison_table(rows, columns)
    st.dataframe(styler, width="stretch")
    st.caption(
        "The count column and the implausible column disagree, and that "
        "disagreement is the project: several methods find the right *number* of "
        "objects, and fewer of them produce the right *regions*."
    )
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="coin_methods.csv",
        mime="text/csv",
    )

with t3:
    st.caption(
        "Two independent decisions — how the mask is made, and how the seeds are "
        "chosen — crossed. Computed live on this image."
    )
    rows = []
    for seed_name, fn in (
        ("Global fraction of dist.max()", cn.segment_watershed_global_seed),
        ("Local maxima", cn.segment_watershed),
    ):
        rows.append(
            {
                "seeding": seed_name,
                "plain": cn.count_coins(fn(img, flatten=False)),
                "tophat": cn.count_coins(fn(img, flatten=True)),
            }
        )
    styler, _ = ui.comparison_table(
        rows,
        [("Plain Otsu mask", "plain", True), ("Top-hat then Otsu", "tophat", True)],
        row_key="seeding",
    )
    st.dataframe(styler, width="stretch")
    st.markdown(
        "The interesting cell is **top-left**. With the same broken mask, one "
        "seeding rule collapses and the other does not — because a threshold "
        "taken as a fraction of the *global* distance maximum is a threshold set "
        "by the largest blob in the image, and a lighting artefact is enough to "
        "create one."
    )
    c = st.columns(2)
    c[0].image(cn._foreground_mask(img, flatten=False), caption="plain Otsu mask", width="stretch")
    c[1].image(cn._foreground_mask(img, flatten=True), caption="top-hat then Otsu", width="stretch")

with t4:
    st.caption(
        "The tutorial's `fg_ratio` swept against the local-maxima count, which "
        "has no such knob."
    )
    ratios = (0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8)
    local = cn.count_coins(cn.segment_watershed(img, min_distance=min_distance, flatten=flatten))
    series = {
        "Global seed": [
            cn.count_coins(cn.segment_watershed_global_seed(img, fg_ratio=r, flatten=flatten))
            for r in ratios
        ],
        "Local maxima": [local] * len(ratios),
    }
    fig = ui.lines_figure(
        list(ratios), series,
        xlabel="fg_ratio (fraction of the GLOBAL distance maximum)",
        ylabel="objects counted",
        title=("The knob has no safe setting"
               + (f"; the true count is {truth}" if truth else "")),
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t5:
    st.caption(
        "Every millimetre in this app rests on one number you typed. This is what "
        "happens when that number is wrong."
    )
    errs = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
    base = float(np.mean(mm)) if mm else 0.0
    measured = []
    for pct in errs:
        scale = cn.calibrate_mm_per_px(props, reference_mm * (1 + pct / 100))
        d = cn.measure_mm(props, scale)
        measured.append(100.0 * (float(np.mean(d)) - base) / base if base else 0.0)
    fig = ui.lines_figure(
        list(errs), {"error in every measured diameter (%)": measured},
        xlabel="error in the assumed reference (%)",
        ylabel="error in every measurement (%)",
        title="Calibration error propagates 1:1, undiminished and unsignalled",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown(
        "The line is `y = x`. A single multiplication cannot attenuate an error, "
        "and nothing downstream can detect one: the measurements stay perfectly "
        "self-consistent with each other, they are just all wrong by the same "
        "factor. **The uncertainty on every number this tool prints is the "
        "uncertainty on the reference**, and that is not visible in the output."
    )

st.divider()
with st.expander("See every method on this image"):
    panels, caps = [img], ["Input"]
    for name in cn.METHODS:
        lab, _ = run(name)
        pr = cn.region_properties(lab)
        panels.append(overlay(lab, pr))
        caps.append(f"{name} — {len(pr)}")
    for start in range(0, len(panels), 3):
        row = st.columns(3)
        for col, im, cap in zip(row, panels[start : start + 3], caps[start : start + 3]):
            col.image(im, caption=cap, width="stretch")

with st.expander("How the seeds are found"):
    mask = cn._foreground_mask(img, flatten=flatten)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    k = 2 * int(min_distance) + 1
    peaks = (dist >= cv2.dilate(dist, np.ones((k, k), np.float32)) - 1e-6) & (dist >= 10.0)
    vis = img.copy()
    ys, xs = np.nonzero(peaks)
    for x, y in zip(xs, ys):
        cv2.circle(vis, (int(x), int(y)), 4, (255, 40, 40), -1)
    c = st.columns(3)
    c[0].image(mask, caption="foreground mask", width="stretch")
    c[1].image(
        (dist / max(float(dist.max()), 1e-6) * 255).astype(np.uint8),
        caption="distance transform", width="stretch",
    )
    c[2].image(vis, caption=f"{int(peaks.sum())} local maxima = seeds", width="stretch")
    st.markdown(
        "The distance transform's value at a pixel **is** the distance to the "
        "nearest background pixel, so at a coin's centre it is that coin's radius "
        "— and it peaks once per coin even where two coins touch. Everything "
        "here turns on finding those peaks *locally*, one per object, rather than "
        "by comparing them all against a single global number."
    )
