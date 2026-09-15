"""Streamlit UI for project 07 — copy-move forgery detection.

    streamlit run ui/app.py

Forge an image yourself — copy a patch, rotate it, rescale it, paste it — or
upload a photo you suspect. Because the forgery is generated, the exact
duplicated mask is known, so the app can answer the question a tampering demo
usually cannot: **not "did it find something" but "how much of what it found was
real, and what did it accuse this photograph of?"**
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import forgery as fg  # noqa: E402
from shared import io as shared_io  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import iou  # noqa: E402

st.set_page_config(
    page_title="Copy-move forgery — Classical CV",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Copy-move forgery detection")
st.caption(
    "The pasted region comes from the same photograph, so it has the same noise, "
    "the same lighting and the same compression history as everything around it. "
    "There is no statistical trace to find — only a duplicate. "
    "No training, no network, no GPU."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Forge an image", "Upload a photo to check"],
        horizontal=True,
        help=(
            "Generating the forgery is what makes the duplicated mask known, and "
            "therefore what makes 'how much of this is real' a measurable question."
        ),
    )
    uploaded, image_name = None, "astronaut"
    size, angle, scale = 96, 0.0, 1.0
    if source.startswith("Forge"):
        image_name = st.selectbox("Image", list(fg.IMAGES), index=0)
        c1, c2 = st.columns(2)
        size = c1.slider("Pasted size (px)", 24, 192, 96, 8)
        angle = c2.slider("Rotate the copy (deg)", 0.0, 90.0, 0.0, 1.0)
        scale = st.slider(
            "Rescale the copy", 0.7, 1.5, 1.0, 0.05,
            help="Block matching survives exactly 1.00. Try 0.95.",
        )
    else:
        uploaded = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "bmp", "webp"])

with right:
    method_name = st.selectbox("Detector", list(fg.METHODS), index=1)
    st.caption(
        {
            "Block matching": "Describe every 16x16 block, sort the descriptors so "
            "near-duplicates land next to each other, and keep offsets with 400+ "
            "votes. Near-perfect on an exact copy. Scores 0.000 at 2 degrees of "
            "rotation — the quadrant means it uses are not rotation invariant, and "
            "there is no partial credit.",
            "SIFT + similarity verify": "SIFT matched against itself, then RANSAC "
            "fits a translation+rotation+scale and the whole image is checked "
            "against it. The only method here that survives an arbitrary angle — "
            "and the one that accuses untampered photographs.",
            "ORB + similarity verify": "The same pipeline on binary descriptors. "
            "ORB is rotation invariant but not scale invariant, so watch it hold "
            "up under the rotation slider and fall under the scale slider.",
            "SIFT + translation verify": "Identical SIFT keypoints and identical "
            "matches to the row above — the ONLY difference is that the verifier "
            "assumes a pure shift. That one assumption is worth +0.12 IoU at "
            "0 degrees and -0.50 at 90.",
            "SIFT blobs (no verify)": "The same matches again, painted as circles "
            "of a fixed radius. No verification at all. It never wins and never "
            "collapses, and it flags something on every untampered image tested.",
            "Predict nothing (control)": "Returns an empty mask. It scores 91.8% "
            "pixel accuracy, which is what that metric is worth here.",
        }.get(method_name, "")
    )

st.divider()

truth = None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a photo above, or switch to a forged one.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(image.shape[:2]) > 1000:
        s = 1000 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    clean = shared_io.sample(image_name)
    f = synth.copy_move_forgery(clean, size=size, angle_deg=angle, scale=scale, seed=0)
    image, truth = f.image, f.mask_both

pred, timing = timeit(lambda: fg.METHODS[method_name](image), runs=1, warmup=0)

cols = st.columns(4 if truth is not None else 2)
i = 0
if truth is not None:
    cols[0].image(clean, caption="1 · Original", width="stretch")
    i = 1
cols[i].image(image, caption=f"{i + 1} · {'Forged' if truth is not None else 'Input'}", width="stretch")
cols[i + 1].image(pred, caption=f"{i + 2} · {method_name}", width="stretch")
if truth is not None:
    cols[3].image(truth, caption="4 · Truth (both copies)", width="stretch")


st.subheader("Measurements")
m = st.columns(5)
m[0].metric("Time", f"{timing.median_ms:.0f} ms")
m[1].metric("Flagged", f"{float((pred > 0).mean()):.2%}")

fit = fg.describe_match(image)
m[2].metric("Matched pairs", fit["pairs"], help="SIFT keypoints matched against themselves")
m[3].metric(
    "Fitted rotation",
    "—" if fit["angle_deg"] is None else f"{fit['angle_deg']:.1f}°",
    delta=(None if truth is None else f"{angle:.0f}° applied"),
    delta_color="off",
)

if truth is not None:
    p, t = pred > 0, truth > 0
    hit = float((p & t).sum())
    precision = hit / max(float(p.sum()), 1.0)
    recall = hit / max(float(t.sum()), 1.0)
    m[4].metric("Mask IoU", f"{iou(pred, truth):.3f}")

    st.markdown(
        f"`{method_name}` found **{recall:.0%}** of the duplicated pixels at "
        f"**{precision:.0%}** precision (IoU {iou(pred, truth):.3f}). The truth "
        f"panel marks **both** copies, because after the paste they are identical "
        f"— nothing in the image says which one is the forgery, and scoring "
        f"against the pasted half alone would cap precision at 50% for every "
        f"method here."
    )
    if method_name == "Block matching" and (angle > 0 or scale != 1.0):
        st.error(
            f"Block matching scores **{iou(pred, truth):.3f}** here. Its descriptor "
            "is four quadrant means, which is not rotation or scale invariant — "
            "and unlike a method that degrades, it fails *completely*: the "
            "duplicated blocks simply no longer describe the same numbers, so no "
            "offset reaches 400 votes and the mask comes back empty. Set rotation "
            "to 0 and scale to 1.00 to see it reach 0.99."
        )
    if method_name == "SIFT + translation verify" and angle >= 2:
        st.warning(
            "This row shares every keypoint and every match with "
            "`SIFT + similarity verify`. The only difference is that its verifier "
            "shifts the image instead of warping it — and a rotated copy is not "
            "reachable by a shift. The descriptor was rotation invariant all "
            "along; the *hypothesis* was not."
        )
else:
    m[4].metric("Mask IoU", "unknown")
    st.warning(
        f"**{float((pred > 0).mean()):.1%} of this photograph was flagged.** There "
        "is no ground truth for an image you supplied, so this is not evidence of "
        "anything on its own — measured on untampered photographs, "
        "`SIFT + similarity verify` flags 6.3% of pixels on average and up to "
        "34% on a repeating texture. Check the fitted rotation and pair count "
        "above: a fit on hundreds of pairs at a distinctly non-zero angle is a "
        "finding; a handful of pairs at 0.0° is a flat sky."
    )


st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4, t5 = st.tabs(
    [
        "Rotation sweep",
        "Scale sweep",
        "The matches themselves",
        "Method comparison",
        "False alarms",
    ]
)

base = shared_io.sample(image_name if truth is not None else "astronaut")

with t1:
    st.caption(
        "The project's central experiment, recomputed live on this image. Watch "
        "block matching leave the chart between 0° and 2°."
    )
    angles = (0.0, 2.0, 5.0, 15.0, 30.0, 45.0, 90.0)
    names = [n for n in fg.METHODS if not n.startswith("Predict nothing")]
    series = {n: [] for n in names}
    for a in angles:
        ff = synth.copy_move_forgery(base, size=size, angle_deg=a, scale=scale, seed=0)
        for n in names:
            series[n].append(round(iou(fg.METHODS[n](ff.image), ff.mask_both), 4))
    fig = ui.lines_figure(
        list(angles), series, xlabel="paste rotation (degrees)", ylabel="mask IoU",
        title="The best method on an exact copy is the worst at every other angle",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t2:
    st.caption(
        "And rescaling. ORB is rotation invariant but not scale invariant, so the "
        "green line should behave differently here than it did above."
    )
    scales = (0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5)
    names = [n for n in fg.METHODS if not n.startswith("Predict nothing")]
    series = {n: [] for n in names}
    for s in scales:
        ff = synth.copy_move_forgery(base, size=size, angle_deg=angle, scale=s, seed=0)
        for n in names:
            series[n].append(round(iou(fg.METHODS[n](ff.image), ff.mask_both), 4))
    fig = ui.lines_figure(
        list(scales), series, xlabel="paste scale factor", ylabel="mask IoU",
        title="5% is enough to end block matching",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t3:
    st.caption(
        "What the keypoint stage actually produces: every surviving self-match, "
        "drawn as a line from one copy to the other. These are sparse and they "
        "land on corners — turning them into a region is a separate problem, and "
        "the one the verifier solves."
    )
    pairs = fg.self_matches(image, fg.make_sift(), cv2.NORM_L2, 0.6, 30)
    overlay = image.copy()
    for p1, p2 in pairs:
        cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)), (255, 60, 60), 1)
        cv2.circle(overlay, tuple(np.int32(p1)), 3, (60, 255, 60), -1)
    c = st.columns(3)
    c[0].image(overlay, caption=f"{len(pairs)} self-matches", width="stretch")
    c[1].image(fg.detect_sift(image), caption="Similarity verify", width="stretch")
    c[2].image(fg.detect_sift_keypoints_only(image), caption="Blobs, no verify", width="stretch")
    st.markdown(
        f"**{fit['pairs']} pairs, {fit['inliers']} of them consistent with a single "
        f"transform** of "
        + ("—" if fit["angle_deg"] is None else f"{fit['angle_deg']:.2f}° and {fit['scale']:.3f}×")
        + ". The inlier count is the number that matters: a large consensus on a "
        "non-trivial transform is what a copy-move looks like."
    )

with t4:
    st.caption("Every detector on **this** image.")
    rows = []
    for name, fn in fg.METHODS.items():
        o, tm = timeit(lambda g=fn: g(image), runs=1, warmup=0)
        row = {"method": name, "flagged": round(float((o > 0).mean()), 4), "ms": round(tm.median_ms, 1)}
        if truth is not None:
            p, t = o > 0, truth > 0
            hit = float((p & t).sum())
            row["iou"] = round(iou(o, truth), 4)
            row["precision"] = round(hit / max(float(p.sum()), 1.0), 4)
            row["recall"] = round(hit / max(float(t.sum()), 1.0), 4)
            row["pixel_acc"] = round(float((p == t).mean()), 4)
        rows.append(row)
    columns = [("Flagged fraction", "flagged", False), ("Time (ms)", "ms", False)]
    if truth is not None:
        columns = [
            ("Mask IoU", "iou", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("Pixel accuracy", "pixel_acc", True),
        ] + columns
    styler, frame = ui.comparison_table(rows, columns)
    st.dataframe(styler, width="stretch")
    if truth is not None:
        st.caption(
            "The control's pixel-accuracy cell is the point of that column: "
            "returning an empty mask is over 90% 'accurate' on a photograph where "
            "the forgery covers a few percent of pixels."
        )
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="forgery_methods.csv",
        mime="text/csv",
    )

with t5:
    st.caption(
        "The number a forgery detector is actually judged on. These photographs "
        "are **untampered** — anything flagged here is a false accusation."
    )
    fa_rows = []
    for name, fn in fg.METHODS.items():
        flagged = [float((fn(shared_io.sample(n)) > 0).mean()) for n in fg.IMAGES]
        fa_rows.append(
            {
                "method": name,
                "mean": round(float(np.mean(flagged)), 5),
                "worst": round(float(np.max(flagged)), 5),
                "accused": int(sum(1 for v in flagged if v > 0)),
            }
        )
    styler, frame = ui.comparison_table(
        fa_rows,
        [("Mean flagged", "mean", False), ("Worst image", "worst", False),
         ("Images accused (of 6)", "accused", False)],
    )
    st.dataframe(styler, width="stretch")
    st.caption(
        "Read this next to the rotation sweep. The methods that survive rotation "
        "are the methods that accuse clean photographs, and the one that never "
        "raises a false alarm is the one that fails at 2°."
    )

st.divider()
with st.expander("See every detector on this image"):
    panels = [image] + [fg.METHODS[n](image) for n in fg.METHODS]
    caps = ["Input"] + list(fg.METHODS)
    for start in range(0, len(panels), 4):
        row = st.columns(4)
        for col, img, cap in zip(row, panels[start : start + 4], caps[start : start + 4]):
            col.image(img, caption=cap, width="stretch")

with st.expander("Why a copy-move leaves no statistical trace"):
    series = {"whole image": to_gray(image).ravel()}
    if truth is not None:
        series["the duplicated pair"] = to_gray(image)[truth > 0]
    fig = ui.histogram_figure(series, bins=192, title="The pasted pixels came from this photograph")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown(
        "Splicing from a *different* photograph leaves traces a forensic tool can "
        "find: a different noise floor, a different JPEG quantisation, a different "
        "illuminant. A copy-move leaves none of them, because the pasted pixels "
        "were already in this image. The only evidence is the duplication itself."
    )
