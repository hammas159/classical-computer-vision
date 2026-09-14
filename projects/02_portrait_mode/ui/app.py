"""Streamlit UI for project 02 — portrait mode.

    streamlit run ui/app.py

Upload a portrait or generate one, pick a matting method and a bokeh kernel, and
see the background blur applied. On generated scenes the true alpha matte is
known, so the app reports the **actual IoU, hair recall and halo error** instead
of only showing a picture.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

import portrait_mode as pm  # noqa: E402
from shared import synth  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_uint8  # noqa: E402
from shared.metrics import dice, iou  # noqa: E402

st.set_page_config(
    page_title="Portrait Mode — Classical CV", layout="wide", initial_sidebar_state="collapsed"
)

st.title("Portrait mode")
st.caption(
    "Cut out the subject, blur the background — six classical matting methods, "
    "four aperture shapes, no depth sensor and no neural network."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Generated portrait (known matte)", "Upload your own photo"],
        horizontal=True,
        help="A generated scene has an exact alpha matte, so the error is measurable.",
    )
    uploaded, bg_name, seed = None, pm.BACKGROUNDS[0], 0
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Portrait photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        c1, c2 = st.columns(2)
        bg_name = c1.selectbox("Background", list(pm.BACKGROUNDS), index=1)
        seed = c2.slider("Scene seed", 0, 40, 0)

with right:
    c1, c2 = st.columns(2)
    matte_name = c1.selectbox("Matting method", list(pm.MATTES), index=2)
    bokeh_name = c2.selectbox("Bokeh kernel", list(pm.BOKEH_KERNELS), index=2)
    c3, c4 = st.columns(2)
    radius = c3.slider("Blur radius (px)", 3, 35, 15)
    comp_name = c4.selectbox("Compositing", list(pm.COMPOSITORS), index=1)

st.divider()

scene = None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a portrait in the panel above, or switch to a generated scene.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(image.shape[:2]) > 1200:
        sc = 1200 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
else:
    scene = synth.portrait_scene(background=bg_name, seed=seed)
    image = scene.image


(mask, out), timing = timeit(
    lambda: pm.portrait(image, matte_name, bokeh_name, radius, comp_name), runs=1, warmup=0
)

c1, c2, c3 = st.columns(3)
c1.image(image, caption="1 · Input", width="stretch")

if mask is None:
    c2.warning("No subject found.")
    st.error(
        f"**{matte_name}** found no face in this image, so there is nothing to cut out. "
        "GrabCut (centre rect) and Skin colour do not need a face — try one of those."
    )
    st.stop()

c2.image(mask, caption="2 · Matte", width="stretch")
c3.image(out, caption="3 · Portrait mode", width="stretch")

st.subheader("Measurements")
m = st.columns(4)
m[0].metric("Pipeline time", f"{timing.median_ms:.0f} ms")

kernel = pm.BOKEH_KERNELS[bokeh_name](radius)
prof = pm.highlight_profile(kernel)
m[1].metric(
    "Bokeh peak / mean",
    f"{prof['peak_to_mean']:.2f}",
    help="1.00 is a flat disc, like a real aperture. Higher means a soft Gaussian smudge.",
)

if scene is not None:
    m[2].metric("Matte IoU", f"{iou(mask, scene.mask):.3f}")
    hair_recall = float((mask[scene.hair > 0] > 0).mean()) if (scene.hair > 0).any() else 0.0
    m[3].metric(
        "Hair recovered",
        f"{hair_recall * 100:.1f}%",
        help="Hair is about 2.5% of the subject, so whole-image IoU barely notices losing it.",
    )

    ideal = pm.composite_reference(scene, kernel)
    band = pm.halo_ring(scene.mask, 12) > 0
    ideal_out = pm.COMPOSITORS[comp_name](image, scene.mask, kernel)
    halo = float(
        np.abs(ideal_out.astype(float) - ideal.astype(float)).mean(axis=-1)[band].mean()
    )
    st.markdown(
        f"With the **true** matte, `{comp_name}` leaves a halo error of **{halo:.2f}** "
        f"(0-255 scale) in the 12 px ring outside the subject. "
        f"Matte Dice: **{dice(mask, scene.mask):.3f}**."
    )
    if comp_name.startswith("Naive"):
        st.warning(
            "Naive compositing blurs across the subject boundary, smearing subject "
            "colour outward. Switch to the masked normalised convolution and watch "
            "the halo number fall."
        )
else:
    m[2].metric("Matte IoU", "unknown")
    m[3].metric("Hair recovered", "unknown")
    st.caption("Accuracy needs a known matte, which an uploaded photo does not have.")

if matte_name.startswith(("Haar + GrabCut", "GrabCut")):
    st.info(
        f"GrabCut is seeded at {pm.GRABCUT_SEED} for reproducibility. Unseeded, the "
        "same image can return anything from a near-perfect matte to a failed one — "
        "see the seed-stability table in the README."
    )

with st.expander("Compare all six matting methods on this image"):
    cols = st.columns(3)
    for i, (name, fn) in enumerate(pm.MATTES.items()):
        got, t = timeit(lambda f=fn: f(image), runs=1, warmup=0)
        label = f"{name} — {t.median_ms:.0f} ms"
        if got is None:
            cols[i % 3].warning(f"{name}: no subject found")
            continue
        if scene is not None:
            label += f" — IoU {iou(got, scene.mask):.3f}"
        cols[i % 3].image(got, caption=label, width="stretch")

with st.expander("Compare all four bokeh kernels"):
    point = np.zeros((radius * 6 + 1, radius * 6 + 1), np.float32)
    point[point.shape[0] // 2, point.shape[1] // 2] = 1.0
    cols = st.columns(4)
    for i, (name, make) in enumerate(pm.BOKEH_KERNELS.items()):
        k = make(radius)
        rendered = cv2.filter2D(point, -1, k)
        p = pm.highlight_profile(k)
        # Normalise to uint8 rather than passing a float array: dividing by the
        # max can land a hair above 1.0 in float32, and Streamlit rejects that
        # with "Data is outside [0.0, 1.0] and clamp is not set".
        cols[i].image(
            to_uint8(np.clip(rendered / max(rendered.max(), 1e-9), 0.0, 1.0)),
            caption=f"{name} — peak/mean {p['peak_to_mean']:.2f}, rim {p['edge_sharpness']:.2f}",
            width="stretch",
        )
