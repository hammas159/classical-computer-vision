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

import matplotlib.pyplot as plt  # noqa: E402

import portrait_mode as pm  # noqa: E402
from shared import io as io_shared, theme  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_uint8  # noqa: E402
from shared.metrics import dice, iou  # noqa: E402

st.set_page_config(
    page_title="Portrait Mode — Classical CV", layout="wide", initial_sidebar_state="collapsed",
    page_icon="🖼️",
)

#: Project 02's identity: "Studio portrait". Each of the 58 apps has its own
#: palette, face and corner radius, so a screenshot says which project it
#: came from before the title is read. See shared/theme.py.
PALETTE = theme.apply(2)

st.title("Portrait mode")
st.caption(
    "Cut out the subject, blur the background — six classical matting methods, "
    "four aperture shapes, no depth sensor and no neural network."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Reference photo (annotated matte)", "Upload your own photo"],
        horizontal=True,
        help=(
            "The reference photo is a real photograph with a matte annotated once "
            "and frozen, so results can be scored against it. An uploaded photo "
            "has no annotation, so it is shown rather than scored."
        ),
    )
    uploaded, bg_name, seed = None, pm.BACKGROUNDS[0], 0
    if source.startswith("Reference"):
        st.caption(
            "A real photograph — real person, real hair, real depth, a real crowd "
            "behind them. The matte was annotated once with GrabCut plus cleanup "
            "and frozen, which is why GrabCut-based methods score well here."
        )
    elif source.startswith("Upload"):
        uploaded = st.file_uploader("Portrait photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        c1, c2 = st.columns(2)
        bg_name = c1.selectbox("Background", list(pm.BACKGROUNDS), index=1)
        seed = c2.slider("Scene seed", 0, 40, 0)

with right:
    c1, c2 = st.columns(2)
    # Default to GrabCut (centre rect) rather than Haar + GrabCut: on a real
    # photograph the face box seeds GrabCut too tightly and the matte loses the
    # arms and legs entirely. The centre rect makes no assumption about a face
    # being found at all, which is the safer default for an arbitrary image.
    matte_name = c1.selectbox("Matting method", list(pm.MATTES), index=3)
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
    scene = synth.portrait_scene()
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
    fine_recall = float((mask[scene.fine > 0] > 0).mean()) if (scene.fine > 0).any() else 0.0
    m[3].metric(
        "Fine detail recovered",
        f"{fine_recall * 100:.1f}%",
        help=(
            "Thin structure -- outstretched limbs and the boundary band. It is a "
            "small fraction of the subject, so whole-image IoU barely notices "
            "losing all of it."
        ),
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
    m[3].metric("Fine detail recovered", "unknown")
    st.caption(
        "IoU and hair recall need a known matte, and a real photograph has none — "
        "nobody labelled which of its pixels are subject. No number is invented here."
    )

if matte_name.startswith(("Haar + GrabCut", "GrabCut")):
    st.info(
        f"GrabCut is seeded at {pm.GRABCUT_SEED} for reproducibility. Unseeded, the "
        "same image can return anything from a near-perfect matte to a failed one — "
        "see the seed-stability table in the README."
    )

# --------------------------------------------------------------------------- #
# distributions and matrices
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Distributions and matrices")

r_tab, h_tab, k_tab, c_tab = st.tabs(
    ["Region matrix", "Halo distribution", "Kernel matrix", "Confusion matrix"]
)

with r_tab:
    if scene is None:
        st.info("Region scores need a known matte — switch to a generated portrait.")
    else:
        st.caption(
            "Every method on **this** image, split by region. Hair is ~2.5% of the "
            "subject's pixels, so whole-image IoU cannot see a method losing all of it."
        )
        rows = []
        for name, fn in pm.MATTES.items():
            got, t = timeit(lambda f=fn: f(image), runs=1, warmup=0)
            if got is None:
                continue
            rr = pm.region_recall(got, scene)
            rows.append(
                {"method": name, **rr, "iou": round(iou(got, scene.mask), 4),
                 "ms": round(t.median_ms, 1)}
            )
        styler, frame = ui.comparison_table(
            rows,
            [
                ("IoU", "iou", True),
                ("Body", "body", True),
                ("Fine", "fine", True),
                ("Background", "background", True),
                ("Time (ms)", "ms", False),
            ],
        )
        st.dataframe(styler, width="stretch")
        st.caption(
            "Green is better per column. Look for the row that wins IoU and loses Fine — "
            "that disagreement is this project's central finding."
        )
        st.download_button(
            "Download this matrix as CSV",
            frame.to_csv().encode("utf-8"),
            file_name="matte_region_matrix.csv",
            mime="text/csv",
        )

with h_tab:
    if scene is None:
        st.info("Halo error needs the clean background plate — use a generated portrait.")
    else:
        st.caption(
            "Blurring across the subject boundary smears subject colour outward. "
            "This is the error in the 12 px ring outside the subject, against the "
            "ideal composite built from the true background plate."
        )
        ideal_c = pm.composite_reference(scene, kernel)
        band = pm.halo_ring(scene.mask, 12) > 0
        series = {}
        for cname, cfn in pm.COMPOSITORS.items():
            out_c = cfn(image, scene.mask, kernel)
            series[cname] = pm.halo_error_map(out_c, ideal_c)[band]
        fig = ui.histogram_figure(
            series,
            bins=80,
            value_range=(0, 60),
            xlabel="absolute error vs the ideal composite (0-255)",
            title="Halo error distribution in the ring outside the subject",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        means = {k: float(np.mean(v)) for k, v in series.items()}
        for k, v in means.items():
            st.markdown(f"- `{k}` — mean **{v:.2f}**")

with k_tab:
    st.caption(
        "The aperture as a matrix. A real lens maps a point of light to the shape of "
        "its aperture — a **flat disc**, not a Gaussian bump."
    )
    show_r = st.slider("Kernel radius to display", 3, 9, 5, key="kernel_display_radius")
    kcols = st.columns(len(pm.BOKEH_KERNELS))
    for i, (kname, make) in enumerate(pm.BOKEH_KERNELS.items()):
        k = make(show_r)
        prof = pm.highlight_profile(k)
        with kcols[i]:
            st.markdown(f"**{kname}**")
            st.caption(f"peak/mean {prof['peak_to_mean']:.2f} · rim {prof['edge_sharpness']:.2f}")
            st.dataframe(
                ui.pixel_grid(k / k.max() * 100, cmap="magma", fmt="{:.0f}"),
                width="stretch",
            )

with c_tab:
    if scene is None:
        st.info("A confusion matrix needs a known matte — use a generated portrait.")
    else:
        st.caption("Rows are the truth, columns are what the method predicted.")
        pick = st.selectbox(
            "Matting method", list(pm.MATTES), index=2, key="confusion_matte_method"
        )
        got = pm.MATTES[pick](image)
        if got is None:
            st.warning(f"{pick} found no subject in this image.")
        else:
            cm = pm.matte_confusion(got, scene.mask)
            styler, counts, _ = ui.confusion_frame(
                cm, ["background", "subject"], ["background", "subject"]
            )
            cc1, cc2 = st.columns(2)
            cc1.markdown("**Row-normalised (recall per class)**")
            cc1.dataframe(styler, width="stretch")
            cc2.markdown("**Raw pixel counts**")
            cc2.dataframe(counts, width="stretch")
            total = int(counts.values.sum())
            correct = int(counts.values.trace())
            st.markdown(
                f"`{pick}` classified **{correct:,}** of **{total:,}** pixels correctly "
                f"({correct / total * 100:.2f}%). Background is the majority class by a "
                "wide margin, which is why that headline percentage stays high even when "
                "the whole boundary is wrong."
            )

st.divider()

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
