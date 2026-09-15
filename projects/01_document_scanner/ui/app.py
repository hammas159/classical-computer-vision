"""Streamlit UI for project 01 — the document scanner.

    streamlit run ui/app.py

Upload a photo of a page (or use a generated scene), pick a detector and a
binariser, and watch each stage of the pipeline. When the generated scene is
used the true corners are known, so the app reports the **actual corner error in
pixels** rather than just showing a picture that looks about right.
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

import document_scanner as ds  # noqa: E402
from shared import io as io_shared  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import corner_error, iou  # noqa: E402

st.set_page_config(
    page_title="Document Scanner — Classical CV",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#: Default lighting for the generated scene. It must be one of ILLUM_LEVELS —
#: st.select_slider raises ValueError for a value outside its options list.
DEFAULT_ILLUM = 0.5

st.title("Document scanner")
st.caption(
    "Find the page, rectify it, make it readable — six classical detectors, "
    "no training and no neural network anywhere."
)


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #

# Controls live in the main pane rather than the sidebar so that the whole app
# is visible in one screenshot and on a narrow screen.
ctrl_left, ctrl_right = st.columns([1, 1])

with ctrl_left:
    source = st.radio(
        "Image source",
        ["Real photograph", "Generated scene (known ground truth)", "Upload your own photo"],
        horizontal=True,
        help=(
            "A real photograph shows the pipeline working on an image nobody "
            "constructed for it -- but it has no ground truth, so no error can be "
            "reported. The generated scene has exact corner positions."
        ),
    )

    uploaded, real_name = None, "newspaper"
    seed, illum = 0, DEFAULT_ILLUM
    if source.startswith("Real"):
        doc_photos = [n for n in io_shared.real_photo_names() if n != "player"]
        real_name = st.selectbox(
            "Photograph", doc_photos,
            index=doc_photos.index("newspaper"),
            format_func=lambda n: f"{n} - {io_shared.REAL_PHOTOS[n][1]}",
        )
    elif source.startswith("Upload"):
        uploaded = st.file_uploader("Photo of a page", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        sc1, sc2 = st.columns(2)
        seed = sc1.slider("Scene seed", 0, 99, 0, help="Each seed is a different camera pose.")
        illum = sc2.select_slider(
            "Lighting",
            options=list(ds.ILLUM_LEVELS),
            value=DEFAULT_ILLUM,
            help="1.0 is flat studio light. Lower values cast a deeper shadow across the frame.",
        )

with ctrl_right:
    pc1, pc2 = st.columns(2)
    # Default to Canny + contour rather than Otsu + contour. On the generated
    # scenes Otsu is more accurate (1.11 px vs 2.70 px), but on a real photo of
    # a bright newspaper it thresholds the WHOLE FRAME as page and returns a
    # wedge across the image. Canny keys off the page border instead, which is
    # the more robust default for an arbitrary photograph.
    detector = pc1.selectbox("Page detector", list(ds.DETECTORS), index=0)
    binariser = pc2.selectbox("Binariser", list(ds.BINARISERS), index=3)
    use_persp = st.checkbox(
        "Recover aspect ratio from perspective",
        value=True,
        help="Off = the usual edge-length heuristic, which distorts the page.",
    )

st.divider()


truth_corners = None
truth_page = None

if source.startswith("Real"):
    image = io_shared.real_photo(real_name)
elif source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a photo in the sidebar, or switch to a generated scene.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # keep very large phone photos manageable without changing the geometry
    if max(image.shape[:2]) > 1600:
        scale = 1600 / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
else:
    image, truth_page, truth_corners = synth.document_scene(seed=seed, illum_min=illum)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

corners, rect, binary = None, None, None
(corners, rect, binary), timing = timeit(
    lambda: ds.scan(image, detector, binariser, use_perspective_aspect=use_persp),
    runs=3,
    warmup=1,
)

overlay = image.copy()
if truth_corners is not None:
    cv2.polylines(overlay, [np.int32(truth_corners)], True, (220, 40, 40), 3)
if corners is not None:
    cv2.polylines(overlay, [np.int32(corners)], True, (0, 200, 0), 3)
    for x, y in np.int32(corners):
        cv2.circle(overlay, (int(x), int(y)), 7, (0, 200, 0), -1)

c1, c2, c3, c4 = st.columns(4)
c1.image(image, caption="1 · Input", width="stretch")
c2.image(
    overlay,
    caption="2 · Detected page (green)" + (" vs truth (red)" if truth_corners is not None else ""),
    width="stretch",
)

if corners is None:
    c3.warning("No page found.")
    st.error(
        f"**{detector}** could not find a quadrilateral in this image. "
        "Try another detector — that difference is the whole point of the comparison."
    )
    st.stop()

c3.image(rect, caption="3 · Rectified", width="stretch")
c4.image(binary, caption="4 · Binarised", width="stretch")


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

st.subheader("Measurements")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Pipeline time", f"{timing.median_ms:.1f} ms", help="Median of 3 runs, warm-up discarded")
m2.metric("Output size", f"{rect.shape[1]} x {rect.shape[0]} px")

ratio = ds.aspect_from_perspective(corners, image.shape)
m3.metric(
    "Recovered w/h",
    "degenerate" if ratio is None else f"{ratio:.3f}",
    help="None means the view is near-affine, so the closed form does not apply.",
)

if truth_corners is not None:
    err = corner_error(corners, truth_corners)
    m4.metric("Corner error", f"{err:.2f} px", help="Mean distance to the true page corners")

    page_h, page_w = truth_page.shape[:2]
    truth_text = ds.text_mask(to_gray(truth_page))
    ideal = to_gray(ds.rectify(image, truth_corners, (page_w, page_h)))
    text_iou = iou(255 - ds.BINARISERS[binariser](ideal), truth_text)

    H, W = image.shape[:2]
    page_ratio = ds.page_illumination_ratio(truth_corners, (W, H), illum)

    st.markdown(
        f"Illumination ratio across this page: **{page_ratio:.2f}** "
        f"(a perfect global threshold exists only above {ds.INK_REFLECTANCE:.2f}). "
        f"Text IoU for **{binariser}**: **{text_iou:.3f}**."
    )
    if page_ratio < 0.5 and binariser == "Otsu (global)":
        st.warning(
            "Otsu is expected to fail here. Switch the binariser to Sauvola or "
            "Adaptive Gaussian and compare — the shadow is what breaks it."
        )
else:
    m4.metric("Corner error", "unknown", help="Ground truth exists only for generated scenes")
    st.caption(
        "Corner error needs ground truth, and a real photograph does not have one — "
        "nobody recorded where its corners truly are. Inventing a number here would "
        "be the exact dishonesty this repo exists to avoid, so none is shown. "
        "Switch to a generated scene to see the pipeline scored."
    )


# --------------------------------------------------------------------------- #
# compare every method on this image
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# distributions and matrices
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Distributions and matrices")

d_tab, m_tab, c_tab, x_tab = st.tabs(
    ["Pixel distribution", "Pixel matrix", "Comparison matrix", "Confusion matrix"]
)

gray_rect = to_gray(rect)
otsu_t = ds.otsu_threshold(gray_rect)

with d_tab:
    st.caption(
        "The distribution a thresholding method has to cut. Drag **Lighting** down "
        "and watch the paper spread across the axis until Otsu's cut lands inside it."
    )
    if truth_corners is not None:
        pop = ds.page_intensities(ideal, truth_text)
        _, oracle_t = ds.binarise_best_global(ideal, truth_text)
        fig = ui.histogram_figure(
            {"paper": pop["paper"], "ink": pop["ink"]},
            vlines={"Otsu picks": ds.otsu_threshold(ideal), "best possible": oracle_t},
            title="Ink and paper populations, with the chosen thresholds",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.markdown(
            f"Otsu chose **{ds.otsu_threshold(ideal)}**; the best possible global "
            f"threshold is **{oracle_t}**. A perfect global cut exists whenever the page "
            f"illumination ratio stays above **{ds.INK_REFLECTANCE:.2f}**."
        )
    else:
        fig = ui.histogram_figure(
            {"all pixels": gray_rect},
            vlines={"Otsu picks": otsu_t},
            title="Rectified page intensity distribution",
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Ink/paper populations need ground truth — use a generated scene.")

with m_tab:
    st.caption(
        "The image as a matrix. A 12x12 patch of the rectified page, before and after "
        "binarisation — the numbers are the argument."
    )
    size = 12
    h, w = gray_rect.shape
    y0 = max(0, min(h - size, h // 3))
    x0 = max(0, min(w - size, w // 4))
    before = gray_rect[y0 : y0 + size, x0 : x0 + size]
    after = binary[y0 : y0 + size, x0 : x0 + size]
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"**Before** — raw greys (Otsu's cut = {otsu_t})")
        st.dataframe(ui.pixel_grid(before), width="stretch")
    with g2:
        st.markdown(f"**After** — `{binariser}`, 0 = ink")
        st.dataframe(ui.pixel_grid(after), width="stretch")

with c_tab:
    st.caption("Every detector run on **this** image, scored on every metric at once.")
    live_rows = []
    for name, fn in ds.DETECTORS.items():
        got, t = timeit(lambda f=fn: f(image), runs=3, warmup=1)
        row = {"method": name, "time_ms": round(t.median_ms, 2)}
        if got is None:
            row.update({"corner_err_px": None, "area_iou": None, "aspect_wh": None})
        else:
            row["aspect_wh"] = ds.aspect_from_perspective(got, image.shape)
            if truth_corners is not None:
                row["corner_err_px"] = round(corner_error(got, truth_corners), 3)
                row["area_iou"] = round(
                    iou(ds.quad_mask(got, image.shape), ds.quad_mask(truth_corners, image.shape)), 4
                )
        live_rows.append(row)

    cols = [("Time (ms)", "time_ms", False), ("Recovered w/h", "aspect_wh", True)]
    if truth_corners is not None:
        cols = [
            ("Corner error (px)", "corner_err_px", False),
            ("Area IoU", "area_iou", True),
        ] + cols
    styler, frame = ui.comparison_table(live_rows, cols)
    st.dataframe(styler, width="stretch")
    st.caption("Green is better in each column; each column is scaled on its own.")
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="detector_comparison.csv",
        mime="text/csv",
    )

with x_tab:
    if truth_corners is None:
        st.info("A confusion matrix needs ground truth — switch to a generated scene.")
    else:
        st.caption(
            "Where the binariser's pixels actually went. Rows are the truth, columns "
            "are what the method predicted."
        )
        cm_rows = []
        for name, fn in ds.BINARISERS.items():
            cm = ds.text_confusion(fn(ideal), truth_text)
            cm_rows.append((name, cm))
        # an explicit key is required: Streamlit derives element IDs from the
        # widget type and its parameters, so a second selectbox labelled
        # "Binariser" collides with the one in the controls above
        pick = st.selectbox(
            "Binariser", [n for n, _ in cm_rows], index=3, key="confusion_binariser"
        )
        cm = dict(cm_rows)[pick]
        styler, counts, _ = ui.confusion_frame(cm, ["paper", "ink"], ["paper", "ink"])
        cc1, cc2 = st.columns([1, 1])
        cc1.markdown("**Row-normalised (recall per class)**")
        cc1.dataframe(styler, width="stretch")
        cc2.markdown("**Raw pixel counts**")
        cc2.dataframe(counts, width="stretch")
        total = int(counts.values.sum())
        correct = int(counts.values.trace())
        st.markdown(
            f"`{pick}` put **{correct:,}** of **{total:,}** pixels in the right class "
            f"({correct / total * 100:.2f}%). Note how little that number moves even when "
            "the ink is destroyed — paper is the overwhelming majority class."
        )

st.divider()

with st.expander("Compare all six detectors on this image"):
    cols = st.columns(3)
    for i, (name, fn) in enumerate(ds.DETECTORS.items()):
        got, t = timeit(lambda f=fn: f(image), runs=3, warmup=1)
        panel = image.copy()
        if truth_corners is not None:
            cv2.polylines(panel, [np.int32(truth_corners)], True, (220, 40, 40), 3)
        label = f"{name} — {t.median_ms:.1f} ms"
        if got is None:
            label += " — FAILED"
        else:
            cv2.polylines(panel, [np.int32(got)], True, (0, 200, 0), 3)
            if truth_corners is not None:
                label += f" — {corner_error(got, truth_corners):.1f} px"
        cols[i % 3].image(panel, caption=label, width="stretch")

with st.expander("Compare all four binarisers on this page"):
    gray = to_gray(rect)
    cols = st.columns(4)
    for i, (name, fn) in enumerate(ds.BINARISERS.items()):
        out, t = timeit(lambda f=fn: f(gray), runs=3, warmup=1)
        cols[i].image(out, caption=f"{name} — {t.median_ms:.2f} ms", width="stretch")
