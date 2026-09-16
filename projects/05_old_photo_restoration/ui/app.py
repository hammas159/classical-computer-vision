"""Streamlit UI for project 05 — old photo restoration.

    streamlit run ui/app.py

Age a photograph by a known amount — scratches, fading, or both — or upload a
real damaged scan, and watch the two halves of restoration run separately.
Because the damage is generated, the exact damaged-pixel mask is known, so the
app can answer the question a restoration demo usually cannot: **how much of the
result came from the inpainting method, and how much from knowing where the
damage was?**
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

import restoration as rs  # noqa: E402
from shared import io as shared_io, theme  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import iou, psnr, rms_contrast, ssim  # noqa: E402

st.set_page_config(
    page_title="Old photo restoration — Classical CV",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#: Project 05's identity: "Sepia print". Each of the 58 apps has its own
#: palette, face and corner radius, so a screenshot says which project it
#: came from before the title is read. See shared/theme.py.
PALETTE = theme.apply(5)

st.title("Old photo restoration")
st.caption(
    "Two unrelated jobs wear the same name. Inpainting replaces pixels that are "
    "**missing**; fade correction fixes pixels that are **present but wrong**. "
    "Neither touches the other's problem. No training, no network, no GPU."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Age a clean photo", "Upload your own damaged scan"],
        horizontal=True,
        help=(
            "Generating the damage is what makes the true mask known, and therefore "
            "what makes 'how much did detection cost you' a measurable question."
        ),
    )
    uploaded, image_name = None, "astronaut"
    thickness, blotches, do_fade = 3, 6, True
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Damaged photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        image_name = st.selectbox("Image", list(rs.IMAGES), index=0)
        c1, c2 = st.columns(2)
        thickness = c1.slider(
            "Scratch width (px)", 1, 40, 3,
            help="The project's main variable. Every method here is interpolation, "
                 "and interpolation stops working when the gap is wide.",
        )
        blotches = c2.slider("Blotches", 0, 20, 6)
        do_fade = st.checkbox(
            "Also fade the print (dye loss, yellowing, flattened contrast)", value=True
        )

with right:
    method_name = st.selectbox("Inpainting method", list(rs.METHODS), index=0)
    st.caption(
        {
            "Telea (fast marching)": "Fills inward from the damage boundary, each "
            "unknown pixel a distance- and gradient-weighted average of known "
            "neighbours. Fast and smooth — a strength on thin scratches, a weakness "
            "on wide holes where 'smooth' means 'blurred'.",
            "Navier-Stokes": "Continues level lines (isophotes) across the gap using "
            "the mathematics of incompressible flow. Best on thin damage, where an "
            "edge genuinely does continue across.",
            "Iterative masked mean": "20 lines. Peel the damage one ring at a time, "
            "averaging over known pixels only. The 'did you need any of this?' "
            "control — and at wide damage it stops losing.",
            "Harmonic diffusion": "Repeatedly blur, keeping known pixels pinned: the "
            "Laplace equation by Jacobi iteration. Always converges to something "
            "smooth, which is why it collapses first as the damage widens.",
        }.get(method_name, "")
    )
    detector_name = st.selectbox("Damage detector", list(rs.DETECTORS), index=2, key="detector")
    fade_name = st.selectbox(
        "Fade correction", list(rs.FADE_METHODS), index=len(rs.FADE_METHODS) - 1, key="fade"
    )

st.divider()

truth, true_mask = None, None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a damaged photo above, or switch to a generated one.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    damaged = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(damaged.shape[:2]) > 1200:
        s = 1200 / max(damaged.shape[:2])
        damaged = cv2.resize(damaged, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    truth = shared_io.sample(image_name)
    base = synth.fade_photo(truth, seed=0) if do_fade else truth
    damaged, true_mask = synth.add_scratches(
        base, thickness=thickness, blotches=blotches, seed=0
    )

detected = rs.DETECTORS[detector_name](damaged)
inpainted, timing = timeit(
    lambda: rs.METHODS[method_name](damaged, detected), runs=1, warmup=0
)
restored = rs.FADE_METHODS[fade_name](inpainted)

cols = st.columns(5 if truth is not None else 4)
i = 0
if truth is not None:
    cols[i].image(truth, caption="1 · Original (ground truth)", width="stretch")
    i += 1
cols[i].image(damaged, caption=f"{i + 1} · Damaged input", width="stretch")
cols[i + 1].image(detected, caption=f"{i + 2} · Detected damage ({detector_name})", width="stretch")
cols[i + 2].image(inpainted, caption=f"{i + 3} · After {method_name}", width="stretch")
cols[i + 3].image(restored, caption=f"{i + 4} · After {fade_name}", width="stretch")


st.subheader("Measurements")
m = st.columns(5)
m[0].metric("Inpaint time", f"{timing.median_ms:.1f} ms")
m[1].metric("Flagged as damaged", f"{float((detected > 0).mean()):.1%}")
m[2].metric(
    "Chroma",
    f"{rs.saturation_of(restored):.1f}",
    delta=f"{rs.saturation_of(restored) - rs.saturation_of(damaged):+.1f}",
    help="Mean LAB chroma. Fading pulls this down; a good correction puts it back "
         "where it was — no further.",
)
m[3].metric(
    "RMS contrast",
    f"{rms_contrast(restored):.4f}",
    delta=f"{rms_contrast(restored) - rms_contrast(damaged):+.4f}",
)

if truth is not None:
    ceiling = rs.FADE_METHODS[fade_name](rs.METHODS[method_name](damaged, true_mask))
    p_in, p_out, p_ceiling = psnr(damaged, truth), psnr(restored, truth), psnr(ceiling, truth)
    m[4].metric("PSNR", f"{p_out:.2f} dB", delta=f"{p_out - p_in:+.2f} dB vs input")

    mask_iou = iou(detected, true_mask)
    p, t = detected > 0, true_mask > 0
    recall = float((p & t).sum()) / max(float(t.sum()), 1.0)
    precision = float((p & t).sum()) / max(float(p.sum()), 1.0)

    st.markdown(
        f"The damaged input scores **{p_in:.2f} dB**. This pipeline reaches "
        f"**{p_out:.2f} dB** (SSIM {ssim(restored, truth):.3f}). Handed the *true* "
        f"mask instead of `{detector_name}`'s, the identical pipeline reaches "
        f"**{p_ceiling:.2f} dB** — so **{p_ceiling - p_out:.2f} dB of this result is "
        f"lost to detection, not to the inpainting method.** The detector found "
        f"**{recall:.0%}** of the damage at **{precision:.0%}** precision "
        f"(IoU {mask_iou:.3f})."
    )
    if recall < 0.85:
        st.warning(
            f"`{detector_name}` missed **{1 - recall:.0%}** of the damaged pixels. "
            "Those pixels are never handed to the inpainter, so no choice of method "
            "can recover them. Precision costs far less: a healthy pixel that is "
            "wrongly flagged gets replaced by an average of its healthy neighbours, "
            "which is approximately itself."
        )
    if method_name == "Harmonic diffusion" and thickness >= 15:
        st.warning(
            f"At {thickness} px, harmonic diffusion is the worst method in the set. "
            "It solves the Laplace equation, whose solution over a wide hole is a "
            "smooth surface with no texture at all — plausible, and wrong."
        )
else:
    m[4].metric("PSNR", "unknown")
    st.caption(
        "PSNR, SSIM and detector IoU all need the undamaged original, which a real "
        "scan does not have. Chroma and contrast need no reference, so those are "
        "still shown — and the pictures are still the honest evidence."
    )


st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4, t5 = st.tabs(
    [
        "Damage width sweep",
        "The scratch, as numbers",
        "Method comparison",
        "Detector comparison",
        "Tone distribution",
    ]
)

with t1:
    st.caption(
        "The experiment that separates the methods. Every method here is "
        "interpolation across a gap; the only question is how wide a gap it "
        "survives. All four are within 2 dB at 1 px."
    )
    if truth is None:
        st.info("This sweep needs the ground truth, so it runs on the sample images.")
    widths = (1, 3, 5, 9, 15, 25, 40)
    base_img = truth if truth is not None else shared_io.sample("astronaut")
    series = {name: [] for name in rs.METHODS}
    for wpx in widths:
        d, mk = synth.add_scratches(base_img, thickness=wpx, blotches=blotches, seed=0)
        for name, fn in rs.METHODS.items():
            series[name].append(round(rs._damage_only(fn(d, mk), base_img, mk), 3))
    fig = ui.lines_figure(
        list(widths),
        series,
        xlabel="scratch width (px)",
        ylabel="PSNR over the damaged pixels only (dB)",
        title="Width, not method, is what decides the result",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t2:
    st.caption(
        "A 12x12 patch straddling a scratch, read as numbers. The damaged column is "
        "a block of values that have nothing to do with their neighbours; every "
        "method is guessing what belongs there."
    )
    mask_for_patch = true_mask if true_mask is not None else detected
    ys, xs = np.nonzero(mask_for_patch)
    if len(ys):
        cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
        size = 12
        y0 = max(0, min(damaged.shape[0] - size, cy - size // 2))
        x0 = max(0, min(damaged.shape[1] - size, cx - size // 2))
        crop = lambda a: to_gray(a)[y0 : y0 + size, x0 : x0 + size]  # noqa: E731
        panels = []
        if truth is not None:
            panels.append(("Original", crop(truth)))
        panels += [("Damaged", crop(damaged)), (method_name, crop(inpainted))]
        gc = st.columns(len(panels))
        for col, (label, patch) in zip(gc, panels):
            col.markdown(f"**{label}**")
            col.dataframe(ui.pixel_grid(patch), width="stretch")
    else:
        st.info("No damage was detected in this image, so there is no patch to show.")

with t3:
    st.caption("Every inpainting method on **this** image, with **this** mask.")
    mask_used = true_mask if true_mask is not None else detected
    rows = []
    for name, fn in rs.METHODS.items():
        o, tm = timeit(lambda f=fn: f(damaged, mask_used), runs=1, warmup=0)
        row = {"method": name, "contrast": round(rms_contrast(o), 4), "ms": round(tm.median_ms, 1)}
        if truth is not None:
            row["psnr"] = round(psnr(o, truth), 3)
            row["damage_psnr"] = round(rs._damage_only(o, truth, mask_used), 3)
            row["ssim"] = round(ssim(o, truth), 4)
        rows.append(row)
    columns = [("RMS contrast", "contrast", True), ("Time (ms)", "ms", False)]
    if truth is not None:
        columns = [
            ("PSNR whole (dB)", "psnr", True),
            ("PSNR on damage (dB)", "damage_psnr", True),
            ("SSIM", "ssim", True),
        ] + columns
    styler, frame = ui.comparison_table(rows, columns)
    st.dataframe(styler, width="stretch")
    if truth is not None:
        st.caption(
            "Watch the two PSNR columns disagree in *range*. Whole-image PSNR barely "
            "moves because 93% of the photograph was never damaged; damage-only PSNR "
            "is the column that is actually about inpainting."
        )
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="restoration_methods.csv",
        mime="text/csv",
    )

with t4:
    st.caption(
        "Three ways to find the damage, and what each costs downstream. Precision "
        "and recall are separated because their costs are not: a missed scratch stays "
        "in the picture, while a falsely flagged pixel is replaced by its own healthy "
        "neighbours, which is approximately itself."
    )
    drows = []
    for name, fn in rs.DETECTORS.items():
        pred, dtm = timeit(lambda f=fn: f(damaged), runs=1, warmup=0)
        row = {
            "detector": name,
            "flagged": round(float((pred > 0).mean()), 4),
            "ms": round(dtm.median_ms, 1),
        }
        if true_mask is not None:
            p, t = pred > 0, true_mask > 0
            hit = float((p & t).sum())
            row["iou"] = round(iou(pred, true_mask), 4)
            row["precision"] = round(hit / max(float(p.sum()), 1.0), 4)
            row["recall"] = round(hit / max(float(t.sum()), 1.0), 4)
            row["psnr"] = round(psnr(rs.METHODS[method_name](damaged, pred), truth), 3)
        drows.append(row)
    dcolumns = [("Flagged fraction", "flagged", False), ("Time (ms)", "ms", False)]
    if true_mask is not None:
        dcolumns = [
            ("Mask IoU", "iou", True),
            ("Precision", "precision", True),
            ("Recall", "recall", True),
            ("Restored PSNR (dB)", "psnr", True),
        ] + dcolumns
    dstyler, dframe = ui.comparison_table(drows, dcolumns, row_key="detector")
    st.dataframe(dstyler, width="stretch")
    if true_mask is not None:
        st.caption(
            "If the highest-IoU detector is not the highest-PSNR one, that is the "
            "asymmetry above showing up as a number."
        )
    st.download_button(
        "Download this matrix as CSV",
        dframe.to_csv().encode("utf-8"),
        file_name="restoration_detectors.csv",
        mime="text/csv",
    )

with t5:
    series = {"damaged input": to_gray(damaged), f"after {fade_name}": to_gray(restored)}
    if truth is not None:
        series = {"original": to_gray(truth), **series}
    fig = ui.histogram_figure(
        series,
        bins=192,
        title="Fading compresses the tonal range and lifts the black point",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(
        "A faded print occupies a narrow band in the middle of the range. The "
        "per-channel stretch is nothing more than putting the two ends back where "
        "they belong — which is why it beats CLAHE, a method that works locally on "
        "a problem that is global."
    )


st.divider()
with st.expander("See every inpainting method on this image"):
    mask_used = true_mask if true_mask is not None else detected
    panels = [damaged] + [rs.METHODS[n](damaged, mask_used) for n in rs.METHODS]
    caps = ["Damaged input"] + list(rs.METHODS)
    for start in range(0, len(panels), 3):
        row = st.columns(3)
        for col, img, cap in zip(row, panels[start : start + 3], caps[start : start + 3]):
            col.image(img, caption=cap, width="stretch")

with st.expander("See every fade correction on this image"):
    panels = [rs.FADE_METHODS[n](inpainted) for n in rs.FADE_METHODS]
    caps = list(rs.FADE_METHODS)
    for start in range(0, len(panels), 3):
        row = st.columns(3)
        for col, img, cap in zip(row, panels[start : start + 3], caps[start : start + 3]):
            col.image(img, caption=f"{cap} — chroma {rs.saturation_of(img):.1f}", width="stretch")
