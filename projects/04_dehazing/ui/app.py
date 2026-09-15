"""Streamlit UI for project 04 — dehazing.

    streamlit run ui/app.py

Add a known amount of haze to a clear image, or upload your own hazy photo, and
watch five methods try to remove it. When the haze is generated the transmission
map is known exactly, so the app reports **how well the physics was recovered**
rather than just showing a picture with more contrast.
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

import dehazing as dz  # noqa: E402
from shared import io as shared_io  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import psnr, rms_contrast, ssim  # noqa: E402

st.set_page_config(
    page_title="Dehazing — Classical CV", layout="wide", initial_sidebar_state="collapsed"
)

st.title("Dehazing")
st.caption(
    "Haze is not a loss of light — it is an additive veil with a known physical "
    "model. Five classical methods try to invert it. No training, no network, no GPU."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Add known haze", "Upload your own hazy photo"],
        horizontal=True,
        help=(
            "Generating the haze is what makes the transmission map known, and "
            "therefore what makes 'did it recover the physics' a measurable question."
        ),
    )
    uploaded, image_name, beta = None, "rocket", 1.4
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Hazy photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        c1, c2 = st.columns(2)
        image_name = c1.selectbox("Image", list(dz.IMAGES), index=0)
        beta = c2.slider(
            "Haze density (beta)", 0.2, 3.0, 1.4, 0.1,
            help="t = exp(-beta*d). Higher means a thicker veil and a lower minimum transmission.",
        )

with right:
    method_name = st.selectbox("Method", list(dz.METHODS), index=1)
    st.caption(
        {
            "Dark channel prior": "Physical. Estimates transmission from the dark "
            "channel, then inverts the scattering model. The patch minimum makes "
            "the map blocky — look for square edges in the sky.",
            "DCP + guided refine": "The same, with the blocky map pushed back onto "
            "the image's own edges by a guided filter. This is the full method.",
            "CLAHE (contrast only)": "Models nothing. It raises local contrast, "
            "which is most of the *visible* effect of dehazing — included so you "
            "can see that looking clearer is not the same as being dehazed.",
            "Multi-scale Retinex": "An illumination model applied to a scattering "
            "problem. Retinex assumes image = illumination x reflectance, which is "
            "multiplicative; haze is additive. It cannot represent the degradation.",
            "Gamma curve (control)": "A plain power curve — the 'did you need any "
            "of this?' control.",
        }.get(method_name, "")
    )

st.divider()

truth, true_t = None, None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a hazy photo above, or switch to generated haze.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    hazy = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(hazy.shape[:2]) > 1200:
        s = 1200 / max(hazy.shape[:2])
        hazy = cv2.resize(hazy, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    truth = dz.load_scene(image_name)
    hazy, true_t = synth.add_haze(truth, beta=beta, airlight=dz.AIRLIGHT)


out, timing = timeit(lambda: dz.METHODS[method_name](hazy), runs=1, warmup=0)

cols = st.columns(4 if truth is not None else 2)
if truth is not None:
    cols[0].image(truth, caption="1 · Original (ground truth)", width="stretch")
    cols[1].image(hazy, caption=f"2 · Hazy, beta {beta:.1f}", width="stretch")
    cols[2].image(out, caption=f"3 · {method_name}", width="stretch")
    cols[3].image(
        dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t),
        caption="4 · Oracle (knows the true transmission)",
        width="stretch",
    )
else:
    cols[0].image(hazy, caption="1 · Input", width="stretch")
    cols[1].image(out, caption=f"2 · {method_name}", width="stretch")


st.subheader("Measurements")
m = st.columns(5)
m[0].metric("Time", f"{timing.median_ms:.1f} ms")
m[1].metric("RMS contrast", f"{rms_contrast(out):.4f}", delta=f"{rms_contrast(out) - rms_contrast(hazy):+.4f}")

a_est = dz.estimate_airlight(hazy)
m[2].metric(
    "Estimated airlight",
    f"{float(np.mean(a_est)):.3f}",
    delta=(None if truth is None else f"{float(np.mean(a_est)) - dz.AIRLIGHT:+.3f} vs true"),
    delta_color="off",
)

if truth is not None:
    oracle_out = dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t)
    p_method, p_oracle, p_hazy = psnr(out, truth), psnr(oracle_out, truth), psnr(hazy, truth)
    m[3].metric("PSNR", f"{p_method:.2f} dB", delta=f"{p_method - p_hazy:+.2f} dB vs hazy")

    t_est = dz.refine_transmission_guided(hazy, dz.transmission_dcp(hazy, a_est))
    t_mae = dz.transmission_error(t_est, true_t)
    m[4].metric("Transmission MAE", f"{t_mae:.4f}", help="How well the physics itself was recovered")

    st.markdown(
        f"The hazy input scores **{p_hazy:.2f} dB**. `{method_name}` reaches "
        f"**{p_method:.2f} dB** (SSIM {ssim(out, truth):.3f}). The oracle — handed "
        f"the *true* transmission map — reaches **{p_oracle:.2f} dB**, and that is "
        f"the ceiling. Minimum transmission at this density is "
        f"**{np.exp(-beta):.3f}**."
    )
    if method_name.startswith("CLAHE"):
        st.warning(
            f"CLAHE raised contrast to **{rms_contrast(out):.4f}** — higher than "
            f"`DCP + guided refine` achieves — while scoring **{p_method:.2f} dB** "
            f"against its **{psnr(dz.dehaze_dcp_refined(hazy), truth):.2f} dB**. "
            "**Higher contrast is not dehazing.** It models no transmission at all; "
            "it simply stretches what the veil left behind."
        )
    if method_name.endswith("Retinex"):
        st.warning(
            "Retinex assumes image = illumination x reflectance — a **multiplicative** "
            "model. Haze is **additive**: `I = J*t + A*(1-t)`. Retinex cannot "
            "represent this degradation, and the score shows it."
        )
else:
    m[3].metric("PSNR", "unknown")
    m[4].metric("Transmission MAE", "unknown")
    st.caption(
        "PSNR and transmission error need the clear image and the true transmission "
        "map, which a real hazy photo does not have. Contrast and the airlight "
        "estimate need no reference, so those are still shown."
    )


st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4 = st.tabs(["Transmission map", "The veil, as numbers", "Comparison matrix", "Tone distribution"])

with t1:
    st.caption(
        "The transmission map is where the physics lives. A method that produces a "
        "good-looking image from a wrong transmission map got there by accident."
    )
    t_blocky = dz.transmission_dcp(hazy, a_est)
    t_refined = dz.refine_transmission_guided(hazy, t_blocky)
    panels = [
        ("Dark channel", dz.dark_channel(hazy)),
        (f"DCP estimate (patch {dz.DEFAULT_PATCH})", t_blocky),
        ("Guided refined", t_refined),
    ]
    if true_t is not None:
        panels.insert(0, ("TRUE transmission", true_t))
    pc = st.columns(len(panels))
    for col, (label, img) in zip(pc, panels):
        col.image(np.clip(img, 0, 1), caption=label, width="stretch", clamp=True)
    if true_t is not None:
        st.markdown(
            f"Transmission MAE — blocky **{dz.transmission_error(t_blocky, true_t):.4f}**, "
            f"refined **{dz.transmission_error(t_refined, true_t):.4f}**. The guided "
            "filter's job is not to be more accurate on average; it is to put the "
            "depth edges on the *object* boundaries instead of on patch boundaries."
        )

with t2:
    st.caption(
        "Haze pulls every pixel toward the airlight. Read the numbers: the hazy "
        "column is compressed into a narrow band near A, and dehazing pushes it "
        "back apart."
    )
    g = to_gray(hazy)
    h, w = g.shape
    y0, x0, size = h // 4, w // 2, 12
    panels = []
    if truth is not None:
        panels.append(("Original", to_gray(truth)[y0 : y0 + size, x0 : x0 + size]))
    panels.append((f"Hazy (A={dz.AIRLIGHT})", g[y0 : y0 + size, x0 : x0 + size]))
    panels.append((method_name, to_gray(out)[y0 : y0 + size, x0 : x0 + size]))
    gc = st.columns(len(panels))
    for col, (label, patch) in zip(gc, panels):
        col.markdown(f"**{label}**")
        col.dataframe(ui.pixel_grid(patch), width="stretch")

with t3:
    st.caption("Every method on **this** image, scored on every metric at once.")
    rows = []
    for name, fn in dz.METHODS.items():
        o, t = timeit(lambda f=fn: f(hazy), runs=1, warmup=0)
        row = {"method": name, "contrast": round(rms_contrast(o), 4), "ms": round(t.median_ms, 1)}
        if truth is not None:
            row["psnr"] = round(psnr(o, truth), 3)
            row["ssim"] = round(ssim(o, truth), 4)
        rows.append(row)
    if truth is not None:
        o = dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t)
        rows.append({
            "method": dz.ORACLE_NAME, "contrast": round(rms_contrast(o), 4), "ms": 0.0,
            "psnr": round(psnr(o, truth), 3), "ssim": round(ssim(o, truth), 4),
        })
    columns = [("RMS contrast", "contrast", True), ("Time (ms)", "ms", False)]
    if truth is not None:
        columns = [("PSNR (dB)", "psnr", True), ("SSIM", "ssim", True)] + columns
    styler, frame = ui.comparison_table(rows, columns)
    st.dataframe(styler, width="stretch")
    st.caption(
        "Watch the contrast column disagree with PSNR — the method with the most "
        "contrast is not the one that recovered the scene."
    )
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="dehazing_comparison.csv",
        mime="text/csv",
    )

with t4:
    series = {"hazy": to_gray(hazy), method_name: to_gray(out)}
    if truth is not None:
        series = {"original": to_gray(truth), **series}
    fig = ui.histogram_figure(
        series, bins=192,
        vlines={"airlight": dz.AIRLIGHT * 255} if truth is not None else None,
        title="Haze compresses the tone range toward the airlight",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)


st.divider()
with st.expander("See all five methods on this image"):
    panels = [hazy] + [dz.METHODS[n](hazy) for n in dz.METHODS]
    caps = ["Hazy input"] + list(dz.METHODS)
    if truth is not None:
        panels.append(dz.dehaze_oracle(hazy, dz.AIRLIGHT, true_t))
        caps.append(dz.ORACLE_NAME)
    for start in range(0, len(panels), 4):
        row = st.columns(4)
        for col, img, cap in zip(row, panels[start : start + 4], caps[start : start + 4]):
            col.image(img, caption=cap, width="stretch")
