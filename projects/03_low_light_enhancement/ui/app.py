"""Streamlit UI for project 03 — low-light enhancement.

    streamlit run ui/app.py

Darken an image by a known gamma, or upload your own dark photo, and watch eight
methods try to bring it back. When the darkening is generated the original is
known exactly, so the app reports **how far each method is from the ceiling**
rather than just showing a brighter picture.
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

import low_light as ll  # noqa: E402
from shared import synth, ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared import io as shared_io  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import entropy, estimate_noise_sigma, psnr, rms_contrast, ssim  # noqa: E402

st.set_page_config(
    page_title="Low-light Enhancement — Classical CV",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Low-light enhancement")
st.caption(
    "Eight classical methods brighten a dark photo — and an oracle shows how much "
    "of the original was still there to recover. No training, no network, no GPU."
)


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source",
        ["Darken a known image", "Upload your own dark photo"],
        horizontal=True,
        help="Darkening a known image is what makes the error measurable.",
    )
    uploaded, image_name, gamma, noise = None, ll.IMAGES[0], 3.0, 4.0
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Dark photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        c1, c2 = st.columns(2)
        image_name = c1.selectbox("Image", list(ll.IMAGES), index=0)
        gamma = c2.slider(
            "Darkness (gamma)", 1.2, 5.0, 3.0, 0.1,
            help="Higher is darker. The tone range is crushed before any method runs.",
        )
        noise = st.slider(
            "Read noise sigma", 0.0, 12.0, 4.0, 0.5,
            help="Sensor noise added after darkening — this is what brightening multiplies.",
        )

with right:
    method_name = st.selectbox("Enhancement method", list(ll.METHODS), index=1)
    st.caption(
        {
            "Gamma 1/2.2 (fixed)": "A fixed curve. Knows nothing about the image — "
            "exactly right at gamma 2.2 and progressively wrong elsewhere.",
            "Gamma (auto-estimated)": "Estimates the exponent by assuming a "
            "well-exposed photo averages mid-grey. The assumption is the method.",
            "Histogram equalisation": "Flattens the histogram. A statistical goal, "
            "not a perceptual one.",
            "CLAHE": "Local equalisation with a clip limit that caps noise gain.",
            "Single-scale Retinex": "Estimates reflectance, not exposure — so raw "
            "PSNR scores it on the wrong thing.",
            "Multi-scale Retinex": "Retinex at three scales, combined.",
            "MSRCR": "Retinex plus a colour restoration term.",
            "LIME": "Estimates an illumination map, then divides it out.",
        }.get(method_name, "")
    )

st.divider()

truth = None
if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a dark photo above, or switch to a generated one.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    dark = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(dark.shape[:2]) > 1200:
        s = 1200 / max(dark.shape[:2])
        dark = cv2.resize(dark, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    truth = shared_io.sample(image_name)
    dark = synth.low_light(truth, gamma=gamma, noise_sigma=noise, seed=0)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

out, timing = timeit(lambda: ll.METHODS[method_name](dark), runs=1, warmup=0)

cols = st.columns(4 if truth is not None else 2)
if truth is not None:
    cols[0].image(truth, caption="1 · Original (ground truth)", width="stretch")
    cols[1].image(dark, caption=f"2 · Darkened, gamma {gamma:.1f}", width="stretch")
    cols[2].image(out, caption=f"3 · {method_name}", width="stretch")
    cols[3].image(
        ll.enhance_oracle(dark, gamma),
        caption="4 · Oracle (knows the true gamma)",
        width="stretch",
    )
else:
    cols[0].image(dark, caption="1 · Input", width="stretch")
    cols[1].image(out, caption=f"2 · {method_name}", width="stretch")


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #

st.subheader("Measurements")
m = st.columns(5)
m[0].metric("Time", f"{timing.median_ms:.1f} ms")
m[1].metric("Entropy", f"{entropy(out):.2f} bits", help="Information content, no reference needed")
m[2].metric("RMS contrast", f"{rms_contrast(out):.3f}")

noise_before = estimate_noise_sigma(dark)
noise_after = estimate_noise_sigma(out)
m[3].metric(
    "Noise amplification",
    f"{noise_after / max(noise_before, 1e-6):.2f}x",
    help="Brightening is a multiplication, so it multiplies the shadow noise too.",
)

if truth is not None:
    oracle_out = ll.enhance_oracle(dark, gamma)
    method_psnr = psnr(out, truth)
    oracle_psnr = psnr(oracle_out, truth)
    m[4].metric(
        "PSNR",
        f"{method_psnr:.2f} dB",
        delta=f"{method_psnr - oracle_psnr:+.2f} dB vs oracle",
        delta_color="normal",
    )

    levels = ll.quantisation_ceiling(gamma)
    st.markdown(
        f"At gamma **{gamma:.1f}**, only **{levels} of 256** tone levels survive the "
        f"darkening. The oracle — which applies the exact inverse — reaches "
        f"**{oracle_psnr:.2f} dB**, and that is the ceiling. "
        f"`{method_name}` is **{oracle_psnr - method_psnr:.2f} dB** below it. "
        f"SSIM **{ssim(out, truth):.3f}**."
    )

    if method_name.startswith("Gamma (auto"):
        estimated = 1.0 / ll.estimate_gamma(dark)
        true_mean = float(truth.astype(np.float64).mean() / 255.0)
        gap = true_mean - ll.AUTO_TARGET_BRIGHTNESS
        st.info(
            f"The method estimated gamma **{estimated:.2f}** against a true "
            f"**{gamma:.1f}** — an error of **{(estimated / gamma - 1) * 100:+.1f}%**. "
            f"This image's true mean brightness is **{true_mean:.3f}**, a gap of "
            f"**{gap:+.3f}** from the {ll.AUTO_TARGET_BRIGHTNESS} the estimator assumes. "
            "The error tracks that gap, not the darkness."
        )
    if method_name.endswith("Retinex") or method_name == "MSRCR":
        matched = psnr(ll.match_exposure(out, truth), truth)
        st.warning(
            f"Retinex estimates **reflectance, not exposure**, so raw PSNR scores it "
            f"on a global brightness offset rather than on recovered detail. After "
            f"matching exposure it scores **{matched:.2f} dB** instead of "
            f"**{method_psnr:.2f} dB** — a {matched - method_psnr:+.2f} dB difference "
            "that is entirely the metric, not the method."
        )
else:
    m[4].metric("PSNR", "unknown")
    st.caption(
        "PSNR needs the original, which an uploaded photo does not have. "
        "Switch to a darkened image to see every method scored."
    )


# --------------------------------------------------------------------------- #
# distributions and matrices
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4 = st.tabs(
    ["Tone distribution", "Pixel matrix", "Comparison matrix", "The ceiling"]
)

with t1:
    st.caption(
        "Darkening crushes the tone range toward zero. Inverting it stretches the "
        "survivors back apart — leaving **gaps**, because the levels in between "
        "were destroyed by 8-bit rounding and cannot be invented back."
    )
    series = {"darkened": to_gray(dark), f"{method_name}": to_gray(out)}
    if truth is not None:
        series = {"original": to_gray(truth), **series,
                  "oracle": to_gray(ll.enhance_oracle(dark, gamma))}
    fig = ui.histogram_figure(series, bins=256, title="Tone distribution, before and after")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t2:
    st.caption("A 12x12 patch as raw numbers. Count the distinct values in each panel.")
    g_dark = to_gray(dark)
    h, w = g_dark.shape
    y0, x0, size = h // 2, w // 2, 12
    panels = []
    if truth is not None:
        p = to_gray(truth)[y0 : y0 + size, x0 : x0 + size]
        panels.append((f"Original — {len(np.unique(p))} distinct", p))
    p = g_dark[y0 : y0 + size, x0 : x0 + size]
    panels.append((f"Darkened — {len(np.unique(p))} distinct", p))
    p = to_gray(out)[y0 : y0 + size, x0 : x0 + size]
    panels.append((f"{method_name} — {len(np.unique(p))} distinct", p))

    grid_cols = st.columns(len(panels))
    for col, (label, patch) in zip(grid_cols, panels):
        col.markdown(f"**{label}**")
        col.dataframe(ui.pixel_grid(patch), width="stretch")
    st.caption(
        "The darkened panel has far fewer distinct values than the original. That "
        "loss happened **before** any method ran, and no method can undo it."
    )

with t3:
    st.caption("Every method on **this** image, scored on every metric at once.")
    rows = []
    for name, fn in ll.METHODS.items():
        o, t = timeit(lambda f=fn: f(dark), runs=1, warmup=0)
        row = {
            "method": name,
            "entropy": round(entropy(o), 3),
            "contrast": round(rms_contrast(o), 4),
            "noise_x": round(estimate_noise_sigma(o) / max(noise_before, 1e-6), 2),
            "ms": round(t.median_ms, 1),
        }
        if truth is not None:
            row["psnr"] = round(psnr(o, truth), 3)
            row["ssim"] = round(ssim(o, truth), 4)
            row["psnr_matched"] = round(psnr(ll.match_exposure(o, truth), truth), 3)
        rows.append(row)
    if truth is not None:
        o = ll.enhance_oracle(dark, gamma)
        rows.append({
            "method": ll.ORACLE_NAME, "entropy": round(entropy(o), 3),
            "contrast": round(rms_contrast(o), 4),
            "noise_x": round(estimate_noise_sigma(o) / max(noise_before, 1e-6), 2),
            "ms": 0.0, "psnr": round(psnr(o, truth), 3),
            "ssim": round(ssim(o, truth), 4),
            "psnr_matched": round(psnr(ll.match_exposure(o, truth), truth), 3),
        })

    columns = [("Entropy", "entropy", True), ("Contrast", "contrast", True),
               ("Noise x", "noise_x", False), ("Time (ms)", "ms", False)]
    if truth is not None:
        columns = [("PSNR (dB)", "psnr", True), ("SSIM", "ssim", True),
                   ("PSNR matched", "psnr_matched", True)] + columns
    styler, frame = ui.comparison_table(rows, columns)
    st.dataframe(styler, width="stretch")
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="low_light_comparison.csv",
        mime="text/csv",
    )

with t4:
    st.caption(
        "The ceiling is arithmetic, not an algorithm. Darkening by gamma maps the "
        "256 input levels onto fewer outputs, and the collapsed ones are gone "
        "before any method sees the image."
    )
    gammas = np.linspace(1.0, 6.0, 51)
    levels = [ll.quantisation_ceiling(float(g)) for g in gammas]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(gammas, levels, color="#2f6f9f", linewidth=2)
    ax.axvline(gamma, color="#b03030", linestyle="--", linewidth=1.5)
    ax.annotate(
        f"you are here: {ll.quantisation_ceiling(gamma)} levels",
        xy=(gamma, ll.quantisation_ceiling(gamma)),
        xytext=(gamma + 0.3, ll.quantisation_ceiling(gamma) + 25),
        color="#b03030", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#b03030"),
    )
    ax.set_xlabel("gamma (higher = darker)", fontsize=9)
    ax.set_ylabel("distinct 8-bit levels surviving", fontsize=9)
    ax.set_ylim(0, 260)
    ax.grid(alpha=0.2, linestyle=":")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown(
        f"At gamma **{gamma:.1f}**, **{ll.quantisation_ceiling(gamma)} of 256** levels "
        f"survive — **{256 - ll.quantisation_ceiling(gamma)} are destroyed** by rounding "
        "the darkened image to 8 bits. That is why even a perfect inverse cannot "
        "reach infinite PSNR."
    )


# --------------------------------------------------------------------------- #
# compare everything on this image
# --------------------------------------------------------------------------- #

st.divider()

with st.expander("See all eight methods on this image"):
    panels = [dark] + [ll.METHODS[n](dark) for n in ll.METHODS]
    captions = ["Darkened input"] + list(ll.METHODS)
    if truth is not None:
        panels.append(ll.enhance_oracle(dark, gamma))
        captions.append(ll.ORACLE_NAME)
    for start in range(0, len(panels), 5):
        row = st.columns(5)
        for col, img, cap in zip(row, panels[start : start + 5], captions[start : start + 5]):
            col.image(img, caption=cap, width="stretch")
