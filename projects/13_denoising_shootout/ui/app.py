"""Streamlit UI for project 13 — denoising shootout.

    streamlit run ui/app.py

Add a known amount of a known kind of noise, and watch six filters try to remove
it. Because the noise is generated, sigma and density are exact and PSNR is
measured against the true clean image rather than against another algorithm's
output.
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

import denoising as dn  # noqa: E402
from shared import io as shared_io, theme  # noqa: E402
from shared import ui  # noqa: E402
from shared.bench import timeit  # noqa: E402
from shared.io import to_gray  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402

st.set_page_config(
    page_title="Denoising shootout — Classical CV", layout="wide",
    page_icon="🧹",
    initial_sidebar_state="collapsed",
)

#: Project 13's identity: "Auto 13". Each of the 58 apps has its own
#: palette, face and corner radius, so a screenshot says which project it
#: came from before the title is read. See shared/theme.py.
PALETTE = theme.apply(13)

st.title("Denoising shootout")
st.caption(
    "Six filters, three noise models, and one claim under test: there is no best "
    "denoiser. No training, no network, no GPU."
)

NOISE_LABELS = {k: lb for k, lv, lb in dn.NOISE_TYPES}
DEFAULT_LEVEL = {k: lv for k, lv, lb in dn.NOISE_TYPES}

left, right = st.columns([1, 1])

with left:
    source = st.radio("Image source", ["Sample image", "Upload your own"], horizontal=True)
    uploaded, image_name = None, "astronaut"
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        image_name = st.selectbox("Image", list(dn.IMAGES), index=0)
    kind = st.selectbox(
        "Noise model", list(NOISE_LABELS), index=0, format_func=lambda k: NOISE_LABELS[k]
    )
    if kind == "salt_pepper":
        level = st.slider("Density", 0.01, 0.30, 0.06, 0.01)
    elif kind == "gaussian":
        level = st.slider("Sigma (0–255 units)", 2.0, 60.0, 25.0, 1.0)
    else:
        level = st.slider("Lambda (lower = noisier)", 5.0, 120.0, 30.0, 5.0)

with right:
    method_name = st.selectbox("Filter", list(dn.METHODS), index=3)
    st.caption(
        {
            "Box": "Unweighted local average. The cheapest thing that works, and "
            "it blurs everything equally because it cannot tell an edge from noise.",
            "Gaussian": "The linear minimum-mean-squared-error filter for white "
            "noise *on a constant signal*. Real images are not constant, which is "
            "exactly why it blurs edges.",
            "Median": "An order statistic, not an average. An outlier cannot drag "
            "a median the way it drags a mean, so impulse noise is rejected "
            "outright — and on Gaussian noise, where every pixel is slightly "
            "wrong rather than a few being completely wrong, that advantage "
            "vanishes.",
            "Bilateral": "Gaussian in space *and* in intensity, so it averages "
            "only similar pixels. `sigma_color` has to be set relative to the "
            "noise level — it is the one parameter in this project whose tuning "
            "measurably transfers to new images (+4.1 dB).",
            "Non-local means": "Averages similar *patches* from anywhere in the "
            "image, on the observation that images repeat themselves. Preserves "
            "detail a local filter must destroy, at 1,830× the cost of a Gaussian "
            "blur.",
            "Wiener (adaptive)": "Shrinks each pixel toward its local mean in "
            "proportion to how much of the local variance is noise. Smooths hard "
            "where the image is flat and barely touches detail.",
            "Do nothing (control)": "Returns the input. Below about sigma 10 it "
            "beats most of the filters above, because their blurring costs more "
            "than the noise does.",
        }.get(method_name, "")
    )
    use_tuned = st.checkbox(
        "Use the measured-best parameter for this noise", value=True,
        help="Otherwise the library/default parameter is used. The difference is "
             "3.2 dB for the bilateral filter and zero for three others.",
    )

st.divider()

if source.startswith("Upload"):
    if uploaded is None:
        st.info("Upload a photo above, or switch to a sample image.")
        st.stop()
    buf = np.frombuffer(uploaded.getvalue(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("That file could not be decoded as an image.")
        st.stop()
    clean = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if max(clean.shape[:2]) > 800:
        s = 800 / max(clean.shape[:2])
        clean = cv2.resize(clean, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
else:
    clean = shared_io.sample(image_name)

noisy = dn.make_noisy(clean, kind, level, seed=0)


def run(name: str, image: np.ndarray):
    if use_tuned and not name.startswith("Do nothing"):
        return timeit(lambda: dn.tuned_call(name, image, kind), runs=1, warmup=0)
    return timeit(lambda: dn.METHODS[name](image), runs=1, warmup=0)


out, timing = run(method_name, noisy)

cols = st.columns(3)
cols[0].image(clean, caption="1 · Clean (the truth)", width="stretch")
cols[1].image(noisy, caption=f"2 · {NOISE_LABELS[kind]} — {psnr(noisy, clean):.2f} dB", width="stretch")
cols[2].image(out, caption=f"3 · {method_name} — {psnr(out, clean):.2f} dB", width="stretch")


st.subheader("Measurements")
m = st.columns(5)
p_in, p_out = psnr(noisy, clean), psnr(out, clean)
m[0].metric("PSNR", f"{p_out:.2f} dB", delta=f"{p_out - p_in:+.2f} dB vs noisy")
m[1].metric("SSIM", f"{ssim(out, clean):.4f}", delta=f"{ssim(out, clean) - ssim(noisy, clean):+.4f}")
m[2].metric("Time", f"{timing.median_ms:.2f} ms")
spec = dn.TUNED.get(kind, {}).get(method_name)
m[3].metric(
    "Parameter",
    f"{spec[0]}={spec[1]:g}" if (use_tuned and spec) else "default",
    help="The measured-best value for this noise type, found by grid search over six images.",
)

# the best alternative on this noise, so the page always names a better option
alternatives = {}
for name in dn.METHODS:
    if name == method_name:
        continue
    o, _ = run(name, noisy)
    alternatives[name] = psnr(o, clean)
best_alt = max(alternatives, key=alternatives.get)
m[4].metric(
    "Best alternative",
    f"{alternatives[best_alt]:.2f} dB",
    delta=f"{alternatives[best_alt] - p_out:+.2f} dB",
    help=f"{best_alt}",
)

if p_out < p_in:
    st.error(
        f"**`{method_name}` made this image worse** — {p_out:.2f} dB against the "
        f"noisy input's {p_in:.2f} dB. At this noise level its blurring costs more "
        f"than the noise does. `{best_alt}` reaches {alternatives[best_alt]:.2f} dB."
    )
elif alternatives[best_alt] > p_out + 1.0:
    st.warning(
        f"`{best_alt}` beats `{method_name}` by "
        f"**{alternatives[best_alt] - p_out:.2f} dB** on this noise. The winner "
        "changes with the noise *model*, not just the level — that is the whole "
        "claim this project tests."
    )
else:
    st.success(
        f"`{method_name}` is within 1 dB of the best filter on this noise "
        f"({best_alt}, {alternatives[best_alt]:.2f} dB)."
    )

st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4 = st.tabs(
    ["Every filter on this noise", "The role reversal", "Noise level sweep", "Noise histograms"]
)

with t1:
    st.caption(
        "All six filters plus the do-nothing control, on **this** image and "
        "**this** noise."
    )
    rows = []
    for name in dn.METHODS:
        o, tm = run(name, noisy)
        rows.append(
            {
                "method": name,
                "psnr": round(psnr(o, clean), 3),
                "ssim": round(ssim(o, clean), 4),
                "ms": round(tm.median_ms, 2),
            }
        )
    styler, frame = ui.comparison_table(
        rows, [("PSNR (dB)", "psnr", True), ("SSIM", "ssim", True), ("Time (ms)", "ms", False)]
    )
    st.dataframe(styler, width="stretch")
    st.caption(
        "Watch where **Do nothing (control)** lands. At low noise it beats several "
        "real filters — which is the practically useful part of the comparison."
    )
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="denoising_methods.csv",
        mime="text/csv",
    )
    panels = [clean, noisy] + [run(n, noisy)[0] for n in dn.METHODS]
    caps = ["Clean", f"Noisy {psnr(noisy, clean):.1f} dB"] + [
        f"{n} {r['psnr']:.1f} dB" for n, r in zip(dn.METHODS, rows)
    ]
    for start in range(0, len(panels), 3):
        row = st.columns(3)
        for col, im, cap in zip(row, panels[start : start + 3], caps[start : start + 3]):
            col.image(im, caption=cap, width="stretch")

with t2:
    st.caption(
        "The same two filters on two noise models. Median is the best filter on "
        "salt and pepper by 6.8 dB and the worst on Gaussian; bilateral is the "
        "reverse. A filter is not good or bad — it is matched or mismatched."
    )
    c = st.columns(3)
    for col, (k, lv, lb) in zip(c, dn.NOISE_TYPES):
        n2 = dn.make_noisy(clean, k, lv, seed=0)
        med = dn.tuned_call("Median", n2, k)
        bil = dn.tuned_call("Bilateral", n2, k)
        col.markdown(f"**{lb}**")
        col.image(n2, caption=f"noisy {psnr(n2, clean):.2f} dB", width="stretch")
        col.image(med, caption=f"Median {psnr(med, clean):.2f} dB", width="stretch")
        col.image(bil, caption=f"Bilateral {psnr(bil, clean):.2f} dB", width="stretch")

with t3:
    st.caption(
        "Every filter as the noise gets worse, with the do-nothing control. Where "
        "a line crosses the control is where denoising starts being worth doing."
    )
    levels = (
        (0.01, 0.03, 0.06, 0.12, 0.20)
        if kind == "salt_pepper"
        else ((5.0, 10.0, 20.0, 35.0, 50.0) if kind == "gaussian" else (120.0, 60.0, 30.0, 15.0, 8.0))
    )
    names = [n for n in dn.METHODS if not n.startswith("Do nothing")]
    series = {"Noisy input (do nothing)": []} | {n: [] for n in names}
    for lv in levels:
        nz = dn.make_noisy(clean, kind, lv, seed=0)
        series["Noisy input (do nothing)"].append(round(psnr(nz, clean), 3))
        for n in names:
            o = dn.tuned_call(n, nz, kind) if use_tuned else dn.METHODS[n](nz)
            series[n].append(round(psnr(o, clean), 3))
    fig = ui.lines_figure(
        list(levels), series,
        xlabel={"gaussian": "sigma", "salt_pepper": "density", "poisson": "lambda"}[kind],
        ylabel="PSNR (dB)",
        title="Where each filter starts beating doing nothing",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t4:
    st.caption(
        "Three noise models, three different shapes. Gaussian moves every pixel a "
        "little; salt and pepper moves a few pixels all the way to 0 or 255; "
        "Poisson's variance grows with the signal. One filter cannot be right for "
        "all three, and the histogram is why."
    )
    fig = ui.histogram_figure(
        {
            "clean": to_gray(clean).ravel(),
            **{lb: to_gray(dn.make_noisy(clean, k, lv, seed=0)).ravel()
               for k, lv, lb in dn.NOISE_TYPES},
        },
        bins=128,
        title="What each noise model does to the pixel distribution",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.divider()
with st.expander("Does tuning transfer? (tuned on 3 images, scored on 3 others)"):
    st.markdown(
        """
Denoising comparisons disagree with each other mostly because of tuning, so this
project tunes every filter by grid search — and then checks whether the tuning
means anything. Fitted on three images and scored on three **held-out** ones:

| Filter | Fitted | Held-out gain over the default |
|---|---|---:|
| Box | ksize=3 | **−0.81 dB** |
| Gaussian | sigma=1.2 | −0.02 dB |
| Median | ksize=5 | 0.00 dB |
| **Bilateral** | sigma_color=120 | **+4.13 dB** |
| Non-local means | h=18 | +0.20 dB |
| Wiener (adaptive) | ksize=5 | 0.00 dB |

**Tuning transfers for one filter of six.** The bilateral filter's `sigma_color`
has to be set relative to the noise level and the library default is wrong for
it, so tuning is worth 4 dB. For everything else the "tuned" number is
essentially the default — and for the box filter the fitted parameter actively
*loses* on images it was not fitted to, which is overfitting on a six-image grid
search over one parameter.
"""
    )
