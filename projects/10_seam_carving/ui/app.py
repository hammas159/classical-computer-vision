"""Streamlit UI for project 10 — seam carving.

    streamlit run ui/app.py

Resize an image two ways at once — content-aware and plain — and see the
difference measured rather than asserted. The region being tracked is found from
the image itself (the highest-energy box), so nothing is pasted in and no
annotation is needed.
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

import seam_carving as sc  # noqa: E402
from shared import io as shared_io  # noqa: E402
from shared import ui  # noqa: E402
from shared.bench import timeit  # noqa: E402

st.set_page_config(
    page_title="Seam carving — Classical CV", layout="wide", initial_sidebar_state="collapsed"
)

# Carving is O(n) sequential dynamic programmes for n removed columns. A 600px
# image takes over a second; capping the working width keeps the app responsive
# and is stated rather than hidden, because it is the project's main finding
# about the method.
MAX_WIDTH = 420

st.title("Seam carving")
st.caption(
    "Content-aware resizing removes the lowest-energy connected path of pixels, "
    "repeatedly, so a resize deletes boring pixels instead of squashing "
    "everything equally. The question is how much better that is than a plain "
    "rescale, and what it costs. No training, no network, no GPU."
)

left, right = st.columns([1, 1])

with left:
    source = st.radio(
        "Image source", ["Sample image", "Upload your own"], horizontal=True
    )
    uploaded, image_name = None, "coffee"
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Photo", type=["png", "jpg", "jpeg", "bmp", "webp"])
    else:
        image_name = st.selectbox("Image", list(sc.IMAGES), index=0)
    reduction = st.slider(
        "Width reduction", 0.05, 0.70, 0.20, 0.05,
        help="Past about 45% there are no low-energy paths left and every seam "
             "has to cross something.",
    )

with right:
    energy_name = st.selectbox("Energy function", list(sc.ENERGIES), index=0)
    st.caption(
        {
            "Gradient |dx|+|dy|": "The energy from the original paper — the L1 sum "
            "of the Sobel derivatives. Cheap and, measured here, indistinguishable "
            "from the alternatives.",
            "Sobel magnitude": "sqrt(dx² + dy²), the true gradient magnitude "
            "rather than the L1 sum. A more principled quantity that carves "
            "0.1 points differently.",
            "Laplacian": "Second-order change, so it peaks on fine detail and is "
            "more sensitive to texture and noise than a first derivative.",
            "Local std (entropy-like)": "High across a textured AREA rather than "
            "only on its edges, so in principle it should keep busy regions whole "
            "instead of protecting their outlines and hollowing them out.",
        }.get(energy_name, "")
    )
    show_seams = st.checkbox("Show the seams that would be removed", value=True)

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
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
else:
    img = shared_io.sample(image_name)

if img.shape[1] > MAX_WIDTH:
    s = MAX_WIDTH / img.shape[1]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    st.caption(
        f"Working at {img.shape[1]}x{img.shape[0]}. Carving is a sequential "
        f"dynamic programme per removed column, so full resolution would take "
        f"seconds per slider move — which is itself the finding."
    )

energy_fn = sc.ENERGIES[energy_name]
mask, _ = sc.subject_region(img, energy_fn)
target = max(8, int(img.shape[1] * (1 - reduction)))

(carved, carved_mask), carve_time = timeit(
    lambda: sc.carve(img, target, energy_fn, track=mask), runs=1, warmup=0
)
rescaled, rescale_time = timeit(
    lambda: cv2.resize(img, (target, img.shape[0]), interpolation=cv2.INTER_AREA),
    runs=1, warmup=0,
)
rescaled_mask = cv2.resize(mask, (target, mask.shape[0]), interpolation=cv2.INTER_NEAREST)


def outline(image, m, colour=(60, 220, 60)):
    out = image.copy()
    contours, _ = cv2.findContours(
        (m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


carved_kept = float((carved_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0)
rescale_kept = float((rescaled_mask > 0).sum()) / max(float((mask > 0).sum()), 1.0)

cols = st.columns(3)
cols[0].image(
    outline(img, mask) if not show_seams else outline(sc.seam_overlay(img, energy_fn, n=min(40, img.shape[1] - target)), mask),
    caption=f"1 · Original {img.shape[1]}px" + (" · seams to be removed in red" if show_seams else ""),
    width="stretch",
)
cols[1].image(
    outline(carved, carved_mask),
    caption=f"2 · Seam carved to {target}px — {carved_kept:.1%} of the region kept",
    width="stretch",
)
cols[2].image(
    outline(rescaled, rescaled_mask),
    caption=f"3 · Plain rescale to {target}px — {rescale_kept:.1%} kept",
    width="stretch",
)


st.subheader("Measurements")
m = st.columns(5)
m[0].metric(
    "Region kept", f"{carved_kept:.1%}",
    delta=f"{(carved_kept - rescale_kept) * 100:+.1f} pts vs rescale",
)
e_carved = float(sc.energy_gradient(carved).sum()) / max(float(sc.energy_gradient(img).sum()), 1e-9)
e_plain = float(sc.energy_gradient(rescaled).sum()) / max(float(sc.energy_gradient(img).sum()), 1e-9)
m[1].metric(
    "Energy kept", f"{e_carved:.1%}",
    delta=f"{(e_carved - e_plain) * 100:+.1f} pts vs rescale",
    help="Seam carving's OWN objective: it claims to remove low-energy pixels.",
)
_, _, base_aspect = sc.region_shape(mask)
_, _, carved_aspect = sc.region_shape(carved_mask)
m[2].metric("Aspect retained", f"{carved_aspect / max(base_aspect, 1e-9):.1%}")
m[3].metric("Carve time", f"{carve_time.median_ms:.0f} ms")
m[4].metric(
    "Cost vs rescale",
    f"{carve_time.median_ms / max(rescale_time.median_ms, 1e-9):.0f}×",
    help=f"A plain rescale took {rescale_time.median_ms:.2f} ms.",
)

advantage = (carved_kept - rescale_kept) * 100
if advantage < 0:
    st.error(
        f"**On this image seam carving is worse than a plain rescale** — "
        f"{carved_kept:.1%} of the region kept against {rescale_kept:.1%}, for "
        f"{carve_time.median_ms / max(rescale_time.median_ms, 1e-9):.0f}× the time. "
        "This happens when the subject fills most of the frame: there are no "
        "low-energy paths to route around it, so every seam has to cross "
        "something. Averaged over four sample images the method wins by 11.6 "
        "points — an average that describes none of them individually, and hides "
        "that there is a regime where it loses."
    )
elif advantage < 5:
    st.warning(
        f"Seam carving is ahead by **{advantage:.1f} points** here, for "
        f"{carve_time.median_ms / max(rescale_time.median_ms, 1e-9):.0f}× the cost. "
        "Worth checking whether that is worth the compute for your use."
    )
else:
    st.success(
        f"Seam carving kept **{advantage:.1f} points more** of the region than a "
        f"plain rescale — and **{(e_carved - e_plain) * 100:.1f} points more** of the "
        "image's gradient energy, which is the objective it actually optimises."
    )

st.divider()
st.subheader("Distributions and matrices")

t1, t2, t3, t4 = st.tabs(
    ["Reduction sweep", "The four energies", "Energy maps", "What it costs"]
)

with t1:
    st.caption(
        "Both methods against the reduction, computed live on **this** image. "
        "Watch the energy lines separate much further than the region lines — "
        "carving wins its own objective more decisively than the one you care about."
    )
    levels = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60)
    series = {k: [] for k in ("carved: region", "rescale: region", "carved: energy", "rescale: energy")}
    for red in levels:
        t = max(8, int(img.shape[1] * (1 - red)))
        c_out, c_mask = sc.carve(img, t, energy_fn, track=mask)
        r_out = cv2.resize(img, (t, img.shape[0]), interpolation=cv2.INTER_AREA)
        r_mask = cv2.resize(mask, (t, mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        total = max(float(sc.energy_gradient(img).sum()), 1e-9)
        series["carved: region"].append(float((c_mask > 0).sum()) / max(float((mask > 0).sum()), 1))
        series["rescale: region"].append(float((r_mask > 0).sum()) / max(float((mask > 0).sum()), 1))
        series["carved: energy"].append(float(sc.energy_gradient(c_out).sum()) / total)
        series["rescale: energy"].append(float(sc.energy_gradient(r_out).sum()) / total)
    fig = ui.lines_figure(
        list(levels), series, xlabel="width reduction", ylabel="fraction retained",
        title="Carving's edge over a rescale, on this image",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t2:
    st.caption(
        "The one-line choice write-ups agonise over, measured. Across four sample "
        "images the four energies span 0.5 points of region retention while the "
        "gap to a plain rescale is 11.6 — a 23× difference."
    )
    rows = []
    total = max(float(sc.energy_gradient(img).sum()), 1e-9)
    for name, fn in sc.ENERGIES.items():
        (out, out_mask), tm = timeit(
            lambda f=fn: sc.carve(img, target, f, track=mask), runs=1, warmup=0
        )
        _, _, asp = sc.region_shape(out_mask)
        rows.append(
            {
                "energy": name,
                "region_kept": round(float((out_mask > 0).sum()) / max(float((mask > 0).sum()), 1), 4),
                "aspect": round(asp / max(base_aspect, 1e-9), 4),
                "energy_kept": round(float(sc.energy_gradient(out).sum()) / total, 4),
                "ms": round(tm.median_ms, 1),
            }
        )
    rows.append(
        {
            "energy": "Plain rescale (control)",
            "region_kept": round(rescale_kept, 4),
            "aspect": round(1 - reduction, 4),
            "energy_kept": round(e_plain, 4),
            "ms": round(rescale_time.median_ms, 1),
        }
    )
    styler, frame = ui.comparison_table(
        rows,
        [
            ("Region kept", "region_kept", True),
            ("Aspect retained", "aspect", True),
            ("Energy kept", "energy_kept", True),
            ("Time (ms)", "ms", False),
        ],
        row_key="energy",
    )
    st.dataframe(styler, width="stretch")
    st.download_button(
        "Download this matrix as CSV",
        frame.to_csv().encode("utf-8"),
        file_name="seam_energies.csv",
        mime="text/csv",
    )
    c = st.columns(4)
    for col, (name, fn) in zip(c, sc.ENERGIES.items()):
        out, _ = sc.carve(img, target, fn)
        col.image(out, caption=name, width="stretch")

with t3:
    st.caption(
        "Four definitions of 'boring'. They look different and, measured, they "
        "carve almost identically — which is the point."
    )
    c = st.columns(len(sc.ENERGIES))
    for col, (name, fn) in zip(c, sc.ENERGIES.items()):
        e = fn(img)
        col.image(
            (e / max(float(e.max()), 1e-9) * 255).astype(np.uint8),
            caption=name, width="stretch",
        )
    fig = ui.histogram_figure(
        {
            name: (fn(img) / max(float(fn(img).max()), 1e-9) * 255).ravel()
            for name, fn in sc.ENERGIES.items()
        },
        bins=96, xlabel="normalised energy",
        title="Every energy is dominated by near-zero pixels — which is why carving works at all",
    )
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with t4:
    st.caption("The trade, stated plainly.")
    ratio = carve_time.median_ms / max(rescale_time.median_ms, 1e-9)
    st.markdown(
        f"""
| | Seam carving | Plain rescale |
|---|---:|---:|
| Region kept | {carved_kept:.1%} | {rescale_kept:.1%} |
| Energy kept | {e_carved:.1%} | {e_plain:.1%} |
| Time | {carve_time.median_ms:.0f} ms | {rescale_time.median_ms:.2f} ms |
| **Relative cost** | **{ratio:.0f}×** | 1× |

Seam carving is `n` sequential dynamic programmes for `n` removed columns, and
each one walks the image row by row — the row loop cannot be vectorised because
row `i` needs row `i−1`. A plain rescale is one interpolation pass.

That ratio is the reason content-aware resizing is a feature you invoke rather
than a default: it buys **{advantage:+.1f} points** of region retention here,
and the question is whether that is worth **{ratio:.0f}×** the compute for what
you are doing.
"""
    )

st.divider()
with st.expander("What extreme reduction looks like"):
    panels, caps = [img], [f"Original {img.shape[1]}px"]
    for red in (0.30, 0.50, 0.70):
        t = max(8, int(img.shape[1] * (1 - red)))
        out, _ = sc.carve(img, t, energy_fn)
        panels.append(out)
        caps.append(f"carved −{red:.0%}")
    c = st.columns(4)
    for col, im, cap in zip(c, panels, caps):
        col.image(im, caption=cap, width="stretch")
    st.markdown(
        "Seam carving works by having somewhere to route *around*. Past roughly "
        "45% of the width there are no low-energy paths left, every remaining "
        "seam has to cross something, and the distortion it was avoiding arrives "
        "all at once — as bent edges rather than as uniform squashing."
    )
