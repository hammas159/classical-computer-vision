"""Grayscale to colour: six methods, three controls, and a metric that had to change.

The question
------------
Throw away the colour of a photograph and try to put it back. The truth is the
original, so this is one of the few image-restoration problems with an exact,
uncontested answer — and that makes the *metric* the interesting part.

> **The claim under test:** PSNR is the wrong measure for colourisation. Half
> true, and the half that is false is worth as much as the half that is not. On
> these twelve photographs PSNR **ranks the methods in exactly the same order** as
> a chroma-only error does — so the usual accusation is wrong here. But its
> *scale* is almost entirely luminance: an image with **no colour at all** scores
> 22.2 dB, and an image with the **exactly correct colours** and flattened
> luminance scores about 13. The number is real; it is just mostly not about the
> thing being estimated.

> **What both metrics agree on is worse.** Two of the four colourisers score
> *below the do-nothing control*: Welsh transfer at 25.3 chroma error and
> pseudo-colour at 45.3, against 18.7 for returning the grey image untouched.
> Inventing colour made the colour error worse.

> **And the ceiling was knowable in advance.** The statistic the twelve
> photographs were *selected* on — how much chroma survives inside one luminance
> level — predicts what an oracle handed that image's own luminance-to-colour
> mapping still gets wrong, across the whole set, monotonically.

The selection axis
------------------
The twelve photographs are spread across **greyscale ambiguity**: the
population-weighted spread of chroma *within* a luminance bin. A picture where
each grey level means one colour (a seal on flat ice, 1.8) is recoverable; one
where the same grey is red here and green there (a woman in a red top against
tulips and grass, 19.7) is not. The axis is the project's own subject matter
rather than a proxy for it, and `oracle_residual` checks that it predicts what it
claims to.

The controls
------------
`Do nothing (grey)` returns the input untouched — the floor that exposes the
metric. `Global mean chroma` gives every pixel the image's average colour, which
is two numbers for the whole picture. `reference_sensitivity` runs the transfer method with
every one of the other eleven photographs as its reference, which is what
separates "the method works" from "the reference was the answer".
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from shared.io import real_photo

EPS = 1e-9

#: The twelve photographs, in order of **greyscale ambiguity** -- the chroma
#: spread within a luminance bin, measured by `ambiguity` below.
IMAGES = (
    "seal_on_grey_ice",
    "farmland_from_the_air",
    "wine_bottles_in_a_rack",
    "llama_at_a_stone_wall",
    "tree_against_tropical_sky",
    "otter_on_a_log",
    "chicks_in_a_nest",
    "lizard_on_pebbles",
    "bears_at_the_water",
    "glacier_cave_mouth",
    "soldier_on_the_grass",
    "woman_in_a_red_top",
)

#: Luminance bins for the ambiguity statistic. 32 is fine enough that a bin is a
#: narrow band of grey and coarse enough that a bin has thousands of pixels.
BINS = 32

#: The optimisation in `levin_scribbles` is solved at this width and the chroma
#: is then upsampled. A 481x321 image is 154,000 unknowns per channel; solving at
#: full resolution works but takes about forty times as long for a result that
#: differs in the third decimal, and chroma is low-frequency anyway.
SOLVE_WIDTH = 200


def load(name: str) -> np.ndarray:
    return real_photo(name)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """Lab with a and b centred on zero, so 'no colour' is literally zero."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def from_lab(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[..., 1] += 128.0
    out[..., 2] += 128.0
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def greyscale(rgb: np.ndarray) -> np.ndarray:
    """The input every method gets: three identical channels, no chroma at all."""
    lab = to_lab(rgb)
    lab[..., 1] = 0.0
    lab[..., 2] = 0.0
    return from_lab(lab)


def ambiguity(rgb: np.ndarray, bins: int = BINS) -> float:
    """How much chroma variation survives inside one luminance level, in Lab units.

    The project's selection axis and the quantity the whole problem turns on. If
    this were zero, luminance would determine colour and colourisation would be a
    lookup table.
    """
    lab = to_lab(rgb)
    L = lab[..., 0].ravel()
    a = lab[..., 1].ravel()
    b = lab[..., 2].ravel()
    idx = np.clip((L / 256.0 * bins).astype(int), 0, bins - 1)
    total, weight = 0.0, 0.0
    for k in range(bins):
        sel = idx == k
        n = int(sel.sum())
        if n < 50:
            continue
        total += float(np.sqrt(a[sel].var() + b[sel].var())) * n
        weight += n
    return total / max(weight, 1.0)


# --------------------------------------------------------------------------- #
# the methods
# --------------------------------------------------------------------------- #


def do_nothing(grey: np.ndarray, truth: np.ndarray | None = None,
               reference: np.ndarray | None = None, **kw) -> np.ndarray:
    """Return the grey image. The control that exposes the metric."""
    return grey.copy()


def global_mean_chroma(grey: np.ndarray, truth: np.ndarray | None = None,
                       reference: np.ndarray | None = None, **kw) -> np.ndarray:
    """Give every pixel the image's mean a and b. Two numbers for the picture.

    A control with a real claim in it: how much of a colourisation score is just
    getting the overall cast right?
    """
    source = truth if truth is not None else reference
    lab = to_lab(grey)
    if source is None:
        return grey.copy()
    ref = to_lab(source)
    lab[..., 1] = float(ref[..., 1].mean())
    lab[..., 2] = float(ref[..., 2].mean())
    return from_lab(lab)


def pseudo_colour(grey: np.ndarray, truth: np.ndarray | None = None,
                  reference: np.ndarray | None = None,
                  colormap: int = cv2.COLORMAP_VIRIDIS, **kw) -> np.ndarray:
    """A colour map applied to luminance. The colourisation that is not one.

    It produces a vividly coloured image with no relation to the scene, and it is
    in the table because it is what "add colour to a grayscale image" means in a
    great deal of software.
    """
    mapped = cv2.applyColorMap(cv2.cvtColor(grey, cv2.COLOR_RGB2GRAY), colormap)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    # keep the original luminance, so only the chroma is the colormap's
    lab = to_lab(rgb)
    lab[..., 0] = to_lab(grey)[..., 0]
    return from_lab(lab)


def luminance_lookup(grey: np.ndarray, truth: np.ndarray | None = None,
                     reference: np.ndarray | None = None, bins: int = BINS,
                     **kw) -> np.ndarray:
    """Colour as a function of luminance alone, learned from a source image.

    With ``truth`` as the source this is the **oracle**: the best any method could
    do if grey determined colour. Its residual is the irreducible ambiguity of
    the photograph, and it is not small.
    """
    source = truth if truth is not None else reference
    if source is None:
        return grey.copy()
    src = to_lab(source)
    lab = to_lab(grey)
    src_idx = np.clip((src[..., 0] / 256.0 * bins).astype(int), 0, bins - 1)
    dst_idx = np.clip((lab[..., 0] / 256.0 * bins).astype(int), 0, bins - 1)

    table_a = np.zeros(bins, np.float32)
    table_b = np.zeros(bins, np.float32)
    for k in range(bins):
        sel = src_idx == k
        if sel.any():
            table_a[k] = float(src[..., 1][sel].mean())
            table_b[k] = float(src[..., 2][sel].mean())
    lab[..., 1] = table_a[dst_idx]
    lab[..., 2] = table_b[dst_idx]
    return from_lab(lab)


#: The neighbourhood Welsh's method compares. A pixel is matched on its own
#: luminance *and* the local standard deviation around it, which is what keeps
#: sky from matching to a sunlit wall of the same brightness.
WELSH_WINDOW = 5
WELSH_SAMPLES = 400


def welsh_transfer(grey: np.ndarray, truth: np.ndarray | None = None,
                   reference: np.ndarray | None = None,
                   samples: int = WELSH_SAMPLES, seed: int = 0, **kw) -> np.ndarray:
    """Welsh, Ashikhmin & Mueller (2002): match on (luminance, local variance).

    Sample pixels from a reference photograph, remap the target's luminance to
    the reference's statistics, then give each target pixel the chroma of the
    closest sample in that two-dimensional space.

    The reference is **another photograph**, not the truth, so this is the one
    method here that could work on an image nobody has ever seen in colour.
    """
    source = reference if reference is not None else truth
    if source is None:
        return grey.copy()

    rng = np.random.default_rng(seed)
    src = to_lab(source)
    lab = to_lab(grey)

    def features(L):
        blur = cv2.blur(L, (WELSH_WINDOW, WELSH_WINDOW))
        sq = cv2.blur(L * L, (WELSH_WINDOW, WELSH_WINDOW))
        return np.sqrt(np.maximum(sq - blur * blur, 0.0))

    src_std = features(src[..., 0])
    dst_std = features(lab[..., 0])

    # luminance remapping, as the paper specifies: the target's luminance is
    # shifted and scaled to the reference's mean and standard deviation, or the
    # match is between two different exposure ranges rather than two textures
    src_L = src[..., 0]
    dst_L = lab[..., 0]
    remapped = (dst_L - dst_L.mean()) * (src_L.std() / max(dst_L.std(), EPS)) \
        + src_L.mean()

    idx = rng.choice(src_L.size, size=min(samples, src_L.size), replace=False)
    sample_L = src_L.ravel()[idx]
    sample_s = src_std.ravel()[idx]
    sample_a = src[..., 1].ravel()[idx]
    sample_b = src[..., 2].ravel()[idx]

    flat_L = remapped.ravel()
    flat_s = dst_std.ravel()
    # nearest neighbour in (luminance, local std), in chunks so the pairwise
    # distance matrix never exists in full
    out_a = np.empty(flat_L.size, np.float32)
    out_b = np.empty(flat_L.size, np.float32)
    chunk = 20000
    for start in range(0, flat_L.size, chunk):
        stop = min(start + chunk, flat_L.size)
        d = ((flat_L[start:stop, None] - sample_L[None, :]) ** 2
             + (flat_s[start:stop, None] - sample_s[None, :]) ** 2)
        nearest = np.argmin(d, axis=1)
        out_a[start:stop] = sample_a[nearest]
        out_b[start:stop] = sample_b[nearest]

    lab[..., 1] = out_a.reshape(lab.shape[:2])
    lab[..., 2] = out_b.reshape(lab.shape[:2])
    return from_lab(lab)


#: How many scribbles the Levin method gets by default, and how wide each is.
N_SCRIBBLES = 40
SCRIBBLE_RADIUS = 3


def sample_scribbles(truth: np.ndarray, n: int = N_SCRIBBLES,
                     radius: int = SCRIBBLE_RADIUS, seed: int = 0):
    """Take ``n`` small colour marks from the truth, at spread-out positions.

    This is the honest version of "a user scribbles on the image": the colours are
    correct because a user would know them, the positions are chosen by a
    stratified grid rather than by looking at which positions help, and the count
    is the thing swept.
    """
    h, w = truth.shape[:2]
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    mask = np.zeros((h, w), bool)
    chosen = 0
    for gy in range(side):
        for gx in range(side):
            if chosen >= n:
                break
            y = int((gy + rng.uniform(0.25, 0.75)) * h / side)
            x = int((gx + rng.uniform(0.25, 0.75)) * w / side)
            y = int(np.clip(y, radius, h - radius - 1))
            x = int(np.clip(x, radius, w - radius - 1))
            mask[y - radius:y + radius + 1, x - radius:x + radius + 1] = True
            chosen += 1
    return mask


def levin_scribbles(grey: np.ndarray, truth: np.ndarray | None = None,
                    reference: np.ndarray | None = None,
                    n_scribbles: int = N_SCRIBBLES, seed: int = 0,
                    width: int = SOLVE_WIDTH, **kw) -> np.ndarray:
    """Levin, Lischinski & Weiss (2004): propagate a few known colours.

    The assumption is one sentence long and does all the work: **neighbouring
    pixels with similar luminance should have similar colour.** That is written as
    a quadratic cost over every pixel and its eight neighbours, with weights that
    fall off with the luminance difference, and minimised subject to the scribbled
    pixels being fixed. The minimum is a sparse linear system.

    Solved at reduced width and upsampled — chroma is low-frequency, and the
    difference at full resolution is in the third decimal for forty times the
    time.
    """
    if truth is None:
        return grey.copy()

    h, w = grey.shape[:2]
    scale = min(1.0, width / float(w))
    sh, sw = max(8, int(round(h * scale))), max(8, int(round(w * scale)))

    small_grey = cv2.resize(grey, (sw, sh), interpolation=cv2.INTER_AREA)
    small_truth = cv2.resize(truth, (sw, sh), interpolation=cv2.INTER_AREA)
    marks = sample_scribbles(small_truth, n_scribbles,
                             max(1, int(SCRIBBLE_RADIUS * scale) + 1), seed)

    Y = to_lab(small_grey)[..., 0] / 100.0
    truth_lab = to_lab(small_truth)

    n = sh * sw
    index = np.arange(n).reshape(sh, sw)
    rows, cols, vals = [], [], []

    for y in range(sh):
        for x in range(sw):
            r = index[y, x]
            if marks[y, x]:
                rows.append(r)
                cols.append(r)
                vals.append(1.0)
                continue
            neighbours = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < sh and 0 <= nx < sw:
                        neighbours.append((ny, nx))
            values = np.array([Y[ny, nx] for ny, nx in neighbours], np.float64)
            centre = float(Y[y, x])
            variance = float(np.var(np.append(values, centre)))
            sigma = max(variance, 2e-6)
            weights = np.exp(-((values - centre) ** 2) / (2.0 * sigma))
            weights /= max(weights.sum(), EPS)
            rows.append(r)
            cols.append(r)
            vals.append(1.0)
            for (ny, nx), wgt in zip(neighbours, weights):
                rows.append(r)
                cols.append(int(index[ny, nx]))
                vals.append(-float(wgt))

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    out = to_lab(small_grey)
    for channel in (1, 2):
        rhs = np.zeros(n, np.float64)
        rhs[index[marks]] = truth_lab[..., channel][marks]
        solution = sla.spsolve(A, rhs)
        out[..., channel] = np.asarray(solution).reshape(sh, sw)

    chroma = cv2.resize(out[..., 1:], (w, h), interpolation=cv2.INTER_LINEAR)
    full = to_lab(grey)
    full[..., 1] = chroma[..., 0]
    full[..., 2] = chroma[..., 1]
    return from_lab(full)


METHODS: dict[str, Callable] = {
    "Do nothing (grey, control)": do_nothing,
    "Global mean chroma (control)": global_mean_chroma,
    "Pseudo-colour (viridis)": pseudo_colour,
    "Welsh transfer (reference)": welsh_transfer,
    "Levin scribbles (40)": levin_scribbles,
    "Luminance lookup (oracle)": luminance_lookup,
}

#: The methods that are given the truth in some form, and so are upper bounds
#: rather than usable algorithms. Named here so no table can quietly rank them
#: against the others without saying so.
ORACLES = ("Levin scribbles (40)", "Luminance lookup (oracle)",
           "Global mean chroma (control)")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def chroma_error(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Mean Euclidean distance in (a, b). The metric that measures the colour.

    Luminance is excluded on purpose: every method here reproduces it exactly, so
    including it measures the part of the problem nobody is solving.
    """
    p = to_lab(prediction)
    t = to_lab(truth)
    return float(np.mean(np.hypot(p[..., 1] - t[..., 1], p[..., 2] - t[..., 2])))


def rgb_psnr(prediction: np.ndarray, truth: np.ndarray) -> float:
    """The metric usually quoted, kept so the table can show what it does."""
    mse = float(np.mean((prediction.astype(np.float64)
                         - truth.astype(np.float64)) ** 2))
    return float("inf") if mse <= EPS else float(10.0 * np.log10(255.0 ** 2 / mse))


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness: is there any colour in the output at all?"""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #


def reference_for(name: str, images=IMAGES) -> str:
    """The reference photograph Welsh transfer is given: the next one along.

    Chosen by a rule rather than by hand, so no image gets a reference picked
    because it happened to work.
    """
    order = list(images)
    return order[(order.index(name) + 1) % len(order)]


def colourise(name: str, method: str, **kw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(grey_input, prediction, truth)`` for one image and method."""
    truth = load(name)
    grey = greyscale(truth)
    reference = load(reference_for(name))
    prediction = METHODS[method](grey, truth=truth, reference=reference, **kw)
    return grey, prediction, truth


def evaluate(images=IMAGES) -> list[dict]:
    """Every method on every photograph, by both metrics."""
    acc = {name: {"chroma": [], "psnr": [], "colourfulness": []} for name in METHODS}
    for image in images:
        truth = load(image)
        for method in METHODS:
            _, prediction, _ = colourise(image, method)
            acc[method]["chroma"].append(chroma_error(prediction, truth))
            acc[method]["psnr"].append(rgb_psnr(prediction, truth))
            acc[method]["colourfulness"].append(colourfulness(prediction))
    return [
        {
            "method": name,
            "chroma_error": round(float(np.mean(a["chroma"])), 3),
            "rgb_psnr_db": round(float(np.mean(a["psnr"])), 3),
            "colourfulness": round(float(np.mean(a["colourfulness"])), 2),
            "is_oracle": name in ORACLES,
        }
        for name, a in acc.items()
    ]


def per_image(images=IMAGES) -> list[dict]:
    rows = []
    for image in images:
        truth = load(image)
        row = {"image": image, "ambiguity": round(ambiguity(truth), 3),
               "colourfulness": round(colourfulness(truth), 2)}
        for method in METHODS:
            _, prediction, _ = colourise(image, method)
            row[method] = round(chroma_error(prediction, truth), 3)
        rows.append(row)
    return rows


def metric_comparison(images=IMAGES) -> dict:
    """Do PSNR and a chroma-only error rank the methods the same way?

    On this set they do, and saying so matters: the standard complaint about
    PSNR for colourisation is that it reorders the methods, and here it does not.
    What it does instead is compress the scale — see `psnr_is_luminance`.
    """
    rows = evaluate(images)
    by_psnr = [r["method"] for r in sorted(rows, key=lambda r: -r["rgb_psnr_db"])]
    by_chroma = [r["method"] for r in sorted(rows, key=lambda r: r["chroma_error"])]
    control = "Do nothing (grey, control)"
    control_chroma = next(r["chroma_error"] for r in rows if r["method"] == control)
    return {
        "by_psnr": by_psnr,
        "by_chroma_error": by_chroma,
        "rankings_agree": by_psnr == by_chroma,
        "control_rank": by_chroma.index(control) + 1,
        "methods_worse_than_doing_nothing": [
            r["method"] for r in rows
            if r["method"] != control and r["chroma_error"] > control_chroma
        ],
    }


def axis_predicts_the_ceiling(images=IMAGES) -> dict:
    """Does the selection axis predict the oracle's residual?

    It has to be asked rather than asserted. `ambiguity` is computed from the
    colour image before anything runs; the oracle residual is what a
    luminance-to-colour lookup fitted on that same image still cannot recover.
    If the first did not predict the second, the axis would be decoration.
    """
    rows = oracle_residual(images)
    x = np.array([r["ambiguity"] for r in rows])
    y = np.array([r["oracle_chroma_error"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 5),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "ambiguity_range": [round(float(x.min()), 3), round(float(x.max()), 3)],
        "oracle_error_range": [round(float(y.min()), 3), round(float(y.max()), 3)],
    }


def oracle_residual(images=IMAGES) -> list[dict]:
    """What the best possible luminance-to-colour mapping still gets wrong.

    The oracle is fitted on the very image it is tested on, so this is not a
    method — it is the question *"if grey determined colour, how good could that
    be?"*, answered per photograph. The answer should track the ambiguity axis
    the twelve were selected on, and that is checked rather than assumed.
    """
    rows = []
    for image in images:
        truth = load(image)
        _, prediction, _ = colourise(image, "Luminance lookup (oracle)")
        rows.append({
            "image": image,
            "ambiguity": round(ambiguity(truth), 3),
            "oracle_chroma_error": round(chroma_error(prediction, truth), 3),
        })
    return rows


def sweep_scribbles(images=IMAGES, counts=(1, 4, 16, 40, 100, 250)) -> list[dict]:
    """How many scribbles Levin's method needs, and where it stops paying."""
    rows = []
    for n in counts:
        errors = []
        for image in images:
            truth = load(image)
            _, prediction, _ = colourise(image, "Levin scribbles (40)", n_scribbles=n)
            errors.append(chroma_error(prediction, truth))
        rows.append({
            "scribbles": n,
            "chroma_error": round(float(np.mean(errors)), 3),
        })
    return rows


def reference_sensitivity(images=IMAGES) -> list[dict]:
    """Welsh transfer with **every** other photograph as the reference.

    The first version of this control shuffled the references, which let three of
    the twelve keep their own and produced a table that said nothing. Running all
    eleven alternatives instead answers the real question: how much of this
    method's score is the method, and how much is which photograph it was handed?

    The answer is that the spread across references is several times the
    difference between the method and the do-nothing control.
    """
    rows = []
    for image in images:
        truth = load(image)
        grey = greyscale(truth)
        errors = {}
        for other in images:
            if other == image:
                continue
            errors[other] = chroma_error(
                welsh_transfer(grey, reference=load(other)), truth)
        ordered = sorted(errors.items(), key=lambda kv: kv[1])
        # the rule is relative to whatever list was passed, so a subset stays
        # self-consistent instead of pointing at an image that is not in it
        rule = reference_for(image, images)
        rows.append({
            "image": image,
            "rule_reference": rule,
            "with_rule_reference": round(errors[rule], 3),
            "best_reference": ordered[0][0],
            "best": round(ordered[0][1], 3),
            "worst_reference": ordered[-1][0],
            "worst": round(ordered[-1][1], 3),
            "median": round(float(np.median(list(errors.values()))), 3),
            "spread": round(ordered[-1][1] - ordered[0][1], 3),
            "grey_control": round(chroma_error(grey, truth), 3),
        })
    return rows


def psnr_is_luminance(images=IMAGES) -> list[dict]:
    """How much of the RGB PSNR each method owes to luminance it did not invent.

    Every method here copies the input's luminance exactly. The comparison is
    against an image with the true chroma and *destroyed* luminance, which is the
    same amount of information the other way round.

    The chroma-only image's own chroma error is not quite zero -- 0.03 to 0.95 Lab
    units. Flattening L pushes saturated colours outside the sRGB gamut and the
    round trip clips them, which is why the two largest residuals are the tropical
    sky and the red top. It is reported rather than rounded away.
    """
    rows = []
    for image in images:
        truth = load(image)
        grey = greyscale(truth)

        lab = to_lab(truth)
        scrambled = lab.copy()
        scrambled[..., 0] = float(lab[..., 0].mean())
        chroma_only = from_lab(scrambled)

        rows.append({
            "image": image,
            "grey_only_psnr": round(rgb_psnr(grey, truth), 3),
            "chroma_only_psnr": round(rgb_psnr(chroma_only, truth), 3),
            "grey_only_chroma_error": round(chroma_error(grey, truth), 3),
            "chroma_only_chroma_error": round(chroma_error(chroma_only, truth), 3),
        })
    return rows
