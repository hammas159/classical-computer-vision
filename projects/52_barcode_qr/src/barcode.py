"""Barcode and QR detection: localise it, then actually decode it.

The question
------------
Most "barcode detection" demos stop at drawing a box. That is the easy half, and
it hides the interesting result.

> **The claim under test:** localisation and decoding degrade at *different*
> rates. A detector will happily find a barcode long after it has become
> impossible to read — so a detection rate massively overstates how well a
> pipeline works. The gap between "found" and "decoded" is the number worth
> reporting.

Decoding gives something rare in this repo: a **binary, objective ground truth**
with no metric choice at all. The encoded string either comes back or it does not.

The codes are generated here, so the expected payload is known exactly, and the
degradations — blur, rotation, perspective, noise, low resolution — are applied
in known amounts.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from shared.bench import timeit
from shared.io import to_float, to_gray, to_uint8

EPS = 1e-9


# --------------------------------------------------------------------------- #
# generating codes
# --------------------------------------------------------------------------- #

#: Code 39 patterns: each character is 9 elements, wide/narrow encoded as 1/0.
#: Chosen because it is self-checking and simple enough to generate correctly
#: without a barcode library, so the project has no extra dependency.
CODE39 = {
    "0": "000110100", "1": "100100001", "2": "001100001", "3": "101100000",
    "4": "000110001", "5": "100110000", "6": "001110000", "7": "000100101",
    "8": "100100100", "9": "001100100", "A": "100001001", "B": "001001001",
    "C": "101001000", "D": "000011001", "E": "100011000", "F": "001011000",
    "G": "000001101", "H": "100001100", "I": "001001100", "J": "000011100",
    "*": "010010100",
}


def make_code39(text: str, height: int = 120, narrow: int = 3, quiet: int = 20) -> np.ndarray:
    """Render a Code 39 barcode as a binary image.

    The **quiet zone** — the blank margin — is not decoration. Scanners use it to
    find where the code starts, and a code rendered flush against the image edge
    frequently fails to decode for that reason alone.
    """
    payload = f"*{text.upper()}*"
    bars = []
    for ch in payload:
        pattern = CODE39.get(ch)
        if pattern is None:
            continue
        for i, wide in enumerate(pattern):
            width = narrow * 3 if wide == "1" else narrow
            bars.append((width, i % 2 == 0))  # alternate bar/space
        bars.append((narrow, False))  # inter-character gap

    total = sum(w for w, _ in bars) + 2 * quiet
    img = np.full((height, total), 255, np.uint8)
    x = quiet
    for width, is_bar in bars:
        if is_bar:
            img[:, x : x + width] = 0
        x += width
    return img


def make_qr(text: str, size: int = 240) -> np.ndarray:
    """Render a QR code using OpenCV's own encoder.

    OpenCV ships a QR encoder, so this needs no extra dependency and — more
    usefully — guarantees the code is validly formed, which a hand-rolled
    generator would not.
    """
    encoder = cv2.QRCodeEncoder.create()
    code = encoder.encode(text)
    return cv2.resize(code, (size, size), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# localisation
# --------------------------------------------------------------------------- #


def locate_gradient_morphology(img: np.ndarray) -> np.ndarray | None:
    """The classic barcode localiser: directional gradient, then closing.

    A 1-D barcode has a strong **horizontal** gradient and almost no vertical
    one, because it is made of vertical bars. Subtracting the y-gradient from the
    x-gradient produces a response that is high on barcodes and low on ordinary
    texture — which is the whole trick, and it is why this does not work on QR
    codes, which are isotropic.
    """
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy)))

    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))


def locate_variance(img: np.ndarray, window: int = 21) -> np.ndarray | None:
    """Localise by local intensity variance.

    A barcode region is high contrast everywhere. Simpler than the gradient
    method and *isotropic*, so unlike the gradient approach it should work on QR
    codes as well as on 1-D barcodes.
    """
    gray = to_float(to_gray(img))
    mean = cv2.blur(gray, (window, window))
    sq = cv2.blur(gray * gray, (window, window))
    var = np.maximum(sq - mean * mean, 0.0)
    norm = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))


def locate_qr_detector(img: np.ndarray) -> np.ndarray | None:
    """OpenCV's dedicated QR detector, which finds the three finder patterns.

    A QR code's three corner squares have a fixed 1:1:3:1:1 ratio along any line
    through them, which is scale and rotation invariant and is why QR codes are
    so much easier to localise than 1-D barcodes.
    """
    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(to_gray(img))
    if not ok or points is None:
        return None
    return points.reshape(-1, 2)


LOCATORS: dict[str, Callable] = {
    "Gradient + morphology": locate_gradient_morphology,
    "Local variance": locate_variance,
    "OpenCV QR detector": locate_qr_detector,
}


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #


def decode_qr(img: np.ndarray) -> str:
    """Decode a QR code. Returns the payload, or an empty string."""
    detector = cv2.QRCodeDetector()
    try:
        data, points, _ = detector.detectAndDecode(to_gray(img))
    except cv2.error:
        return ""
    return data or ""


def decode_barcode(img: np.ndarray) -> str:
    """Decode a 1-D barcode with OpenCV's BarcodeDetector, if available.

    ``cv2.barcode.BarcodeDetector`` moved between modules across OpenCV versions
    and is absent from some builds, so its availability is checked rather than
    assumed — an ImportError here would take the whole experiment down.
    """
    detector = None
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        detector = cv2.barcode.BarcodeDetector()
    elif hasattr(cv2, "BarcodeDetector"):
        detector = cv2.BarcodeDetector()
    if detector is None:
        return ""
    # The arity of detectAndDecode changed between OpenCV versions: 4.5 returns
    # (ok, decoded_info, decoded_type, points) and 4.14 returns
    # (decoded_info, decoded_type, points) with no boolean at all. Unpacking a
    # fixed number raises ValueError on the other one, which took the whole
    # experiment down rather than reporting an empty decode.
    try:
        result = detector.detectAndDecode(to_gray(img))
    except cv2.error:
        return ""

    if len(result) == 4:
        ok, decoded = result[0], result[1]
        if not ok:
            return ""
    else:
        decoded = result[0]

    if decoded is None or len(decoded) == 0:
        return ""
    first = decoded[0] if isinstance(decoded, (list, tuple, np.ndarray)) else decoded
    return str(first) if first else ""


def barcode_decoder_available() -> bool:
    """Whether this OpenCV build can decode 1-D barcodes at all.

    Reported in the results rather than silently producing zeros, because "0%
    decoded" and "no decoder installed" are very different findings.
    """
    return bool(
        (hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"))
        or hasattr(cv2, "BarcodeDetector")
    )


# --------------------------------------------------------------------------- #
# scenes and degradations
# --------------------------------------------------------------------------- #

#: Twelve photographs ranked by **how much they already look like a barcode** to
#: the localiser's own cue: strong horizontal gradient, weak vertical, closed up
#: with a wide rectangular kernel. That is the axis that decides how hard
#: localisation is, and no stock axis in `tools/select_images.py` measures it.
#:
#: The spread is 9% to 77% of the frame responding. A bomber against open sky is
#: the control; zebras, saguaro ribs and coiled rope are genuine false positives
#: — repeating vertical structure is what a 1-D barcode *is*.
IMAGES = (
    "bomber_over_cloud",      # barcode-like  9.4% - the control
    "turquoise_lake",         #              18.9%
    "polar_bears_playing",    #              22.2%
    "zebra_herd",             #              26.1% - actual vertical stripes
    "canoe_on_the_lake",      #              29.7%
    "cougar_among_birches",   #              30.4%
    "trocadero_statue",       #              33.0%
    "skiff_in_weed",          #              33.9%
    "saguaro_blossom",        #              41.2% - dense vertical ribbing
    "coiled_rope",            #              43.4% - strong repeating stripes
    "giraffes_drinking",      #              50.0%
    "buffalo_in_the_river",   #              76.7% - the busiest here
)


def load_scene(name: str) -> np.ndarray:
    """One of the project's background photographs."""
    from shared import io

    return io.real_photo(name)


def barcode_like_share(img: np.ndarray) -> float:
    """Percentage of the frame that responds to the localiser's own cue.

    Horizontal gradient minus vertical, blurred, closed with a wide rectangle,
    Otsu-thresholded — which is `locate_gradient_morphology` up to the contour
    step. It is the axis the pool is ordered by and it predicts false positives
    before any code is placed.
    """
    g = to_gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, -1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, -1)
    resp = cv2.convertScaleAbs(cv2.subtract(np.abs(gx), np.abs(gy)))
    resp = cv2.morphologyEx(cv2.GaussianBlur(resp, (9, 9), 0), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)))
    _, th = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((th > 0).mean() * 100.0)


PAYLOADS = ("HELLO123", "ABC789", "CLASSICAL", "VISION42")
BLUR_LEVELS = (0.0, 1.0, 2.0, 3.5, 5.0, 8.0)
ROTATIONS = (0.0, 5.0, 15.0, 30.0, 45.0, 90.0)
NOISE_LEVELS = (0.0, 10.0, 25.0, 50.0, 80.0)
SCALES = (1.0, 0.7, 0.5, 0.35, 0.25)
PERSPECTIVES = (0.0, 0.05, 0.12, 0.2, 0.3)


def place_on_background(code: np.ndarray, canvas: int = 480, seed: int = 0,
                        rotation: float = 0.0, scale: float = 1.0,
                        perspective: float = 0.0, background: str | None = None):
    """Put a code onto a cluttered background under a known transform.

    Returns ``(image, true_corners)``. The clutter is what makes localisation a
    real problem — on a blank background every method scores 100%.

    ``background`` names one of `IMAGES` and uses that photograph instead of the
    generated rectangles. That is the harder and more honest test: the generated
    clutter has no repeating vertical structure, and a barcode localiser looks
    for exactly that, so a photograph of zebras or coiled rope produces false
    positives that random rectangles never will.
    """
    rng = np.random.default_rng(seed)
    if background is None:
        bg = rng.integers(60, 200, (canvas, canvas), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (0, 0), 3.0)
        for _ in range(10):
            x, y = int(rng.integers(0, canvas - 60)), int(rng.integers(0, canvas - 60))
            cv2.rectangle(bg, (x, y), (x + 55, y + 45), int(rng.integers(0, 255)), -1)
    else:
        photo = to_gray(load_scene(background))
        h0, w0 = photo.shape[:2]
        side = min(h0, w0)
        crop = photo[(h0 - side) // 2:(h0 - side) // 2 + side,
                     (w0 - side) // 2:(w0 - side) // 2 + side]
        bg = cv2.resize(crop, (canvas, canvas), interpolation=cv2.INTER_AREA)

    h, w = code.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    resized = cv2.resize(code, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = (canvas - nw) // 2
    y0 = (canvas - nh) // 2
    src = np.float32([[0, 0], [nw - 1, 0], [nw - 1, nh - 1], [0, nh - 1]])
    dst = src + np.float32([[x0, y0]] * 4)

    if perspective > 0:
        jitter = rng.uniform(-perspective, perspective, (4, 2)) * np.float32([nw, nh])
        dst = dst + jitter.astype(np.float32)
    if rotation:
        m = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), rotation, 1.0)
        dst = cv2.transform(dst.reshape(-1, 1, 2), m).reshape(-1, 2)

    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(code, H, (canvas, canvas), borderValue=255)
    mask = cv2.warpPerspective(np.full((nh, nw), 255, np.uint8), H, (canvas, canvas))

    out = bg.copy()
    out[mask > 0] = warped[mask > 0]
    return out, dst.astype(np.float32)


def degrade(img: np.ndarray, blur: float = 0.0, noise_sigma: float = 0.0, seed: int = 0):
    from shared import synth

    out = img
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur, borderType=cv2.BORDER_REFLECT)
    if noise_sigma > 0:
        out = synth.gaussian_noise(out, sigma=noise_sigma, seed=seed)
    return out


def localisation_iou(predicted, truth, shape) -> float:
    """IoU of two quadrilaterals, rasterised."""
    from shared.metrics import iou

    if predicted is None:
        return 0.0
    a = np.zeros(shape[:2], np.uint8)
    b = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(a, [np.int32(predicted)], 255)
    cv2.fillPoly(b, [np.int32(truth)], 255)
    return iou(a, b)


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

FOUND_IOU = 0.4


def evaluate_qr(payloads=PAYLOADS, blur: float = 0.0, rotation: float = 0.0,
                scale: float = 1.0, perspective: float = 0.0, noise_sigma: float = 0.0,
                runs: int = 1):
    """Localisation *and* decoding for QR codes under one condition.

    Both numbers on the same row is the entire point — the gap between them is
    the project's finding.
    """
    acc = {n: {"iou": [], "found": [], "ms": []} for n in LOCATORS}
    decoded = []

    for i, payload in enumerate(payloads):
        code = make_qr(payload)
        scene, truth = place_on_background(
            code, seed=i, rotation=rotation, scale=scale, perspective=perspective
        )
        scene = degrade(scene, blur=blur, noise_sigma=noise_sigma, seed=i)

        for name, fn in LOCATORS.items():
            box, timing = timeit(lambda f=fn: f(scene), runs=runs, warmup=0)
            score = localisation_iou(box, truth, scene.shape)
            acc[name]["iou"].append(score)
            acc[name]["found"].append(score >= FOUND_IOU)
            acc[name]["ms"].append(timing.median_ms)

        decoded.append(decode_qr(scene) == payload)

    rows = [
        {
            "locator": n,
            "mean_iou": round(float(np.mean(a["iou"])), 4),
            "found_rate": round(float(np.mean(a["found"])), 4),
            "median_ms": round(float(np.median(a["ms"])), 2),
        }
        for n, a in acc.items()
    ]
    return rows, {"decode_rate": round(float(np.mean(decoded)), 4)}


def sweep_blur(payloads=PAYLOADS, levels=BLUR_LEVELS):
    """The headline sweep: detection survives blur far longer than decoding.

    Detection needs only the code's *presence*; decoding needs every module
    resolved. So the two curves should separate, and the size of that separation
    is how misleading a detection-only number is.
    """
    rows = []
    for blur in levels:
        located, extra = evaluate_qr(payloads=payloads, blur=blur)
        best = max(r["found_rate"] for r in located)
        rows.append(
            {
                "blur_sigma": blur,
                "best_found_rate": round(best, 4),
                "decode_rate": extra["decode_rate"],
                "gap": round(best - extra["decode_rate"], 4),
            }
        )
    return rows


def sweep_rotation(payloads=PAYLOADS, rotations=ROTATIONS):
    """QR finder patterns are rotation invariant; the gradient localiser is not."""
    rows = []
    for rot in rotations:
        located, extra = evaluate_qr(payloads=payloads, rotation=rot)
        row: dict[str, float] = {"rotation_deg": rot, "decode_rate": extra["decode_rate"]}
        for r in located:
            row[r["locator"]] = r["found_rate"]
        rows.append(row)
    return rows


def sweep_scale(payloads=PAYLOADS, scales=SCALES):
    """Resolution limit: below roughly four pixels per module, decoding must fail.

    That is an information-theoretic floor, not an implementation weakness — the
    modules are no longer separable on the sensor.
    """
    rows = []
    for s in scales:
        located, extra = evaluate_qr(payloads=payloads, scale=s)
        best = max(r["found_rate"] for r in located)
        rows.append(
            {
                "scale": s,
                "approx_px_per_module": round(240 * s / 25.0, 2),
                "best_found_rate": round(best, 4),
                "decode_rate": extra["decode_rate"],
            }
        )
    return rows


def sweep_perspective(payloads=PAYLOADS, levels=PERSPECTIVES):
    """QR decoding rectifies perspective internally; how much can it absorb?"""
    rows = []
    for p in levels:
        located, extra = evaluate_qr(payloads=payloads, perspective=p)
        best = max(r["found_rate"] for r in located)
        rows.append(
            {
                "perspective": p,
                "best_found_rate": round(best, 4),
                "decode_rate": extra["decode_rate"],
            }
        )
    return rows


def sweep_noise(payloads=PAYLOADS, levels=NOISE_LEVELS):
    rows = []
    for sigma in levels:
        located, extra = evaluate_qr(payloads=payloads, noise_sigma=sigma)
        best = max(r["found_rate"] for r in located)
        rows.append(
            {
                "noise_sigma": sigma,
                "best_found_rate": round(best, 4),
                "decode_rate": extra["decode_rate"],
            }
        )
    return rows


def evaluate_on_photographs(images=None, payloads=PAYLOADS, blur: float = 0.0,
                            rotation: float = 0.0, scale: float = 1.0):
    """Locate and decode a QR code placed on each real photograph.

    The generated background has no repeating vertical structure, so a barcode
    localiser never sees a false positive on it. A photograph of zebras does. The
    per-image column is the point: localisation difficulty tracks the background,
    and decoding does not care about it at all.
    """
    rows = []
    images = IMAGES if images is None else images
    for name in images:
        share = barcode_like_share(load_scene(name))
        found = {"Gradient + morphology": [], "Local variance": [], "QRCodeDetector": []}
        decoded = []
        for i, payload in enumerate(payloads):
            code = make_qr(payload)
            img, truth = place_on_background(code, seed=i, rotation=rotation,
                                             scale=scale, background=name)
            if blur:
                img = degrade(img, blur=blur, seed=i)
            for label, fn in (("Gradient + morphology", locate_gradient_morphology),
                              ("Local variance", locate_variance),
                              ("QRCodeDetector", locate_qr_detector)):
                corners = fn(img)
                found[label].append(
                    localisation_iou(corners, truth, img.shape) >= FOUND_IOU)
            decoded.append(decode_qr(img) == payload)

        row: dict[str, float | str] = {"image": name, "barcode_like": round(share, 1)}
        for label, hits in found.items():
            row[label] = round(float(np.mean(hits)), 4)
        row["decoded"] = round(float(np.mean(decoded)), 4)
        rows.append(row)
    return rows


def photo_versus_generated_background(payloads=PAYLOADS):
    """Does a real background make localisation harder than generated clutter?

    The honest check on the synthetic arm. If the generated rectangles are as hard
    as a photograph, the whole project could stay synthetic; if they are not, the
    synthetic numbers are optimistic and should be labelled so.
    """
    rows = []
    for label, backgrounds in (("Generated clutter", [None]),
                               ("Photographs", list(IMAGES))):
        found = {"Gradient + morphology": [], "Local variance": [], "QRCodeDetector": []}
        decoded = []
        for background in backgrounds:
            for i, payload in enumerate(payloads):
                code = make_qr(payload)
                img, truth = place_on_background(code, seed=i, background=background)
                for name, fn in (("Gradient + morphology", locate_gradient_morphology),
                                 ("Local variance", locate_variance),
                                 ("QRCodeDetector", locate_qr_detector)):
                    corners = fn(img)
                    found[name].append(
                        localisation_iou(corners, truth, img.shape) >= FOUND_IOU)
                decoded.append(decode_qr(img) == payload)

        row: dict[str, float | str] = {"background": label}
        for name, hits in found.items():
            row[name] = round(float(np.mean(hits)), 4)
        row["decoded"] = round(float(np.mean(decoded)), 4)
        rows.append(row)
    return rows


def evaluate_barcode_1d(payloads=PAYLOADS, blur_levels=BLUR_LEVELS):
    """The 1-D case, where the directional gradient localiser is designed to win."""
    available = barcode_decoder_available()
    rows = []
    for blur in blur_levels:
        ious, decodes = [], []
        for i, payload in enumerate(payloads):
            code = make_code39(payload)
            scene, truth = place_on_background(code, seed=i)
            scene = degrade(scene, blur=blur, seed=i)
            box = locate_gradient_morphology(scene)
            ious.append(localisation_iou(box, truth, scene.shape))
            decodes.append(decode_barcode(scene) == payload if available else False)
        rows.append(
            {
                "blur_sigma": blur,
                "mean_iou": round(float(np.mean(ious)), 4),
                "found_rate": round(float(np.mean([s >= FOUND_IOU for s in ious])), 4),
                "decode_rate": round(float(np.mean(decodes)), 4) if available else None,
                "decoder_available": available,
            }
        )
    return rows


def scan(img: np.ndarray, kind: str = "qr"):
    """Localise and decode in one call, for the UI."""
    if kind == "qr":
        return locate_qr_detector(img), decode_qr(img)
    return locate_gradient_morphology(img), decode_barcode(img)
