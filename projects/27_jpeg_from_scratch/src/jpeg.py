"""JPEG from scratch: build the codec, then plot the rate-distortion curve.

The question
------------
JPEG is four ideas stacked — colour transform, DCT, quantisation, entropy coding
— and only one of them actually loses information.

> **The claim under test:** essentially all of JPEG's compression comes from
> **quantisation**, and the DCT itself is lossless (to floating-point precision).
> Each stage can be switched off independently to measure its individual
> contribution, which no "how JPEG works" article ever does.

And the headline output:

> **The rate-distortion curve *is* the result.** Quality is not a number, it is a
> curve of bits-per-pixel against PSNR, and comparing codecs at a single quality
> setting is meaningless.

Everything here is implemented directly — the DCT, the standard quantisation
tables, zig-zag ordering, run-length and an entropy estimate — so each stage can
be disabled. The comparison against OpenCV's real encoder then measures how much
a production implementation buys over the textbook one.
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.io import to_float, to_gray, to_uint8
from shared.metrics import psnr, ssim

EPS = 1e-9

BLOCK = 8

#: The luminance quantisation table from the JPEG standard (Annex K).
#: The values rise toward the bottom-right because that corner holds the highest
#: spatial frequencies, which the eye is least sensitive to. The table IS the
#: perceptual model — everything else in JPEG is bookkeeping.
Q_LUMA = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    np.float32,
)

#: Chrominance table. Far coarser than luma, because human vision has much lower
#: spatial resolution for colour than for brightness — the single biggest reason
#: JPEG works at all.
Q_CHROMA = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    np.float32,
)

#: Zig-zag order. Reading a quantised block this way groups the zeros together at
#: the end, which is what makes run-length coding effective -- the ordering is
#: doing real compression work, not just tidiness.
ZIGZAG = np.array(
    [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]
)


def quality_scale(quality: int) -> float:
    """Scale factor for the quantisation tables, per the standard's formula."""
    q = int(np.clip(quality, 1, 100))
    return (5000.0 / q if q < 50 else 200.0 - 2.0 * q) / 100.0


def scaled_table(table: np.ndarray, quality: int) -> np.ndarray:
    scaled = np.floor(table * quality_scale(quality) + 0.5)
    return np.clip(scaled, 1, 255).astype(np.float32)


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #


def dct_2d(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def idct_2d(block: np.ndarray) -> np.ndarray:
    return cv2.idct(block.astype(np.float32))


def blockwise(channel: np.ndarray, fn) -> np.ndarray:
    """Apply a function to every non-overlapping 8x8 block.

    The block structure is the source of JPEG's characteristic artefact: each
    block is quantised independently, so neighbouring blocks land on different
    reconstruction levels and the seam becomes visible. Nothing else in the codec
    creates it.
    """
    h, w = channel.shape
    ph = (BLOCK - h % BLOCK) % BLOCK
    pw = (BLOCK - w % BLOCK) % BLOCK
    padded = cv2.copyMakeBorder(channel, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    out = np.zeros_like(padded, np.float32)
    for y in range(0, padded.shape[0], BLOCK):
        for x in range(0, padded.shape[1], BLOCK):
            out[y : y + BLOCK, x : x + BLOCK] = fn(padded[y : y + BLOCK, x : x + BLOCK])
    return out[:h, :w]


# --------------------------------------------------------------------------- #
# the codec, with each stage switchable
# --------------------------------------------------------------------------- #


def encode_decode(
    img: np.ndarray,
    quality: int = 50,
    use_dct: bool = True,
    use_quantisation: bool = True,
    use_chroma_subsampling: bool = True,
    use_colour_transform: bool = True,
):
    """Round-trip an image through the codec, returning ``(reconstructed, stats)``.

    Every stage has an off switch on purpose: turning one off and re-measuring is
    how each stage's individual contribution to both size and distortion is
    isolated.
    """
    stats: dict[str, float] = {}

    if use_colour_transform and img.ndim == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        channels = [ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]]
        tables = [Q_LUMA, Q_CHROMA, Q_CHROMA]
    else:
        gray = to_gray(img).astype(np.float32)
        channels = [gray]
        tables = [Q_LUMA]

    out_channels = []
    nonzero_total = 0
    coeff_total = 0
    entropy_bits = 0.0

    for idx, (channel, table) in enumerate(zip(channels, tables)):
        full_shape = channel.shape
        work = channel
        if use_chroma_subsampling and idx > 0:
            # 4:2:0 -- halve the chroma resolution before anything else. This is
            # a 4x reduction in chroma data for almost no visible cost.
            work = cv2.resize(
                work, (max(1, work.shape[1] // 2), max(1, work.shape[0] // 2)),
                interpolation=cv2.INTER_AREA,
            )

        shifted = work - 128.0  # centre the range, as the standard specifies
        q = scaled_table(table, quality)

        def forward(block):
            coeffs = dct_2d(block) if use_dct else block.copy()
            if use_quantisation:
                coeffs = np.round(coeffs / q)
            return coeffs

        coeffs = blockwise(shifted, forward)

        nonzero_total += int(np.count_nonzero(coeffs))
        coeff_total += int(coeffs.size)
        entropy_bits += estimate_entropy_bits(coeffs)

        def inverse(block):
            c = block * q if use_quantisation else block.copy()
            return idct_2d(c) if use_dct else c

        restored = blockwise(coeffs, inverse) + 128.0
        if use_chroma_subsampling and idx > 0:
            restored = cv2.resize(
                restored, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_LINEAR
            )
        out_channels.append(np.clip(restored, 0, 255))

    if len(out_channels) == 3:
        merged = np.stack(out_channels, axis=-1).astype(np.uint8)
        recon = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    else:
        recon = out_channels[0].astype(np.uint8)

    pixels = float(img.shape[0] * img.shape[1])
    stats["nonzero_fraction"] = nonzero_total / max(coeff_total, 1)
    stats["estimated_bpp"] = entropy_bits / max(pixels, 1.0)
    return recon, stats


def estimate_entropy_bits(coeffs: np.ndarray) -> float:
    """Shannon entropy of the run-length symbol stream, in bits.

    A real encoder uses Huffman or arithmetic coding; both approach the entropy.
    Estimating it directly gives a table-independent lower bound on the bitrate,
    which is the honest thing to plot when the point is the rate-distortion
    trade-off rather than one implementation's table choices.
    """
    flat = zigzag_flatten(coeffs)
    symbols = run_length_symbols(flat)
    if not symbols:
        return 0.0
    _, counts = np.unique(np.array(symbols, dtype=object).astype(str), return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)) * len(symbols))


def zigzag_flatten(coeffs: np.ndarray) -> np.ndarray:
    """Read every 8x8 block in zig-zag order and concatenate."""
    order = np.argsort(ZIGZAG.ravel())
    h, w = coeffs.shape
    out = []
    for y in range(0, h - h % BLOCK, BLOCK):
        for x in range(0, w - w % BLOCK, BLOCK):
            out.append(coeffs[y : y + BLOCK, x : x + BLOCK].ravel()[order])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def run_length_symbols(flat: np.ndarray) -> list:
    """(run of zeros, value) pairs — JPEG's actual symbol alphabet."""
    symbols = []
    run = 0
    for v in flat:
        iv = int(v)
        if iv == 0:
            run += 1
            if run == 16:
                symbols.append((15, 0))
                run = 0
        else:
            symbols.append((run, iv))
            run = 0
    if run:
        symbols.append((0, 0))  # end of block
    return symbols


# --------------------------------------------------------------------------- #
# artefact measurement
# --------------------------------------------------------------------------- #


def blockiness(img: np.ndarray) -> float:
    """Discontinuity at 8-pixel boundaries relative to elsewhere.

    JPEG's signature artefact is a seam every 8 pixels. Comparing the mean
    gradient *at* block boundaries with the mean gradient between them isolates
    it from the image's own content — a value near 1.0 means no blocking.
    """
    g = to_float(to_gray(img))
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    on_edge = (cols + 1) % BLOCK == 0
    if not on_edge.any() or (~on_edge).sum() == 0:
        return 1.0
    return float(dx[:, on_edge].mean() / max(dx[:, ~on_edge].mean(), EPS))


# --------------------------------------------------------------------------- #
# experiments
# --------------------------------------------------------------------------- #

IMAGES = ("astronaut", "coffee", "chelsea", "camera", "brick")
QUALITIES = (5, 10, 20, 30, 50, 70, 85, 95)


def rate_distortion(images=IMAGES, qualities=QUALITIES):
    """The curve that *is* the result: bits per pixel against PSNR and SSIM.

    Both this implementation and OpenCV's production encoder are measured, so the
    gap between a textbook codec and a real one is a number rather than a
    hand-wave.
    """
    from shared import io

    rows = []
    for q in qualities:
        ours_psnr, ours_ssim, ours_bpp, ours_block = [], [], [], []
        cv_psnr, cv_bpp = [], []
        for name in images:
            clean = io.sample(name)
            recon, stats = encode_decode(clean, quality=q)
            ours_psnr.append(psnr(recon, clean))
            ours_ssim.append(ssim(recon, clean))
            ours_bpp.append(stats["estimated_bpp"])
            ours_block.append(blockiness(recon))

            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(clean, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(q)],
            )
            if ok:
                decoded = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                cv_psnr.append(psnr(decoded, clean))
                cv_bpp.append(len(buf) * 8.0 / (clean.shape[0] * clean.shape[1]))

        rows.append(
            {
                "quality": q,
                "bpp_ours": round(float(np.mean(ours_bpp)), 4),
                "psnr_ours": round(float(np.mean(ours_psnr)), 3),
                "ssim_ours": round(float(np.mean(ours_ssim)), 4),
                "blockiness": round(float(np.mean(ours_block)), 3),
                "bpp_opencv": round(float(np.mean(cv_bpp)), 4) if cv_bpp else None,
                "psnr_opencv": round(float(np.mean(cv_psnr)), 3) if cv_psnr else None,
            }
        )
    return rows


def stage_ablation(quality: int = 50, images=IMAGES):
    """Turn each stage off and measure what it was contributing.

    The row that matters is "DCT off, quantisation off": if the reconstruction is
    essentially perfect there, the transform really is lossless and every bit of
    the loss belongs to quantisation.
    """
    from shared import io

    configs = {
        "Full codec": {},
        "No chroma subsampling": {"use_chroma_subsampling": False},
        "No colour transform (RGB)": {"use_colour_transform": False},
        "No quantisation": {"use_quantisation": False},
        "No DCT (quantise pixels)": {"use_dct": False},
        "No DCT, no quantisation": {"use_dct": False, "use_quantisation": False},
    }
    rows = []
    for label, kwargs in configs.items():
        p, s, bpp, block = [], [], [], []
        for name in images:
            clean = io.sample(name)
            recon, stats = encode_decode(clean, quality=quality, **kwargs)
            p.append(psnr(recon, clean))
            s.append(ssim(recon, clean))
            bpp.append(stats["estimated_bpp"])
            block.append(blockiness(recon))
        rows.append(
            {
                "configuration": label,
                "psnr_db": round(float(np.mean(p)), 3),
                "ssim": round(float(np.mean(s)), 4),
                "estimated_bpp": round(float(np.mean(bpp)), 4),
                "blockiness": round(float(np.mean(block)), 3),
            }
        )
    return rows


def coefficient_statistics(quality: int = 50, images=IMAGES):
    """How many DCT coefficients survive quantisation, by frequency band.

    The compression, stated as a count: at moderate quality the overwhelming
    majority of coefficients quantise to exactly zero, and they are almost all in
    the high-frequency corner.
    """
    from shared import io

    rows = []
    for q in (10, 30, 50, 75, 95):
        surviving = np.zeros((BLOCK, BLOCK), np.float64)
        total = 0
        for name in images:
            clean = io.sample(name)
            gray = to_gray(clean).astype(np.float32) - 128.0
            table = scaled_table(Q_LUMA, q)
            h, w = gray.shape
            for y in range(0, h - h % BLOCK, BLOCK):
                for x in range(0, w - w % BLOCK, BLOCK):
                    coeffs = np.round(dct_2d(gray[y : y + BLOCK, x : x + BLOCK]) / table)
                    surviving += coeffs != 0
                    total += 1
        rows.append(
            {
                "quality": q,
                "nonzero_fraction": round(float(surviving.sum() / max(total * 64, 1)), 4),
                "dc_survival": round(float(surviving[0, 0] / max(total, 1)), 4),
                "highest_freq_survival": round(float(surviving[7, 7] / max(total, 1)), 4),
            }
        )
    return rows


def compress(img: np.ndarray, quality: int = 50):
    """Round-trip one image, for the UI."""
    return encode_decode(img, quality=quality)
