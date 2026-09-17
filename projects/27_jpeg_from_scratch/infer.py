"""Compress your own image with this codec and see its rate-distortion curve.

    python infer.py photo.jpg
    python infer.py photo.jpg --quality 30 --out small.png
    python infer.py photo.jpg --ablate            # turn each stage off on your image

Every number here is real: a codec's loss is measured against the original,
which you have by definition. This is one of the few projects in the repository
where nothing has to be withheld.

What it adds to running `cv2.imwrite` is the **curve for your specific image**.
Compressibility is a property of the picture, not of the quality setting — the
twelve photographs behind this project differ by 5 dB at the same quality 95 —
so the useful output is where *your* image's curve bends.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR / "src")]

import numpy as np  # noqa: E402

from shared import io  # noqa: E402
from shared.metrics import psnr, ssim  # noqa: E402
from shared.report import init_console  # noqa: E402

import jpeg as jp  # noqa: E402

#: PSNR below which most viewers see the artefacts without being told to look.
VISIBLE_DB = 30.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--quality", type=int, default=None,
                    help="compress at one quality instead of sweeping")
    ap.add_argument("--ablate", action="store_true",
                    help="turn each codec stage off in turn on this image")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    init_console()
    img = io.imread(args.image)
    texture = jp.texture_of(img)
    print(f"{args.image}  {img.shape[1]}x{img.shape[0]}  texture {texture:.1f}")
    print(f"  (this project's pool spans texture 54 to 83, and differs by 5 dB "
          f"at the same quality because of it)")

    if args.ablate:
        configs = {
            "Full codec": {},
            "No chroma subsampling": {"use_chroma_subsampling": False},
            "No colour transform (RGB)": {"use_colour_transform": False},
            "No quantisation": {"use_quantisation": False},
            "No DCT (quantise pixels)": {"use_dct": False},
            "No DCT, no quantisation": {"use_dct": False, "use_quantisation": False},
        }
        print(f"\n{'configuration':28s} {'PSNR':>8s} {'SSIM':>7s} {'bpp':>7s} {'blocky':>8s}")
        panels = [img]
        for label, kwargs in configs.items():
            recon, stats = jp.encode_decode(img, quality=50, **kwargs)
            panels.append(recon)
            print(f"{label:28s} {psnr(recon, img):8.2f} {ssim(recon, img):7.4f} "
                  f"{stats['estimated_bpp']:7.3f} {jp.blockiness(recon):8.3f}")
        print("\nThe row to read is 'No DCT, no quantisation': if it is near-"
              "lossless then the transform loses nothing and every decibel the "
              "codec gives up belongs to quantisation.")
        out = np.hstack(panels)

    elif args.quality is not None:
        recon, stats = jp.encode_decode(img, quality=args.quality)
        p, s = psnr(recon, img), ssim(recon, img)
        print(f"\nquality {args.quality}: {stats['estimated_bpp']:.3f} bpp, "
              f"{p:.2f} dB, SSIM {s:.4f}, blockiness {jp.blockiness(recon):.3f}")
        print(f"  {stats['nonzero_fraction']:.1%} of the DCT coefficients survived "
              "quantisation; the rest rounded to zero.")
        if p < VISIBLE_DB:
            print(f"  Below {VISIBLE_DB:g} dB — the blocking is likely visible "
                  "without looking for it.")
        out = np.hstack([img, recon])

    else:
        print(f"\n{'quality':>8s} {'bpp':>8s} {'PSNR':>8s} {'SSIM':>8s} {'blocky':>8s}")
        rows = []
        for q in jp.QUALITIES:
            recon, stats = jp.encode_decode(img, quality=q)
            rows.append({"quality": q, "bpp": stats["estimated_bpp"],
                         "psnr": psnr(recon, img), "ssim": ssim(recon, img),
                         "blockiness": jp.blockiness(recon)})
            print(f"{q:8d} {rows[-1]['bpp']:8.3f} {rows[-1]['psnr']:8.2f} "
                  f"{rows[-1]['ssim']:8.4f} {rows[-1]['blockiness']:8.3f}")

        # where does this image stop being worth more bits?
        gains = [(rows[i + 1]["psnr"] - rows[i]["psnr"]) /
                 max(rows[i + 1]["bpp"] - rows[i]["bpp"], 1e-9)
                 for i in range(len(rows) - 1)]
        knee = int(np.argmin(gains))
        good = [r for r in rows if r["psnr"] >= VISIBLE_DB]
        print(f"\nThe curve flattens after quality {rows[knee]['quality']} — past "
              f"there each extra bit per pixel buys {gains[knee]:.1f} dB, against "
              f"{max(gains):.1f} dB at the steepest point.")
        if good:
            cheapest = min(good, key=lambda r: r["bpp"])
            print(f"Cheapest setting still above {VISIBLE_DB:g} dB on this image: "
                  f"quality {cheapest['quality']} at {cheapest['bpp']:.3f} bpp.")
        else:
            print(f"No setting reaches {VISIBLE_DB:g} dB on this image — it is "
                  "more detailed than the codec can carry cheaply.")
        out = np.hstack([img] + [jp.encode_decode(img, quality=q)[0]
                                 for q in (5, 20, 50, 95)])

    out_path = Path(args.out) if args.out else PROJECT_DIR / "results" / "compressed.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.imwrite(out_path, out)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
