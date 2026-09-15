# Real photographs

Everything else in this repository is generated, because generated data has
**exact** ground truth. These four are real photographs, included for a different
purpose: to show each pipeline working on an image nobody constructed for it.

They are scored differently, and the difference is stated wherever they appear:

| | Generated scene | Real photograph |
|---|---|---|
| Ground truth | exact, by construction | **none** |
| Reports IoU / PSNR | yes | **no** |
| Answers | "how accurate is this method" | "does this work on a real image" |

A number quoted against a real photo here would be invented, so none is.

## Provenance

All of these come from [`opencv/opencv/samples/data`](https://github.com/opencv/opencv/tree/4.x/samples/data),
distributed under the **BSD 3-Clause** licence with OpenCV itself.

| File | What it is | Used by |
|---|---|---|
| `messi5.jpg` | a footballer on a pitch — OpenCV's own GrabCut tutorial image | 02 portrait mode |
| `sudoku.png` | a newspaper page photographed at an angle | 01 document scanner |
| `imageTextN.png` | a page of clean printed text | 01 document scanner |
| `text_defocus.jpg` | printed text, defocused | 01, 20 deblurring |
| `squirrel.jpg` | a squirrel on a branch against foliage | 02 portrait mode |
| `butterfly.jpg` | a butterfly on a leaf — a flat subject on a busy background | 02 portrait mode |
| `fruits.jpg` | a cut orange among other fruit — a still life with no face | 02 portrait mode |
