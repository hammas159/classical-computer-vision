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

### OpenCV sample data

From [`opencv/opencv/samples/data`](https://github.com/opencv/opencv/tree/4.x/samples/data),
distributed under the **BSD 3-Clause** licence with OpenCV itself.

| File | What it is | Used by |
|---|---|---|
| `messi5.jpg` | a footballer on a pitch — OpenCV's own GrabCut tutorial image | 02 portrait mode |
| `messi5_matte.png` | the reference alpha matte for the above, annotated once and frozen | 02 portrait mode |
| `sudoku.png` | a newspaper page photographed at an angle | 01 document scanner |
| `imageTextN.png` | a page of clean printed text | 01 document scanner |
| `text_defocus.jpg` | printed text, defocused | 01, 20 deblurring |
| `squirrel.jpg` | a squirrel on a branch against foliage | 02 portrait mode |
| `butterfly.jpg` | a butterfly on a leaf — a flat subject on a busy background | 02 portrait mode |
| `fruits.jpg` | a cut orange among other fruit — a still life with no face | 02 portrait mode |
| `sheet_music.png` | a page of musical notation — ruled staves, no prose | 01 document scanner |
| `handwritten_digits.png` | a grid of handwritten digits | 01 document scanner |
| `printed_text_rotated.png` | printed prose, already rotated | 01 document scanner |
| `motion_text.jpg` | printed text smeared by camera motion | 01 document scanner |

### Kodak Lossless True Color Image Suite

Twelve photographs from the **Kodak PhotoCD PCD0992** suite, the standard test
set for image compression and restoration work since 1993. Downloaded from
[this mirror](https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite);
Kodak released them for unrestricted research use. Re-encoded here as JPEG
quality 92 to keep the repository small — the originals are lossless PNG.

They were added for **project 04 dehazing**, which needs scenes with real depth
structure, because haze depends on exactly one physical quantity: distance.

| File | Kodak # | What it is | Depth family |
|---|---:|---|---|
| `old_street.jpg` | 08 | a timbered street receding to a vanishing point | street |
| `painted_chalet.jpg` | 24 | a painted alpine chalet against a wooded hill | street |
| `stone_house.jpg` | 01 | a stone farmhouse wall, flat on | flat |
| `motocross.jpg` | 05 | a row of motocross bikes on a dirt bank | flat |
| `mountain_stream.jpg` | 13 | a stream running out of snow-capped mountains | mountain |
| `whitewater_raft.jpg` | 14 | a raft of people in whitewater | mountain |
| `lighthouse_cliff.jpg` | 21 | a lighthouse on a rocky headland | coast |
| `lighthouse_lawn.jpg` | 19 | a squat lighthouse behind a white fence | coast |
| `moored_boat.jpg` | 06 | a wooden boat moored in turquoise shallows | water |
| `sailboats.jpg` | 09 | sailboats under coloured spinnakers | water |
| `tropical_island.jpg` | 16 | a low island under towering cloud | sky |
| `warplane.jpg` | 20 | a propeller fighter against a pale sky | sky |

### Berkeley Segmentation Dataset (BSDS500)

Twelve natural scenes from the **BSDS500** test split, via
[this mirror](https://github.com/BIDS/BSDS500). Added for **project 07
copy-move forgery**, where what matters is the texture *around* the paste — a
block matcher is judged on what it can tell a duplicate apart from.

| File | What it is | Texture family |
|---|---|---|
| `bear_on_ice.jpg` | a bear on flat ice | almost none |
| `penguin_pebbles.jpg` | a penguin on a pebble beach | self-similar |
| `coral_reef.jpg` | a coral reef | self-similar |
| `fighter_jet.jpg` | a jet on tarmac | man-made straight edges |
| `family_by_van.jpg` | a family beside a van | man-made |
| `elephant_herd.jpg` | a herd of elephants | genuinely repeated objects |
| `rhinos_grass.jpg` | two rhinos on grass | repeated objects |
| `tortoise_rock.jpg` | a tortoise on broken rock | natural |
| `deer_in_brush.jpg` | a deer in winter brush | natural, low contrast |
| `lioness_savanna.jpg` | a lioness on dry grass | natural |
| `tiger_rocks.jpg` | a tiger on rocks | high contrast |
| `wolf_woods.jpg` | a wolf in leaf litter | fine scattered detail |

🚨 **Licence: research use, not a permissive licence.** BSDS500 is distributed
by UC Berkeley for research and education; its photographs originate in a
commercial stock collection. It is the standard segmentation benchmark and
appears in thousands of public repositories, but it is **not** public domain and
not equivalent to the OpenCV or Kodak material above. Stated here rather than
left for someone to discover. If that is a problem for a given use, these twelve
are the ones to replace.

### Source not recorded

These eight were added for project 02's subject-variety comparison and **their
exact source was not written down at the time.** That is a gap, and it is stated
here rather than backfilled with a guess:

`girl.jpg`, `dog.jpg`, `coffee_cup.jpg`, `woman_field.jpg`, `leopard.jpg`,
`man_camera.jpg`, `hiker.jpg`, `man_skyline.jpg` — all 700 × 525, no EXIF.

They are shown, never scored, and appear only in comparison figures. If their
licence cannot be established they should be replaced with Kodak or OpenCV
images, which carry explicit terms.
