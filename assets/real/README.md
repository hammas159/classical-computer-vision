# Real photographs

259 photographs, one pool per project and **no image shared between two
projects** — enforced by perceptual hash in `tools/check_image_reuse.py`, not by
filename.

There are three kinds of ground truth in this repository and they are not
interchangeable. Which one a project has decides what it is allowed to report,
and that is stated wherever a number appears.

| | Generated scene | Photograph, **annotated** | Photograph, **no annotation** |
|---|---|---|---|
| Truth | exact, by construction | five to seven people drew it | none |
| Reports a score | yes | yes, against the humans | **no** |
| Ceiling | 1.0 | **what one annotator scores against the others — about F 0.90** | — |
| Answers | "how accurate is this method" | "how close is it to a person" | "does this work on a real image" |

The middle column arrived with BSDS500's human segmentations and is the most
useful of the three, because its ceiling is measured rather than assumed.
Projects 24, 28 and 32 use it. Where a photograph has no annotation, no accuracy
number is quoted — it would be invented.

A fourth case is worth naming because it looks like the second and is not:
several projects define the truth *by construction* on a real photograph —
project 22 declares Otsu's binarisation to be the target, then adds noise and
measures what morphology removes. That is exact and self-consistent, and it is
not a claim about what the picture contains.

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

### BSDS500, the rest of it — 203 photographs

Projects 16 onward each draw **twelve images nobody else uses**, selected on a
*measured* axis rather than by eye (`tools/select_images.py --axis detail`,
`--axis texture`, and so on). At twelve per project that needs hundreds of
distinct photographs, which is why the pool moved wholesale to BSDS500: it has
500, and — decisively — it ships **five to seven human segmentations per
image**.

Those annotations are the only real ground truth in this repository. Everywhere
else the truth is generated (a flow field, a blur kernel) or defined by
construction (Otsu's binarisation declared to be the target). Projects 24, 28
and 32 score against what people actually drew, and measure the **human
ceiling** — one annotator against the others' consensus — which is around
F 0.90 and not 1.0.

Cached by `tools/fetch_images.py cache --set bsds` and
`--set bsds_gt`; the `.mat` annotations live outside the repository, in
`~/.cache/classical-cv-images/`, because 500 photographs are not a dependency of
this project and the handful each experiment keeps are.

`manifest.json` maps every committed filename back to its BSDS id by perceptual
hash. That is what `tools/check_image_reuse.py` uses to enforce the rule that no
photograph reaches two projects — filenames cannot do it, because every image is
renamed to something descriptive on the way in.

🚨 **The licence caveat above applies to all of these.** BSDS500 is distributed
by UC Berkeley for research and education and is **not** public domain. It is
the standard segmentation benchmark and appears in thousands of public
repositories; that is a reason it is low-risk, not a reason it is permissive. If
that is a problem for a given use, these are the images to replace.

<details>
<summary>All 203 names, with their BSDS ids</summary>

| Name | BSDS id | Name | BSDS id |
|---|---|---|---|
| `acacia_and_herd` | 253036 | `albatross_pair` | 103029 |
| `alpine_chalet_snow` | 61086 | `alpine_church` | 126007 |
| `angelfish_reef` | 306005 | `archer_dancer` | 217013 |
| `baboon_in_foliage` | 16052 | `barges_and_blocks` | 78098 |
| `bay_with_boats` | 68077 | `beached_dinghy` | 384022 |
| `bear_grass` | 100080 | `bear_on_ice` | 100007 |
| `bear_riverbank` | 100099 | `bear_tree_bark` | 100039 |
| `bears_on_hillside` | 309004 | `bench_bare_hedge` | 346016 |
| `bighorn_rock` | 304074 | `blossom_pavilion` | 95006 |
| `blue_footed_boobies` | 103070 | `boat_shed` | 140088 |
| `bobcat_rock` | 41085 | `bomber_overcast` | 3096 |
| `borobudur_stupas` | 217090 | `camel_at_sunset` | 271031 |
| `carved_figurine` | 71076 | `carved_mask_thatch` | 296058 |
| `castle_gatehouse` | 17067 | `caterpillar_on_stem` | 35028 |
| `cheetah_walking` | 134008 | `child_fur_hood` | 14092 |
| `child_on_water` | 26031 | `child_red_jumper` | 187029 |
| `climber_on_dome` | 14037 | `clouded_leopard` | 160067 |
| `clownfish_anemone` | 210088 | `collared_lizard` | 41096 |
| `conical_hat_worker` | 279005 | `coral_reef` | 101027 |
| `cougar_and_kitten` | 94095 | `couple_autumn_bank` | 365073 |
| `covered_wagons` | 216041 | `coyotes_in_haze` | 109053 |
| `crocodile_bank` | 130026 | `deer_and_fawn` | 317080 |
| `deer_bare_branches` | 77062 | `deer_in_brush` | 104010 |
| `deer_water` | 104022 | `desert_arch` | 295087 |
| `diver_dark_reef` | 45096 | `diver_sea_fans` | 156065 |
| `eagle_flat_sky` | 135069 | `eagle_in_flight` | 135037 |
| `egrets_in_thicket` | 311068 | `elder_headscarf` | 260081 |
| `elder_in_shawl` | 187083 | `elephant_grass` | 107014 |
| `elephant_herd` | 107072 | `elephant_pair` | 296059 |
| `elephant_waterhole` | 107045 | `elk_water` | 104055 |
| `family_by_van` | 102062 | `fighter_jet` | 10081 |
| `firefighter_debris` | 285079 | `firefighters_map` | 23084 |
| `fjord_harbour` | 219090 | `flag_and_parade` | 145086 |
| `florence_duomo` | 24004 | `fox_cubs` | 159008 |
| `gallery_visitors` | 128035 | `geese_and_goslings` | 43070 |
| `geisha_costume` | 65084 | `geisha_street` | 145053 |
| `geologist_rocks` | 89072 | `gilded_stupa` | 76053 |
| `giraffe` | 130014 | `girl_with_basin` | 23025 |
| `glass_pyramid` | 223061 | `glass_roof_trees` | 148026 |
| `glass_tower_tulips` | 86000 | `graffiti_wall` | 292066 |
| `gulls_on_ledge` | 163096 | `gunner_reenactor` | 243095 |
| `harbour_boat` | 118015 | `hawk_and_chick` | 268048 |
| `hawk_in_scrub` | 70011 | `hawk_on_stump` | 70090 |
| `hazy_ridges` | 55067 | `headland_lighthouse` | 228076 |
| `held_sunfish` | 185092 | `helicopter_dusk` | 179084 |
| `hilltop_ruin` | 20008 | `horse_blossom` | 291000 |
| `hotel_rossiya` | 274007 | `iceberg_cloud` | 176039 |
| `iceberg_watcher` | 188005 | `iguana_surf` | 103078 |
| `indian_corn` | 169012 | `lake_shrine` | 120003 |
| `leopard_in_tree` | 134049 | `lioness_savanna` | 105027 |
| `lionesses` | 105053 | `lions_plain` | 105019 |
| `lizard_on_gravel` | 87046 | `lone_palm_beach` | 46076 |
| `longtail_boats` | 81095 | `man_drying_fish` | 365072 |
| `man_floral_shirt` | 302022 | `man_fur_hat` | 15062 |
| `man_green_parka` | 230098 | `man_laying_paving` | 85048 |
| `man_striped_shirt` | 302008 | `man_yellow_barrels` | 65019 |
| `man_yellow_turban` | 189029 | `mare_and_foal` | 113009 |
| `mare_foal_meadow` | 113016 | `mare_foal_meadow_two` | 113044 |
| `marmot_boulder` | 41069 | `memorial_arch` | 148089 |
| `moated_chateau` | 102061 | `model_gloves` | 198087 |
| `model_red_black` | 198023 | `monitor_lizard_grass` | 130034 |
| `monk_at_table` | 145014 | `moonlit_pines` | 238011 |
| `morel_mushrooms` | 208078 | `ocelot_on_rock` | 326038 |
| `ostrich_head` | 66075 | `owl_in_grass` | 8143 |
| `ox_in_pasture` | 296007 | `palms_at_dusk` | 384089 |
| `paraglider_peak` | 60079 | `parasol_boat` | 147021 |
| `parasols_willows` | 65033 | `parthenon_columns` | 67079 |
| `penguin_dark_shore` | 106025 | `penguin_pebbles` | 106005 |
| `polar_bear_rail` | 183066 | `polar_bears_snow` | 183055 |
| `polo_riders` | 361010 | `porcupine_on_branch` | 347031 |
| `portrait_yellow` | 388006 | `potted_bonsai` | 353013 |
| `raked_zen_garden` | 86016 | `red_canoes` | 232076 |
| `regatta_spinnakers` | 172032 | `rhino_road` | 112090 |
| `rhinos_grass` | 112056 | `rider_and_herd` | 220075 |
| `roadrunner_rocks` | 196015 | `rocky_coast` | 117025 |
| `rocky_cove` | 144067 | `sampan_still_water` | 15088 |
| `scuba_diver_fish` | 156079 | `sea_shell_coral` | 12074 |
| `shark_shallows` | 86068 | `skiers_woods` | 277053 |
| `snake_coiled` | 87015 | `snake_on_sand` | 196073 |
| `snowboarder_pines` | 225017 | `sparkler_family` | 20069 |
| `sphinx_and_pyramid` | 161045 | `sprinter_start` | 153077 |
| `squirrel_rock` | 123057 | `station_platform` | 249021 |
| `statues_stairwell` | 24077 | `steam_train_viaduct` | 182053 |
| `stone_arch` | 118072 | `stone_archway` | 5096 |
| `stone_bridge_river` | 231015 | `stone_face_leaves` | 101084 |
| `stone_wellhead` | 92014 | `surfer_barrel` | 300091 |
| `temple_dragon` | 120093 | `three_astronauts` | 323016 |
| `three_owlets` | 42044 | `tiger_in_shade` | 108082 |
| `tiger_rocks` | 108069 | `tiger_wading` | 108041 |
| `tortoise_rock` | 103006 | `tower_and_spire` | 277095 |
| `train_on_viaduct` | 351093 | `tulip_beds` | 140055 |
| `two_beefeaters` | 376086 | `two_horses_field` | 28075 |
| `two_rhinos_scrub` | 112082 | `two_women_street` | 23050 |
| `waterfall_cliff` | 27059 | `whitewashed_chapel` | 118035 |
| `whitewashed_harbour` | 118020 | `windmills` | 118031 |
| `wolf_dark_wood` | 42078 | `wolf_on_snowline` | 167062 |
| `wolf_woods` | 109055 | `woman_and_child` | 189013 |
| `woman_black_beret` | 181018 | `woman_bundling_straw` | 15004 |
| `woman_by_tree` | 181091 | `woman_child_fur` | 14085 |
| `woman_hanbok` | 239007 | `woman_on_steps` | 181021 |
| `woman_wading` | 81066 | `woman_white_fence` | 388018 |
| `worker_with_pails` | 271035 |  |  |

</details>

### Source not recorded

These eight were added for project 02's subject-variety comparison and **their
exact source was not written down at the time.** That is a gap, and it is stated
here rather than backfilled with a guess:

`girl.jpg`, `dog.jpg`, `coffee_cup.jpg`, `woman_field.jpg`, `leopard.jpg`,
`man_camera.jpg`, `hiker.jpg`, `man_skyline.jpg` — all 700 × 525, no EXIF.

They are shown, never scored, and appear only in comparison figures. If their
licence cannot be established they should be replaced with Kodak or OpenCV
images, which carry explicit terms.
