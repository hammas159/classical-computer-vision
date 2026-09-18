"""Image loading, saving and dtype conversion.

The BGR/RGB rule
----------------
``cv2.imread`` returns **BGR**. ``matplotlib.pyplot.imshow`` expects **RGB**.
Getting this wrong makes every figure look blue and produces no error at all.

This module enforces one convention: **everything in this repo is RGB uint8**.
Conversion to BGR happens only at the moment a cv2 function needs it, and is
handled inside :func:`imwrite`. No project file should ever call
``cv2.cvtColor(..., COLOR_BGR2RGB)`` directly.

The coordinate rule
-------------------
``cv2`` takes points as ``(x, y)``. ``numpy`` indexes as ``[row, col]`` which is
``(y, x)``. They are transposed relative to each other and neither will complain.
Helpers here always document which convention they use.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage import data as skdata

# --------------------------------------------------------------------------- #
# dtype conversion
# --------------------------------------------------------------------------- #


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 [0, 255] -> float32 [0, 1]. A float image is passed through."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        return img.astype(np.float32, copy=False)
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] -> uint8 [0, 255], clipped. A uint8 image is passed through.

    Clipping matters: Retinex, sharpening and deconvolution all overshoot past
    1.0, and a silent wraparound turns a bright highlight into a black hole.
    """
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #


def imread(path: str | Path, gray: bool = False) -> np.ndarray:
    """Read an image from disk **as RGB uint8** (or 2-D grayscale if ``gray``).

    Raises FileNotFoundError rather than returning ``None``, which is what cv2
    does for a missing path and which turns into a confusing ``NoneType`` error
    several lines later.
    """
    path = Path(path)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR

    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read an image at {path}")
    if gray:
        return img

    # OpenCV 4.14 on this machine raises "Unknown C++ exception" from cvtColor
    # (and remap, and others) when the machine is under memory pressure -- which
    # is what a std::bad_alloc inside OpenCV surfaces as, since the binding
    # cannot name the C++ type. Measured: launching 40 python processes
    # back-to-back failed 11 times; the same code with the machine idle ran
    # 900 reads in one process and 30 processes in a row without a single
    # failure. It is an environment condition, not a property of any file, and
    # retrying inside the process does not help.
    #
    # Reversing the channel axis in numpy is exactly the same operation and
    # allocates through a different path, so the fallback is a real substitute
    # rather than a hope. `ascontiguousarray` because a reversed view upsets
    # some cv2 calls downstream. It cannot rescue a cv2 call elsewhere in a
    # pipeline -- if runs start failing in other places, the machine is out of
    # memory and that is the thing to fix.
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return np.ascontiguousarray(img[..., ::-1])


def imwrite(path: str | Path, img: np.ndarray) -> Path:
    """Write an **RGB** (or grayscale) image to disk, converting to BGR for cv2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_uint8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"cv2 failed to write {path}")
    return path


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> 2-D grayscale uint8. Already-gray input is passed through."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """2-D grayscale -> 3-channel RGB, so figures can stack mixed results."""
    if img.ndim == 3:
        return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# --------------------------------------------------------------------------- #
# sample images that ship with scikit-image (no download, ever)
# --------------------------------------------------------------------------- #

#: Names accepted by :func:`sample`, mapped to the ``skimage.data`` loader.
#: Every one of these is bundled with scikit-image, so no project in this repo
#: needs a network connection to produce a result.
_SAMPLES = {
    "astronaut": skdata.astronaut,  # colour portrait, faces, fine detail
    "camera": skdata.camera,        # grayscale, strong edges, the classic test
    "coins": skdata.coins,          # grayscale, touching objects -> watershed
    "chelsea": skdata.chelsea,      # colour cat, fur texture
    "coffee": skdata.coffee,        # colour, saturated, circular shapes
    "page": skdata.page,            # uneven illumination -> Otsu vs Sauvola
    "text": skdata.text,            # binarisation target
    "moon": skdata.moon,            # low contrast -> histogram work
    "brick": skdata.brick,          # texture
    "grass": skdata.grass,          # texture
    "gravel": skdata.gravel,        # texture
    "horse": skdata.horse,          # binary shape + ground-truth mask
    "rocket": skdata.rocket,        # colour, sky/ground split
    "immunohistochemistry": skdata.immunohistochemistry,
    "retina": skdata.retina,        # thin structures, punishing for IoU
    "cell": skdata.cell,            # microscopy, low contrast
}


def sample(name: str = "astronaut", gray: bool = False) -> np.ndarray:
    """Return a bundled sample image as RGB uint8 (or 2-D grayscale).

    ``skimage.data.horse`` is boolean and ``cell``/``camera`` are 2-D; both are
    normalised here so callers always get uint8 with a predictable shape.
    """
    if name not in _SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {sorted(_SAMPLES)}")
    img = _SAMPLES[name]()

    if img.dtype == bool:
        img = (~img).astype(np.uint8) * 255  # horse is True on the *background*
    elif img.dtype != np.uint8:
        img = to_uint8(img / max(img.max(), 1))

    if gray:
        return to_gray(img) if img.ndim == 3 else img
    return ensure_rgb(img)


#: Real photographs bundled with the repo, from OpenCV's BSD-licensed sample
#: data. They exist to show a pipeline working on an image nobody constructed
#: for it. They have **no ground truth**, so nothing that needs one may be
#: reported against them -- see assets/real/README.md.
REAL_PHOTOS = {
    "player": ("messi5.jpg", "a footballer on a pitch, real depth and a real crowd"),
    "newspaper": ("sudoku.png", "a newspaper page photographed at an angle"),
    "printed_text": ("imageTextN.png", "a page of clean printed text"),
    "defocused_text": ("text_defocus.jpg", "printed text, defocused"),
    "squirrel": ("squirrel.jpg", "a squirrel on a branch against bright foliage"),
    "butterfly": ("butterfly.jpg", "a butterfly on a leaf — a flat subject on a busy background"),
    "fruits": ("fruits.jpg", "a cut orange among other fruit — a still life with no face"),
    "sheet_music": ("sheet_music.png", "a page of musical notation — ruled staves, no prose"),
    "handwritten_digits": ("handwritten_digits.png", "a grid of handwritten digits"),
    "printed_text_rotated": ("printed_text_rotated.png", "printed prose, already rotated"),
    "motion_text": ("motion_text.jpg", "printed text smeared by camera motion"),
    # people, animals and objects — one subject each, for the projects whose
    # figures need genuine variety of shape rather than four of the same thing
    "girl": ("girl.jpg", "a girl in sunglasses holding flowers"),
    "dog": ("dog.jpg", "a black dog, head on, against planking"),
    "coffee_cup": ("coffee_cup.jpg", "a cappuccino on a wooden table"),
    "woman_field": ("woman_field.jpg", "a woman in a field, backlit"),
    "leopard": ("leopard.jpg", "a leopard walking a dirt track"),
    "man_camera": ("man_camera.jpg", "a man holding a camera, monochrome"),
    "hiker": ("hiker.jpg", "a hiker on a mountain ridge"),
    "man_skyline": ("man_skyline.jpg", "a man seated against a city skyline"),
    # outdoor scenes with real depth structure, for the projects whose physics
    # depends on distance — haze thickens with depth, so a flat wall and a
    # street receding to a vanishing point are different problems entirely
    "stone_house": ("stone_house.jpg", "a stone farmhouse wall, flat on — almost no depth range"),
    "motocross": ("motocross.jpg", "a row of motocross bikes on a dirt bank"),
    "moored_boat": ("moored_boat.jpg", "a wooden boat moored in turquoise shallows"),
    "old_street": ("old_street.jpg", "a timbered street receding to a vanishing point"),
    "sailboats": ("sailboats.jpg", "sailboats under coloured spinnakers"),
    "mountain_stream": ("mountain_stream.jpg", "a stream running out of snow-capped mountains"),
    "whitewater_raft": ("whitewater_raft.jpg", "a raft of people in whitewater"),
    "tropical_island": ("tropical_island.jpg", "a low island under towering cloud — sky-dominant"),
    "lighthouse_lawn": ("lighthouse_lawn.jpg", "a squat lighthouse behind a white fence"),
    "warplane": ("warplane.jpg", "a propeller fighter against a pale sky"),
    "lighthouse_cliff": ("lighthouse_cliff.jpg", "a lighthouse on a rocky headland"),
    "painted_chalet": ("painted_chalet.jpg", "a painted alpine chalet against a wooded hill"),
    # people, for project 05. An old-photo restorer is judged on faces: nobody
    # scans a landscape to save it, they scan the photograph of their family.
    # Skin tone is also the hardest thing to get right after a colour cast, so
    # these are the honest test as well as the evocative one.
    "boy_laughing": ("boy_laughing.jpg", "a boy laughing, close up"),
    "child_face_paint": ("child_face_paint.jpg", "a young child with painted face"),
    "girl_red_hat": ("girl_red_hat.jpg", "a girl in a red hat against pink cloth"),
    "woman_dress": ("woman_dress.jpg", "a woman in a grey dress, outdoors"),
    "young_woman": ("young_woman.jpg", "a young woman against a dark background"),
    "man_glasses": ("man_glasses.jpg", "a young man in glasses, indoor light"),
    "man_glasses_dark": ("man_glasses_dark.jpg", "the same man under much darker light"),
    "man_outdoors": ("man_outdoors.jpg", "a man outdoors against bright sky"),
    "couple_beach": ("couple_beach.jpg", "a couple walking a shoreline"),
    "two_men": ("two_men.jpg", "two men in suits, shallow depth of field"),
    "street_people": ("street_people.jpg", "people walking past a bus"),
    "two_men_indoor": ("two_men_indoor.jpg", "two men in a corridor, flat indoor light"),
    # scenes that are bright in DIFFERENT WAYS, for project 03. A low-light
    # method is judged on its tone distribution, so a pool of six images that
    # are all mid-key measures one thing six times: a saturated red door, a
    # white sail against water, dark fur, a pale office block and a night-ish
    # pier each put the histogram somewhere else entirely.
    "red_door": ("red_door.jpg", "a weathered red door, filling the frame"),
    "caps_row": ("caps_row.jpg", "a row of bright caps on a wall, hard shadows"),
    "window_flowers": ("window_flowers.jpg", "hibiscus against a shuttered window"),
    "sailboat_race": ("sailboat_race.jpg", "white sails against water — high key"),
    "boat_pier": ("boat_pier.jpg", "a boat beside a weathered pier"),
    "stone_statue": ("stone_statue.jpg", "a pale stone statue holding a gilded urn"),
    "red_barn": ("red_barn.jpg", "a red barn reflected in still water"),
    "parrots": ("parrots.jpg", "two macaws, saturated primaries"),
    "apple_desk": ("apple_desk.jpg", "an apple lit from one side — a dark still life"),
    "baboon": ("baboon.jpg", "a mandrill's face — dense fur detail"),
    "office_block": ("office_block.jpg", "a concrete office block, flat grey"),
    "clock_tower": ("clock_tower.jpg", "a clock tower against open sky"),
    # natural scenes for project 07. A copy-move forgery is only interesting
    # where the paste could plausibly hide, so these are chosen for the texture
    # AROUND the subject — ice, tarmac, pebbles, brush, grass — because that is
    # what a block matcher has to tell a real duplicate apart from.
    "bear_on_ice": ("bear_on_ice.jpg", "a bear on flat ice — almost no background texture"),
    "fighter_jet": ("fighter_jet.jpg", "a jet on tarmac — man-made straight edges"),
    "coral_reef": ("coral_reef.jpg", "a coral reef — dense self-similar texture"),
    "family_by_van": ("family_by_van.jpg", "a family beside a van at a riverbank"),
    "tortoise_rock": ("tortoise_rock.jpg", "a tortoise on broken rock"),
    "deer_in_brush": ("deer_in_brush.jpg", "a deer in winter brush — busy, low contrast"),
    "lioness_savanna": ("lioness_savanna.jpg", "a lioness on dry savanna"),
    "penguin_pebbles": ("penguin_pebbles.jpg", "a penguin on a pebble beach — repeating stones"),
    "elephant_herd": ("elephant_herd.jpg", "a herd of elephants — genuine repeated objects"),
    "tiger_rocks": ("tiger_rocks.jpg", "a tiger lying on rocks"),
    "rhinos_grass": ("rhinos_grass.jpg", "two rhinos on open grass"),
    "wolf_woods": ("wolf_woods.jpg", "a wolf howling in leaf litter"),
    # scenes for project 10. Seam carving is judged on what it has room to
    # remove, so these differ in SHAPE rather than subject: a tall thin animal,
    # a wide empty coastline, architecture with strong verticals a seam cannot
    # cross, and a small subject in a large removable background.
    "rocky_coast": ("rocky_coast.jpg", "a rocky coastline — wide, no single subject"),
    "harbour_boat": ("harbour_boat.jpg", "a boat and a lighthouse along a sea wall"),
    "windmills": ("windmills.jpg", "two windmills behind a stone wall"),
    "stone_arch": ("stone_arch.jpg", "a stone archway — strong verticals"),
    "lake_shrine": ("lake_shrine.jpg", "a shrine table against a wide lake"),
    "temple_dragon": ("temple_dragon.jpg", "a temple dragon against tower blocks"),
    "gallery_visitors": ("gallery_visitors.jpg", "people looking at paintings"),
    "giraffe": ("giraffe.jpg", "a giraffe — tall and thin in a wide frame"),
    "boat_shed": ("boat_shed.jpg", "a boat moored beside a shed"),
    "woman_child_fur": ("woman_child_fur.jpg", "a woman and child in fur hoods, close up"),
    "squirrel_rock": ("squirrel_rock.jpg", "a squirrel on a rock — small subject, large background"),
    "elephant_grass": ("elephant_grass.jpg", "an elephant in dry grass"),
    # scenes for project 13, ordered by DETAIL DENSITY — mean |Laplacian|, which
    # is the quantity denoising destroys. A pool of equally detailed images
    # cannot show the trade a filter makes, so these span 7 to 45, a 6x range.
    "bear_grass": ("bear_grass.jpg", "a bear in soft grass — detail 7"),
    "albatross_pair": ("albatross_pair.jpg", "two albatrosses, smooth plumage — detail 7"),
    "bear_riverbank": ("bear_riverbank.jpg", "a bear on a riverbank — detail 8"),
    "lionesses": ("lionesses.jpg", "lionesses in dry grass — detail 9"),
    "elk_water": ("elk_water.jpg", "an elk standing in water — detail 11"),
    "lions_plain": ("lions_plain.jpg", "lions on an open plain — detail 15"),
    "deer_water": ("deer_water.jpg", "deer among winter scrub — detail 17"),
    "iguana_surf": ("iguana_surf.jpg", "an iguana in breaking surf — detail 21"),
    "rhino_road": ("rhino_road.jpg", "a rhino on a gravel road — detail 23"),
    "bear_tree_bark": ("bear_tree_bark.jpg", "a bear against tree bark — detail 35"),
    "stone_face_leaves": ("stone_face_leaves.jpg", "carved stone among leaves — detail 45"),
    "moonlit_pines": ("moonlit_pines.jpg", "a moon over silhouetted pines — a night scene using a quarter of the tone range"),
    "desert_arch": ("desert_arch.jpg", "a juniper before a red sandstone arch under deep blue sky"),
    "two_horses_field": ("two_horses_field.jpg", "a dark and a pale horse in an open green field"),
    "ostrich_head": ("ostrich_head.jpg", "an ostrich head against dark foliage — one subject, shallow depth"),
    "mare_and_foal": ("mare_and_foal.jpg", "a mare and foal behind a white fence"),
    "horse_blossom": ("horse_blossom.jpg", "a chestnut horse under a flowering tree, bright midday"),
    "covered_wagons": ("covered_wagons.jpg", "covered wagons drawn across dry prairie grass"),
    "skiers_woods": ("skiers_woods.jpg", "two cross-country skiers in snowy woods — a high-key scene"),
    "penguin_dark_shore": ("penguin_dark_shore.jpg", "a penguin on a dark shore — a bright subject on a near-black ground"),
    "beached_dinghy": ("beached_dinghy.jpg", "a painted dinghy on a beach below a headland"),
    "alpine_chalet_snow": ("alpine_chalet_snow.jpg", "a chalet in snow with a figure in red"),
    "child_on_water": ("child_on_water.jpg", "a child silhouetted against sun-glittered water — extreme backlight"),
    "eagle_in_flight": ("eagle_in_flight.jpg", "a bald eagle in flight against blown-out sky — large smooth regions with no structure to track"),
    "man_striped_shirt": ("man_striped_shirt.jpg", "a portrait against black, striped collar — strong 1-D structure, the aperture problem made visible"),
    "angelfish_reef": ("angelfish_reef.jpg", "two angelfish over coral — dense fine texture everywhere"),
    "iceberg_watcher": ("iceberg_watcher.jpg", "a figure at a rail watching an iceberg — flat sea and sky either side"),
    "woman_by_tree": ("woman_by_tree.jpg", "a woman leaning against a lakeside tree"),
    "bighorn_rock": ("bighorn_rock.jpg", "a bighorn sheep on a rock ledge — subject and background share a texture"),
    "polar_bears_snow": ("polar_bears_snow.jpg", "two polar bears nose to nose on snow — white on white, almost no gradient"),
    "firefighters_map": ("firefighters_map.jpg", "two firefighters before a wall map — man-made lines in every direction"),
    "stone_archway": ("stone_archway.jpg", "a stone archway opening onto a field — a strong depth discontinuity"),
    "mare_foal_meadow": ("mare_foal_meadow.jpg", "a mare and foal in a flowering meadow"),
    "elephant_waterhole": ("elephant_waterhole.jpg", "an elephant walking past a waterhole at dusk"),
    "carved_mask_thatch": ("carved_mask_thatch.jpg", "a carved mask against thatch — the most textured frame in the pool"),
    "eagle_flat_sky": ("eagle_flat_sky.jpg", "an eagle against plain blue — almost no structure for a detector to find"),
    "regatta_spinnakers": ("regatta_spinnakers.jpg", "sailboats under spinnakers in flat light"),
    "scuba_diver_fish": ("scuba_diver_fish.jpg", "a diver among fish over a reef"),
    "portrait_yellow": ("portrait_yellow.jpg", "a portrait against a yellow dotted backdrop — repeated identical features"),
    "blue_footed_boobies": ("blue_footed_boobies.jpg", "two boobies on a nest of stones"),
    "polar_bear_rail": ("polar_bear_rail.jpg", "a polar bear leaning on a rail on ice"),
    "geologist_rocks": ("geologist_rocks.jpg", "a geologist working on a rock shelf"),
    "man_fur_hat": ("man_fur_hat.jpg", "a man in a fur hat against a stone wall"),
    "glass_pyramid": ("glass_pyramid.jpg", "a glass pyramid — dense man-made corners and repeated structure"),
    "bobcat_rock": ("bobcat_rock.jpg", "a bobcat on a rock in dry grass"),
    "snake_coiled": ("snake_coiled.jpg", "a coiled snake on sand — self-similar scales"),
    "monitor_lizard_grass": ("monitor_lizard_grass.jpg", "a monitor lizard in grass — the busiest frame in the pool"),
    "paraglider_peak": ("paraglider_peak.jpg", "a paraglider above a snow peak — a smooth sky with almost nothing to restore"),
    "ox_in_pasture": ("ox_in_pasture.jpg", "an ox in a hazy pasture under trees"),
    "surfer_barrel": ("surfer_barrel.jpg", "a surfer inside a breaking wave"),
    "three_owlets": ("three_owlets.jpg", "three owlets on a branch against black — fine down on a dark ground"),
    "geisha_costume": ("geisha_costume.jpg", "a performer in an embroidered kimono"),
    "child_fur_hood": ("child_fur_hood.jpg", "a laughing child inside a fur hood"),
    "tiger_wading": ("tiger_wading.jpg", "a tiger wading — stripes are directional structure a motion blur can hide in"),
    "woman_white_fence": ("woman_white_fence.jpg", "a woman leaning on a white fence among pines"),
    "castle_gatehouse": ("castle_gatehouse.jpg", "a stone gatehouse over water — hard man-made edges"),
    "man_laying_paving": ("man_laying_paving.jpg", "a man in a straw hat laying paving slabs"),
    "marmot_boulder": ("marmot_boulder.jpg", "a marmot between granite boulders"),
    "owl_in_grass": ("owl_in_grass.jpg", "an owl in dry grass — the finest detail in the pool"),
    "hawk_in_scrub": ("hawk_in_scrub.jpg", "a hawk flying through dry scrub — subject and background share one texture"),
    "helicopter_dusk": ("helicopter_dusk.jpg", "a helicopter on a beach at dusk — large smooth sky"),
    "rider_and_herd": ("rider_and_herd.jpg", "a rider before a herd of cattle"),
    "indian_corn": ("indian_corn.jpg", "rows of multicoloured corn — dense regular structure at the pixel scale"),
    "man_green_parka": ("man_green_parka.jpg", "a man in a hooded parka crouching on grass"),
    "longtail_boats": ("longtail_boats.jpg", "longtail boats moored in turquoise shallows"),
    "moated_chateau": ("moated_chateau.jpg", "a moated chateau with fine roof detail and a reflection"),
    "woman_wading": ("woman_wading.jpg", "a woman wading beside a boat in clear water"),
    "parasols_willows": ("parasols_willows.jpg", "figures with white parasols under willows"),
    "steam_train_viaduct": ("steam_train_viaduct.jpg", "a steam train crossing a viaduct"),
    "shark_shallows": ("shark_shallows.jpg", "a shark in shallow water over rippled sand"),
    "raked_zen_garden": ("raked_zen_garden.jpg", "a raked gravel garden — the finest repeating texture in the pool"),
    "bomber_overcast": ("bomber_overcast.jpg", "a propeller bomber against overcast — one clean silhouette, almost no other edge"),
    "lone_palm_beach": ("lone_palm_beach.jpg", "a single palm on a bright beach"),
    "hilltop_ruin": ("hilltop_ruin.jpg", "a ruin on a wooded hillside"),
    "climber_on_dome": ("climber_on_dome.jpg", "a climber on a granite dome above a valley"),
    "carved_figurine": ("carved_figurine.jpg", "a painted carved figurine among flowers"),
    "elder_headscarf": ("elder_headscarf.jpg", "an elderly man in a patterned headscarf"),
    "monk_at_table": ("monk_at_table.jpg", "a monk kneeling at a low table on a red mat"),
    "parthenon_columns": ("parthenon_columns.jpg", "the Parthenon colonnade — regular vertical structure"),
    "woman_bundling_straw": ("woman_bundling_straw.jpg", "a woman bundling straw"),
    "two_rhinos_scrub": ("two_rhinos_scrub.jpg", "two rhinos in dry scrub"),
    "diver_sea_fans": ("diver_sea_fans.jpg", "a diver among sea fans — thin branching structure"),
    "bench_bare_hedge": ("bench_bare_hedge.jpg", "a stone bench before a bare hedge — the finest twig structure in the pool"),
    "hazy_ridges": ("hazy_ridges.jpg", "receding hazy ridges — almost pure low frequency"),
    "whitewashed_chapel": ("whitewashed_chapel.jpg", "a whitewashed chapel with a red dome against deep blue"),
    "child_red_jumper": ("child_red_jumper.jpg", "a child in a red jumper on cobbles"),
    "geese_and_goslings": ("geese_and_goslings.jpg", "geese and goslings on rippled water"),
    "fjord_harbour": ("fjord_harbour.jpg", "a harbour below steep fjord peaks"),
    "woman_hanbok": ("woman_hanbok.jpg", "a woman in a hanbok before a painted pavilion"),
    "graffiti_wall": ("graffiti_wall.jpg", "figures at a graffiti-covered wall — dense mid-frequency texture"),
    "stone_wellhead": ("stone_wellhead.jpg", "a stone wellhead before a cottage"),
    "lizard_on_gravel": ("lizard_on_gravel.jpg", "a lizard on gravel — subject and ground share a spectrum"),
    "hawk_and_chick": ("hawk_and_chick.jpg", "a hawk and chick at a nest among bamboo"),
    "tower_and_spire": ("tower_and_spire.jpg", "a glass tower behind a stone spire — strong regular periodic structure"),
    "couple_autumn_bank": ("couple_autumn_bank.jpg", "two people on an autumn riverbank — the busiest frame in the pool"),
    "wolf_on_snowline": ("wolf_on_snowline.jpg", "a wolf on a snow ridge against black forest — two flat regions and one small subject"),
    "tiger_in_shade": ("tiger_in_shade.jpg", "a tiger lying in dappled shade"),
    "train_on_viaduct": ("train_on_viaduct.jpg", "a train crossing a viaduct among trees"),
    "caterpillar_on_stem": ("caterpillar_on_stem.jpg", "a caterpillar on a green stem — subject and background share a hue"),
    "gunner_reenactor": ("gunner_reenactor.jpg", "a re-enactor seated beside a cannon"),
    "morel_mushrooms": ("morel_mushrooms.jpg", "three morels on leaf litter"),
    "woman_and_child": ("woman_and_child.jpg", "a woman carrying a child on a dirt path"),
    "fox_cubs": ("fox_cubs.jpg", "three fox cubs at a den mouth"),
    "memorial_arch": ("memorial_arch.jpg", "a memorial arch on a city street"),
    "florence_duomo": ("florence_duomo.jpg", "the Duomo above a tiled roofscape"),
    "two_beefeaters": ("two_beefeaters.jpg", "two guards in scarlet and black uniform"),
    "man_yellow_barrels": ("man_yellow_barrels.jpg", "a man before stacked yellow ceremonial barrels — the most saturated frame here"),
    "archer_dancer": ("archer_dancer.jpg", "a dancer with a bow on a dark stage — very little of the frame carries information"),
    "sea_shell_coral": ("sea_shell_coral.jpg", "a spired shell on coral"),
    "potted_bonsai": ("potted_bonsai.jpg", "an orange-flowered bonsai in a bowl"),
    "porcupine_on_branch": ("porcupine_on_branch.jpg", "a porcupine on a branch against a pale slope"),
    "roadrunner_rocks": ("roadrunner_rocks.jpg", "a roadrunner among rocks and grass"),
    "gilded_stupa": ("gilded_stupa.jpg", "a gilded stupa over a wide plain — repeated architecture"),
    "headland_lighthouse": ("headland_lighthouse.jpg", "a lighthouse and cottages on a rocky headland"),
    "station_platform": ("station_platform.jpg", "travellers waiting on a tiled platform"),
    "leopard_in_tree": ("leopard_in_tree.jpg", "a leopard draped along a branch"),
    "snowboarder_pines": ("snowboarder_pines.jpg", "a snowboarder above frosted pines"),
    "sparkler_family": ("sparkler_family.jpg", "a family with a sparkler at dusk"),
    "man_drying_fish": ("man_drying_fish.jpg", "a man beside hanging dried fish — the most information-dense frame here"),
    "wolf_dark_wood": ("wolf_dark_wood.jpg", "a wolf on snow against black woodland — the darkest frame in the pool"),
    "gulls_on_ledge": ("gulls_on_ledge.jpg", "gulls nesting on a layered rock ledge"),
    "conical_hat_worker": ("conical_hat_worker.jpg", "a worker in a wide conical hat"),
    "egrets_in_thicket": ("egrets_in_thicket.jpg", "two egrets in dense green thicket"),
    "elephant_pair": ("elephant_pair.jpg", "two elephants on dry grassland"),
    "man_yellow_turban": ("man_yellow_turban.jpg", "a man in a yellow turban with a gourd instrument"),
    "flag_and_parade": ("flag_and_parade.jpg", "a striped flag before a parade ground"),
    "iceberg_cloud": ("iceberg_cloud.jpg", "a towering cloud over ice — near-white throughout"),
    "statues_stairwell": ("statues_stairwell.jpg", "pale statues on a stairwell before stained glass"),
    "stone_bridge_river": ("stone_bridge_river.jpg", "a stone bridge over a shallow river"),
    "sampan_still_water": ("sampan_still_water.jpg", "a sampan on flat green water"),
    "woman_on_steps": ("woman_on_steps.jpg", "a woman seated on bright white steps — the brightest frame here"),
    "sprinter_start": ("sprinter_start.jpg", "a sprinter pushing off the blocks"),
    "girl_with_basin": ("girl_with_basin.jpg", "a girl holding a pink basin against a plaster wall"),
    "model_gloves": ("model_gloves.jpg", "a model in long gloves against a plain ground — large flat areas"),
    "worker_with_pails": ("worker_with_pails.jpg", "a worker carrying pails in a dusty yard"),
    "polo_riders": ("polo_riders.jpg", "two polo riders on a mown field"),
    "borobudur_stupas": ("borobudur_stupas.jpg", "latticed stone stupas — dense regular geometry"),
    "coyotes_in_haze": ("coyotes_in_haze.jpg", "two coyotes in hazy woodland"),
    "crocodile_bank": ("crocodile_bank.jpg", "a crocodile on a dry bank"),
    "bay_with_boats": ("bay_with_boats.jpg", "boats in a bay below a green headland"),
    "mare_foal_meadow_two": ("mare_foal_meadow_two.jpg", "a chestnut mare and foal in a flowering meadow"),
    "tulip_beds": ("tulip_beds.jpg", "banked tulip beds beside water — saturated colour at every scale"),
    "snake_on_sand": ("snake_on_sand.jpg", "a sidewinder on rippled sand — the finest uniform texture here"),
    "acacia_and_herd": ("acacia_and_herd.jpg", "an acacia and a distant herd on a plain — one silhouette and open sky"),
    "clownfish_anemone": ("clownfish_anemone.jpg", "a clownfish among anemone tentacles"),
    "three_astronauts": ("three_astronauts.jpg", "three astronauts in white suits against black"),
    "alpine_church": ("alpine_church.jpg", "a twin-towered church below snowy peaks"),
    "man_floral_shirt": ("man_floral_shirt.jpg", "a man in a patterned floral shirt"),
    "hawk_on_stump": ("hawk_on_stump.jpg", "a hawk perched on a stump"),
    "baboon_in_foliage": ("baboon_in_foliage.jpg", "a baboon half-hidden in dense foliage"),
    "deer_and_fawn": ("deer_and_fawn.jpg", "a deer nursing a fawn beside a trunk"),
    "firefighter_debris": ("firefighter_debris.jpg", "a firefighter working in debris"),
    "blossom_pavilion": ("blossom_pavilion.jpg", "a pavilion framed by blossom and tulips"),
    "deer_bare_branches": ("deer_bare_branches.jpg", "a deer among bare winter branches"),
    "bears_on_hillside": ("bears_on_hillside.jpg", "a bear and cubs on a grassy hillside — the busiest frame here"),
    "diver_dark_reef": ("diver_dark_reef.jpg", "a diver over a dark reef — detail buried in shadow, the closest analogue here to the book's bone scan"),
    "red_canoes": ("red_canoes.jpg", "red canoes drawn up on a shore"),
    "parasol_boat": ("parasol_boat.jpg", "a boat under a parasol on dark water"),
    "model_red_black": ("model_red_black.jpg", "a model in red and black against a chequered ground"),
    "sphinx_and_pyramid": ("sphinx_and_pyramid.jpg", "the Sphinx before a pyramid in flat desert light"),
    "elder_in_shawl": ("elder_in_shawl.jpg", "an elderly man in a shawl, shaded"),
    "whitewashed_harbour": ("whitewashed_harbour.jpg", "whitewashed houses above a harbour"),
    "glass_tower_tulips": ("glass_tower_tulips.jpg", "a glass tower behind red tulips"),
    "waterfall_cliff": ("waterfall_cliff.jpg", "a waterfall down a shaded cliff"),
    "cougar_and_kitten": ("cougar_and_kitten.jpg", "a cougar and kitten in dry scrub — low contrast throughout"),
    "geisha_street": ("geisha_street.jpg", "a woman in kimono in a wooden street"),
    "collared_lizard": ("collared_lizard.jpg", "a collared lizard on a rock"),
    "camel_at_sunset": ("camel_at_sunset.jpg", "a camel on a flat horizon at sunset — one long straight line and little else"),
    "woman_black_beret": ("woman_black_beret.jpg", "a woman in a black beret and white collar"),
    "held_sunfish": ("held_sunfish.jpg", "a sunfish held in two hands"),
    "clouded_leopard": ("clouded_leopard.jpg", "a clouded leopard on a stony shore"),
    "palms_at_dusk": ("palms_at_dusk.jpg", "palms and figures on a beach at dusk — strong vertical trunks"),
    "cheetah_walking": ("cheetah_walking.jpg", "a cheetah crossing dry grass"),
    "two_women_street": ("two_women_street.jpg", "two women walking a busy street"),
    "rocky_cove": ("rocky_cove.jpg", "a rocky cove with pines above clear water"),
    "glass_roof_trees": ("glass_roof_trees.jpg", "a glazed roof among trees — dense man-made straight lines"),
    "barges_and_blocks": ("barges_and_blocks.jpg", "barges moored before apartment blocks — long horizontal courses"),
    "hotel_rossiya": ("hotel_rossiya.jpg", "a hotel facade behind a bare tree — a grid of windows"),
    "ocelot_on_rock": ("ocelot_on_rock.jpg", "an ocelot on a fissured rock face — the busiest frame here"),
    "tree_bark_ridged": ("tree_bark_ridged.jpg", "deeply ridged tree bark - coarse, irregular, no dominant orientation"),
    "herringbone_weave": ("herringbone_weave.jpg", "fine herringbone cloth - regular, two interleaved diagonal orientations"),
    "wood_grain": ("wood_grain.jpg", "planed wood grain - strongly directional, low contrast"),
    "brick_paving": ("brick_paving.jpg", "vertically stacked paving bricks - a periodic lattice with mortar lines"),
    "thatch_fibres": ("thatch_fibres.jpg", "thatched fibres - one strong diagonal orientation, uneven illumination"),
    "coarse_stucco": ("coarse_stucco.jpg", "coarse stucco render - high contrast, blobby, no orientation"),
    "dry_straw": ("dry_straw.jpg", "dry straw - long thin strands, one orientation, high dynamic range"),
    "sand_ripples": ("sand_ripples.jpg", "wind ripples in sand - the lowest-contrast texture in the set"),
    "stipple_plaster": ("stipple_plaster.jpg", "stippled plaster - fine isotropic bumps, bright"),
    "perforated_metal": ("perforated_metal.jpg", "perforated metal sheet - an exactly periodic hexagonal dot lattice"),
    "crushed_gravel": ("crushed_gravel.jpg", "crushed white gravel - large high-contrast stones, no orientation"),
    "packed_cobbles": ("packed_cobbles.jpg", "packed round cobbles - a cellular texture of blobs separated by dark grout"),
    "penguin_on_pebbles": ("penguin_on_pebbles.jpg", "a penguin on grey pebbles - the least colourful photograph in the pool"),
    "teotihuacan_pyramids": ("teotihuacan_pyramids.jpg", "stone pyramids under a pale sky - muted earth tones"),
    "helicopter_and_pilot": ("helicopter_and_pilot.jpg", "a white and red helicopter with its pilot on tarmac"),
    "milking_the_cow": ("milking_the_cow.jpg", "a cream cow being milked outside a stone cottage with a red roof"),
    "black_bear_wading": ("black_bear_wading.jpg", "a black bear in water among green reeds - dark subject, saturated surround"),
    "ducks_in_reeds": ("ducks_in_reeds.jpg", "two ducks among golden reeds reflected in dark water"),
    "fox_and_daisies": ("fox_and_daisies.jpg", "a fox on a log beside white daisies, against near-black shade"),
    "beached_boats": ("beached_boats.jpg", "pale wooden boats hauled up on a shore under a blue sky"),
    "damselfly_on_leaf": ("damselfly_on_leaf.jpg", "an iridescent blue-green damselfly on a green leaf - close to isoluminant"),
    "runners_on_track": ("runners_on_track.jpg", "athletes in saturated red, blue and yellow kit mid-race"),
    "roller_coaster_loop": ("roller_coaster_loop.jpg", "a yellow and red roller coaster loop against a deep blue sky - the most colourful in the pool"),
    "leopard_along_branch": ("leopard_along_branch.jpg", "a leopard lying along a branch - yellow fur against green leaves"),
    "greek_amphora": ("greek_amphora.jpg", "a decorated amphora with two handles - an outline with holes between handle and body"),
    "collie_standing": ("collie_standing.jpg", "a rough collie standing in profile - the most articulated silhouette in the pool"),
    "flatfish_on_sand": ("flatfish_on_sand.jpg", "a spotted flatfish on the sea floor - a smooth oval with a tail"),
    "man_in_a_fez": ("man_in_a_fez.jpg", "a man wearing a fez, head and shoulders against a plain wall"),
    "made_up_face": ("made_up_face.jpg", "a made-up face resting on hands - a segmentation that fragments into fifteen contours"),
    "roman_amphitheatre": ("roman_amphitheatre.jpg", "an oval Roman amphitheatre seen from above"),
    "green_mountain_ridge": ("green_mountain_ridge.jpg", "a green mountain ridge above a valley - a long jagged boundary"),
    "basket_of_grain": ("basket_of_grain.jpg", "a basket of yellow grain - a near-circular blob"),
    "buttressed_trunk": ("buttressed_trunk.jpg", "a buttressed tree trunk lit green - a straight-sided polygon"),
    "brain_coral": ("brain_coral.jpg", "a dome of brain coral with fish above it"),
    "lizard_on_a_leaf": ("lizard_on_a_leaf.jpg", "a lizard on a broad green leaf - a leaf outline with a lizard-shaped hole"),
    "spotted_fish_head": ("spotted_fish_head.jpg", "the head of a spotted fish, an angular wedge"),
    "hippo_in_green_water": ("hippo_in_green_water.jpg", "a hippo in algae-green water - the lowest-entropy photograph in the pool"),
    "porcupine_on_a_branch": ("porcupine_on_a_branch.jpg", "a porcupine on a branch against dark foliage"),
    "leopard_in_bare_tree": ("leopard_in_bare_tree.jpg", "a leopard resting in a bare tree against a pale sky"),
    "cricket_on_the_green": ("cricket_on_the_green.jpg", "a cricket match on a green - many near-identical white figures"),
    "weaver_at_her_loom": ("weaver_at_her_loom.jpg", "a weaver in a red striped dress among striped cloth"),
    "bugling_elk": ("bugling_elk.jpg", "an elk bugling in dry grass"),
    "elephants_and_grooms": ("elephants_and_grooms.jpg", "caparisoned elephants with their grooms on a field"),
    "boy_in_a_wide_hat": ("boy_in_a_wide_hat.jpg", "a boy under a very wide straw hat"),
    "boy_with_a_fish_trap": ("boy_with_a_fish_trap.jpg", "a boy holding a woven fish trap - a strongly repetitive weave"),
    "two_at_a_railing": ("two_at_a_railing.jpg", "two people with glasses beside a white railing"),
    "street_piper": ("street_piper.jpg", "a piper playing on a pavement"),
    "monks_at_a_noticeboard": ("monks_at_a_noticeboard.jpg", "monks in orange robes reading a grid of notices - the highest-entropy photograph here"),
    "mushers_and_husky": ("mushers_and_husky.jpg", "two mushers and a husky in snow - the scene mean sits 0.6 degrees from grey"),
    "gentoo_on_shingle": ("gentoo_on_shingle.jpg", "a penguin on grey shingle"),
    "beach_baseball": ("beach_baseball.jpg", "a baseball game on pale sand"),
    "desert_dune_ripples": ("desert_dune_ripples.jpg", "rippled desert dunes under a wide sky"),
    "louvre_pyramid": ("louvre_pyramid.jpg", "the glass pyramid of the Louvre against cloud"),
    "cyclists_on_a_lane": ("cyclists_on_a_lane.jpg", "three cyclists on a lane between green verges"),
    "tiger_in_the_shade": ("tiger_in_the_shade.jpg", "a tiger lying in dappled green shade"),
    "swallowtail_on_phlox": ("swallowtail_on_phlox.jpg", "a yellow swallowtail butterfly on white phlox"),
    "leopard_in_dry_grass": ("leopard_in_dry_grass.jpg", "a leopard walking through dry golden grass"),
    "lakeside_verandah": ("lakeside_verandah.jpg", "a pine verandah looking out over a lake"),
    "spotted_cat_on_a_log": ("spotted_cat_on_a_log.jpg", "a clouded leopard lying along a mossy log"),
    "red_chrysanthemums": ("red_chrysanthemums.jpg", "two red chrysanthemums filling the frame - 29 degrees from grey, where grey-world cannot work"),
    "ladybird_on_a_leaf": ("ladybird_on_a_leaf.jpg", "a ladybird on a smooth green leaf - the least detailed photograph in the pool"),
    "swallow_tailed_gulls": ("swallow_tailed_gulls.jpg", "two gulls on grey boulders"),
    "carved_stone_relief": ("carved_stone_relief.jpg", "a weathered stone relief in flat light"),
    "three_schoolchildren": ("three_schoolchildren.jpg", "three children posed together, fine hair detail"),
    "toadstool_in_moss": ("toadstool_in_moss.jpg", "a toadstool among moss and ferns"),
    "skier_on_a_slope": ("skier_on_a_slope.jpg", "a skier on bright snow"),
    "two_at_a_wagon": ("two_at_a_wagon.jpg", "two people in front of a wooden shelter and a wagon"),
    "deer_in_bare_woods": ("deer_in_bare_woods.jpg", "a deer among bare winter trees - dense fine branches"),
    "otters_on_gravel": ("otters_on_gravel.jpg", "two otters on a gravel bank"),
    "foxes_under_a_ledge": ("foxes_under_a_ledge.jpg", "two foxes sheltering under a rock ledge"),
    "mayan_stone_carving": ("mayan_stone_carving.jpg", "a deeply carved Mayan stone panel - the most detailed photograph here"),
    "pintail_at_dusk": ("pintail_at_dusk.jpg", "a pintail duck standing in still water at dusk - smooth water, fine feather edges"),
    "firewalkers_at_night": ("firewalkers_at_night.jpg", "figures walking through flame at night - the darkest photograph in the pool"),
    "young_monks_crowding": ("young_monks_crowding.jpg", "young monks in red robes crowded together in shade"),
    "golden_pavilion": ("golden_pavilion.jpg", "the golden pavilion reflected in a dark pond"),
    "clapboard_houses": ("clapboard_houses.jpg", "white clapboard houses along a green lane"),
    "three_girls_by_hay": ("three_girls_by_hay.jpg", "three girls in hats in front of stacked hay"),
    "bighorn_ram": ("bighorn_ram.jpg", "a bighorn ram against pale rock"),
    "chipmunk_on_granite": ("chipmunk_on_granite.jpg", "a chipmunk on bright granite"),
    "horses_in_a_meadow": ("horses_in_a_meadow.jpg", "three horses in a bright meadow"),
    "woman_by_a_wall": ("woman_by_a_wall.jpg", "a woman leaning against a sunlit stone wall"),
    "monk_under_a_tree": ("monk_under_a_tree.jpg", "a monk in saffron robes in dappled sunlight"),
    "aircrew_on_tarmac": ("aircrew_on_tarmac.jpg", "aircrew walking across bright tarmac - the brightest photograph here"),
    "market_fruit_stall": ("market_fruit_stall.jpg", "a woman and child at a fruit stall - saturated yellows and greens at mid brightness"),
    "giraffe_head_on": ("giraffe_head_on.jpg", "a giraffe looking straight at the camera - the smoothest scene in the pool"),
    "elk_in_long_grass": ("elk_in_long_grass.jpg", "an elk standing in long dry grass"),
    "husky_puppies": ("husky_puppies.jpg", "three husky puppies on grass"),
    "long_jetty": ("long_jetty.jpg", "a wooden jetty running out over still water"),
    "yacht_and_bridge": ("yacht_and_bridge.jpg", "a yacht moored below a steel bridge"),
    "marmot_on_rock": ("marmot_on_rock.jpg", "a marmot asleep on pale rock"),
    "biwa_player": ("biwa_player.jpg", "a musician with a biwa among papers"),
    "sled_dogs_on_ice": ("sled_dogs_on_ice.jpg", "sled dogs and figures on snow"),
    "cannon_on_cobbles": ("cannon_on_cobbles.jpg", "a cannon on a cobbled square"),
    "layered_sandstone": ("layered_sandstone.jpg", "layered sandstone steps under a blue sky"),
    "stone_guardian": ("stone_guardian.jpg", "a carved stone figure in dense woodland"),
    "snake_on_needles": ("snake_on_needles.jpg", "a snake among pine needles and roots - the busiest scene here"),
    "bird_in_a_meadow": ("bird_in_a_meadow.jpg", "a small bird in a green meadow - the narrowest tonal range in the pool"),
    "tent_on_the_ice": ("tent_on_the_ice.jpg", "a tent and figure on blue polar ice"),
    "zebra_in_grass": ("zebra_in_grass.jpg", "a zebra standing in dry grass"),
    "carved_boat_houses": ("carved_boat_houses.jpg", "carved wooden houses with upswept roofs"),
    "soldier_and_child": ("soldier_and_child.jpg", "a soldier crouching to greet a small child"),
    "ploughing_with_oxen": ("ploughing_with_oxen.jpg", "a farmer ploughing dark soil behind two oxen"),
    "skier_mid_air": ("skier_mid_air.jpg", "a skier in red and yellow against blue ice"),
    "spear_fisher": ("spear_fisher.jpg", "a figure with a spear standing in shallow water"),
    "bobcat_and_daisies": ("bobcat_and_daisies.jpg", "a bobcat on a log among yellow daisies"),
    "snowshoes_on_snow": ("snowshoes_on_snow.jpg", "a pair of snowshoes in snow below mountains - the widest tonal range here"),
    "wallaby_in_scrub": ("wallaby_in_scrub.jpg", "a wallaby among dry scrub - warm, low contrast"),
    "stone_viaduct": ("stone_viaduct.jpg", "a stone railway viaduct below a snow-dusted fell"),
    "packhorse_bridge": ("packhorse_bridge.jpg", "a mossy packhorse bridge in dense woodland - the least sparse image in the pool"),
    "black_panther": ("black_panther.jpg", "a black panther resting in dappled undergrowth"),
    "lone_tree_on_a_hill": ("lone_tree_on_a_hill.jpg", "a single tree on a green headland above a bay"),
    "church_spire": ("church_spire.jpg", "a stone church framed by overhanging branches"),
    "warthogs_drinking": ("warthogs_drinking.jpg", "three warthogs kneeling to drink"),
    "wallaby_and_joey": ("wallaby_and_joey.jpg", "a wallaby with a joey in the pouch on leaf litter"),
    "warbler_at_the_nest": ("warbler_at_the_nest.jpg", "a yellow warbler over its nest"),
    "four_children_on_a_wall": ("four_children_on_a_wall.jpg", "four children sitting on a low wall"),
    "cormorants_nesting": ("cormorants_nesting.jpg", "cormorants on a pale rock ledge"),
    "woman_among_roses": ("woman_among_roses.jpg", "a woman among pink roses"),
    "taj_mahal_reflected": ("taj_mahal_reflected.jpg", "the Taj Mahal mirrored in its reflecting pool"),
    "hawk_on_a_branch": ("hawk_on_a_branch.jpg", "a hawk on a bare branch against a plain sky - the sparsest image here"),
    "lobsters_and_wine": ("lobsters_and_wine.jpg", "cooked lobsters and a wine bottle on a quay - the most colour-distinct object in the pool"),
    "stacked_timber": ("stacked_timber.jpg", "a worker in blue among stacked orange timber"),
    "anteater_at_sunset": ("anteater_at_sunset.jpg", "an anteater silhouetted against an orange sky"),
    "kabuki_pair": ("kabuki_pair.jpg", "two performers, one in a yellow kimono"),
    "tomato_stall": ("tomato_stall.jpg", "a crate of red tomatoes at a market"),
    "kalmar_castle": ("kalmar_castle.jpg", "a sandstone castle with green copper domes"),
    "yellow_trousers": ("yellow_trousers.jpg", "a figure in yellow trousers walking a dog"),
    "woman_in_blue_dress": ("woman_in_blue_dress.jpg", "a woman in a long blue dress before a white house"),
    "red_robed_figures": ("red_robed_figures.jpg", "two figures in red robes winnowing grain"),
    "red_sports_car": ("red_sports_car.jpg", "a red sports car on dark tarmac"),
    "westminster_pair": ("westminster_pair.jpg", "two figures in dark coats in front of Big Ben"),
    "green_field_worker": ("green_field_worker.jpg", "a worker in a green field"),
    "sea_stacks": ("sea_stacks.jpg", "sea stacks in grey surf - no saturated edges at all, the control"),
    "kangaroo_resting": ("kangaroo_resting.jpg", "a kangaroo lying on grass"),
    "laden_donkey": ("laden_donkey.jpg", "a laden donkey beside a stone byre"),
    "two_in_headscarves": ("two_in_headscarves.jpg", "two men in white headscarves against a dark ground"),
    "bison_in_snow": ("bison_in_snow.jpg", "a bison on a snowy ridge"),
    "mono_lake_tufa": ("mono_lake_tufa.jpg", "tufa towers reflected in a violet lake at dusk"),
    "drying_racks": ("drying_racks.jpg", "drying racks on a pale valley floor"),
    "diver_and_coral": ("diver_and_coral.jpg", "a diver above pink soft coral"),
    "lynx_on_birch": ("lynx_on_birch.jpg", "a lynx kitten climbing a birch"),
    "villa_on_the_lake": ("villa_on_the_lake.jpg", "an ochre villa above a lake"),
    "sandstone_ladder": ("sandstone_ladder.jpg", "a ladder in a red sandstone cleft"),
    "flounder_on_gravel": ("flounder_on_gravel.jpg", "a flounder on coloured gravel - saturated texture across the whole frame"),
    "bomber_over_cloud": ("bomber_over_cloud.jpg", "a bomber against open sky - almost nothing for a barcode localiser to latch onto"),
    "turquoise_lake": ("turquoise_lake.jpg", "a turquoise lake below forested slopes"),
    "polar_bears_playing": ("polar_bears_playing.jpg", "two polar bears wrestling in dry scrub"),
    "zebra_herd": ("zebra_herd.jpg", "a group of zebras - vertical stripes, the classic barcode false positive"),
    "cougar_among_birches": ("cougar_among_birches.jpg", "a cougar between pale birch trunks"),
    "canoe_on_the_lake": ("canoe_on_the_lake.jpg", "a figure paddling a canoe on still water"),
    "trocadero_statue": ("trocadero_statue.jpg", "a gilded statue with the Eiffel Tower behind"),
    "skiff_in_weed": ("skiff_in_weed.jpg", "a beached skiff among green weed"),
    "saguaro_blossom": ("saguaro_blossom.jpg", "saguaro cactus in flower - dense vertical ribbing"),
    "coiled_rope": ("coiled_rope.jpg", "coiled mooring rope on a deck - strong repeating stripes"),
    "giraffes_drinking": ("giraffes_drinking.jpg", "giraffes bending to drink"),
    "buffalo_in_the_river": ("buffalo_in_the_river.jpg", "a buffalo being washed in a river - the most barcode-like background here"),
    "two_women_in_headdress": ("two_women_in_headdress.jpg", "two women facing the camera in embroidered headdresses; both faces frontal and well lit"),
    "girl_with_tulips": ("girl_with_tulips.jpg", "a girl in a lace cap holding red tulips against a blue sky"),
    "woman_in_red_scarf": ("woman_in_red_scarf.jpg", "studio portrait of a woman wearing a red scarf, with flowers below"),
    "girl_in_pink_shirt": ("girl_in_pink_shirt.jpg", "studio portrait of a young woman in a pink shirt against a pale wall"),
    "two_firefighters": ("two_firefighters.jpg", "two firefighters in helmets in front of a red fire engine"),
    "woman_with_curly_hair": ("woman_with_curly_hair.jpg", "a woman with dark curly hair and red lipstick on a leopard-print coat"),
    "orange_lichen_on_rock": ("orange_lichen_on_rock.jpg", "orange lichen spreading over grey rock, dense with small round patches"),
    "children_carrying_pots": ("children_carrying_pots.jpg", "two children carrying clay pots on their heads over red earth"),
    "feather_duster_worms": ("feather_duster_worms.jpg", "two feather-duster worms on a red coral wall studded with dark holes"),
    "runners_in_the_stadium": ("runners_in_the_stadium.jpg", "two runners on a track in front of a packed stadium crowd"),
    "scattered_sweets": ("scattered_sweets.jpg", "coloured sugar-shelled sweets scattered on a pale surface; dozens of small round red discs"),
    "red_brick_house": ("red_brick_house.jpg", "the gable end of a red brick house against a blue sky"),
    "seal_on_grey_ice": ("seal_on_grey_ice.jpg", "a seal lying on grey ice in flat light; almost monochrome, one luminance level per colour"),
    "farmland_from_the_air": ("farmland_from_the_air.jpg", "aerial view of blue-grey farmland divided into rectangular fields"),
    "wine_bottles_in_a_rack": ("wine_bottles_in_a_rack.jpg", "rows of dark wine bottles stacked in a pale concrete rack"),
    "llama_at_a_stone_wall": ("llama_at_a_stone_wall.jpg", "a brown llama standing against an Inca stone wall"),
    "tree_against_tropical_sky": ("tree_against_tropical_sky.jpg", "a dark green tree against a saturated blue sky with white cloud"),
    "otter_on_a_log": ("otter_on_a_log.jpg", "a wet otter on a mossy log by a river bank"),
    "chicks_in_a_nest": ("chicks_in_a_nest.jpg", "two pale downy chicks in a twig nest against green reeds"),
    "lizard_on_pebbles": ("lizard_on_pebbles.jpg", "a banded lizard on a bed of pale pebbles of many different colours"),
    "bears_at_the_water": ("bears_at_the_water.jpg", "a brown bear and two cubs walking along a turquoise lake shore"),
    "glacier_cave_mouth": ("glacier_cave_mouth.jpg", "the mouth of an ice cave framing a blue sky and a distant mountain"),
    "soldier_on_the_grass": ("soldier_on_the_grass.jpg", "a soldier in khaki crouching on green grass before a grey rock face"),
    "woman_in_a_red_top": ("woman_in_a_red_top.jpg", "a woman in a red top beside a lamp post, with a bed of red and yellow tulips"),
    "whitewashed_bell_tower": ("whitewashed_bell_tower.jpg", "a whitewashed church bell tower against a clear blue sky; almost no fine texture"),
    "mossy_boulders_in_a_valley": ("mossy_boulders_in_a_valley.jpg", "large moss-covered boulders on a green valley floor"),
    "golfer_by_the_sea": ("golfer_by_the_sea.jpg", "a golfer mid-swing on a coastal tee with a ruined tower behind"),
    "wall_across_the_hills": ("wall_across_the_hills.jpg", "a dry stone wall running away over open hills under grey cloud"),
    "partridge_on_gravel": ("partridge_on_gravel.jpg", "a barred partridge feeding on gravel and short grass"),
    "two_jackals": ("two_jackals.jpg", "two jackals standing on bare dusty ground"),
    "race_cars_on_a_bend": ("race_cars_on_a_bend.jpg", "two racing cars on a track bend with advertising hoardings"),
    "picnic_in_the_snow": ("picnic_in_the_snow.jpg", "four people in ski jackets around a checked picnic cloth in snow"),
    "alpine_cottage_with_flowers": ("alpine_cottage_with_flowers.jpg", "a stone alpine cottage with a bank of yellow and purple flowers"),
    "lily_pond_and_pagoda": ("lily_pond_and_pagoda.jpg", "a lily-covered pond with flowering shrubs and a pagoda beyond"),
    "waterfall_under_a_bridge": ("waterfall_under_a_bridge.jpg", "a waterfall dropping beneath a mossy stone arch in dense woodland"),
    "mountain_lake_and_scree": ("mountain_lake_and_scree.jpg", "a high mountain lake below a scree slope and a snow-streaked ridge"),
    "brick_wall_courses": ("brick_wall_courses.jpg", "a wall of identical brick courses; every correspondence is ambiguous, so RANSAC finds no consistent homography at all"),
    "roof_shingles": ("roof_shingles.jpg", "overlapping roof shingles in raking light: repetitive, but each row is distinguishable"),
    "coastal_city_from_the_air": ("coastal_city_from_the_air.jpg", "an aerial view of a coastal city and its harbour"),
    "irrigated_fields_from_the_air": ("irrigated_fields_from_the_air.jpg", "an aerial view of irrigated fields divided by straight canals"),
    "tugboat_under_the_bridge": ("tugboat_under_the_bridge.jpg", "a tugboat on flat water with a suspension bridge behind it in haze"),
    "coarse_woven_fabric": ("coarse_woven_fabric.jpg", "a coarse woven fabric with no large-scale structure"),
    "elephant_crossing_a_road": ("elephant_crossing_a_road.jpg", "an elephant crossing a dirt road in front of a parked car"),
    "cyclists_on_a_track": ("cyclists_on_a_track.jpg", "two cyclists racing on a banked track before a packed crowd"),
    "tank_in_scrub": ("tank_in_scrub.jpg", "a tank in low scrub photographed from above; strong texture, one small object"),
    "street_grid_from_the_air": ("street_grid_from_the_air.jpg", "an aerial view of a dense street grid with a river through it"),
    "fibrous_matting": ("fibrous_matting.jpg", "dense fibrous matting: enormous numbers of features, none of them distinctive"),
    "hillside_town_from_the_air": ("hillside_town_from_the_air.jpg", "an aerial view of a hillside town with terraced roofs and winding roads"),
    "surface_brick_paving": ("surface_brick_paving.jpg", "brick paving laid in a regular grid; the most uniform surface in the set"),
    "surface_fine_weave": ("surface_fine_weave.jpg", "a fine even weave with almost no large-scale variation"),
    "surface_water_ripples": ("surface_water_ripples.jpg", "wind ripples on shallow water: regular, directional, low contrast"),
    "surface_field_mosaic": ("surface_field_mosaic.jpg", "cultivated fields from the air, a mosaic of rectangular plots"),
    "surface_pebbled_render": ("surface_pebbled_render.jpg", "a pebbled render wall in raking light"),
    "surface_coarse_cloth": ("surface_coarse_cloth.jpg", "coarse woven cloth with a visible thread grid"),
    "surface_brick_wall": ("surface_brick_wall.jpg", "a brick wall in stretcher bond, strongly directional"),
    "surface_dry_grass": ("surface_dry_grass.jpg", "dry grass from above: dense, high contrast, no repeating unit"),
    "surface_roof_slates": ("surface_roof_slates.jpg", "overlapping roof slates photographed at an angle"),
    "surface_straw_thatch": ("surface_straw_thatch.jpg", "straw thatch: long thin elements in many directions"),
    "surface_knitted_fabric": ("surface_knitted_fabric.jpg", "a knitted fabric with a fine diagonal rib"),
    "surface_sand_ripple": ("surface_sand_ripple.jpg", "fine ripples in sand: a regular directional surface at low contrast"),
}


def real_photo(name: str) -> np.ndarray:
    """Load one of the bundled **real** photographs as RGB uint8.

    Deliberately a separate function from :func:`sample`. The two are used for
    different things and scored differently: a generated scene has exact ground
    truth and gets a PSNR or an IoU, a real photograph has neither and gets
    shown rather than scored. Keeping them apart at the API makes it hard to
    accidentally quote an accuracy for an image that has no answer.
    """
    if name not in REAL_PHOTOS:
        raise KeyError(f"unknown real photo {name!r}; choose from {sorted(REAL_PHOTOS)}")
    filename, _ = REAL_PHOTOS[name]
    path = Path(__file__).resolve().parent.parent / "assets" / "real" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The real photographs live in assets/real/; "
            "see assets/real/README.md for their provenance."
        )
    return imread(path)


def real_photo_names() -> list[str]:
    return sorted(REAL_PHOTOS)


def sample_names() -> list[str]:
    """Sorted list of every bundled sample name."""
    return sorted(_SAMPLES)
