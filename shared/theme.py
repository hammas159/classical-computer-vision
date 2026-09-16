"""One visual identity per project.

Fifty-eight Streamlit apps that all render in Streamlit's stock dark theme look
like fifty-eight copies of the same app. A reader flicking through the repo
cannot tell from a screenshot which project they are looking at, and a portfolio
of identical screenshots reads as one piece of work repeated rather than as
fifty-eight.

So every project gets its own palette, its own accent, its own heading font and
its own surface treatment, keyed by project number. The palettes are not random:
each is chosen to suit what the project is *about* — the document scanner is ink
on paper, dehazing is a pale grey-blue veil, old photo restoration is sepia,
coin counting is brass on felt. The look carries information.

Nothing here changes a measurement. It changes only ``st.markdown`` CSS and the
matplotlib rcParams the figures inherit, so a themed app and an unthemed one
produce identical numbers.

Usage, once per app, immediately after ``st.set_page_config``::

    from shared import theme
    theme.apply(1)          # project number

``apply`` is idempotent and safe to call from a script that Streamlit re-runs on
every interaction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """One project's visual identity.

    ``bg`` is the page, ``surface`` the cards and table rows, ``ink`` the body
    text, ``accent`` the interactive colour (buttons, slider fills, the active
    tab underline) and ``accent_soft`` the same hue at low saturation, used for
    borders and hovers. ``font`` is a Google-hosted display face for headings
    only — body text stays in the system stack, because a display face at 14px
    costs legibility and buys nothing.
    """

    name: str
    bg: str
    surface: str
    ink: str
    muted: str
    accent: str
    accent_soft: str
    font: str
    #: Matplotlib colour cycle, so the figures drawn inside the app belong to
    #: the same picture as the chrome around them.
    cycle: tuple[str, ...]
    #: Heading letter-spacing. A wide face needs less, a condensed one more.
    tracking: str = "-0.01em"
    #: Corner radius. Varying this matters more than colour for making two
    #: screenshots look like different software.
    radius: str = "10px"


#: Project number -> palette. Every entry is deliberate; see the module
#: docstring. Numbers with no project yet still have a palette reserved so the
#: look is decided once and does not drift when the project is built.
PALETTES: dict[int, Palette] = {
    # ---- 01-12: the flagship dozen -------------------------------------- #
    1: Palette(
        "Ink on paper", "#f7f5ef", "#ffffff", "#1a1a18", "#6b6862",
        "#c0392b", "#e8d5d1", "Spectral", ("#c0392b", "#1a1a18", "#8a7f6d", "#4a6fa5", "#7a9a3f", "#b8860b"),
        radius="2px",
    ),
    2: Palette(
        "Studio portrait", "#14121a", "#1e1b26", "#f0ecf5", "#9a92a8",
        "#e0a458", "#4a3d2e", "Cormorant Garamond", ("#e0a458", "#c76b6b", "#7b9ea8", "#a88bc4", "#8fbf7f", "#d4d4d4"),
        radius="20px",
    ),
    3: Palette(
        "Night shift", "#0a0e17", "#131a28", "#e6edf7", "#7d8899",
        "#4dd0e1", "#14404a", "Space Grotesk", ("#4dd0e1", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176"),
        radius="4px",
    ),
    4: Palette(
        "Sea fog", "#eef2f4", "#ffffff", "#1c2b33", "#6d7f8a",
        "#2e7d8f", "#cfe0e5", "Barlow", ("#2e7d8f", "#c2703f", "#5a7d5a", "#8a6f9a", "#b8934a", "#4a5f7a"),
        radius="14px",
    ),
    5: Palette(
        "Sepia print", "#f4ece0", "#fdf8f0", "#2b2118", "#7a6a56",
        "#8c6239", "#e3d2ba", "Playfair Display", ("#8c6239", "#5c7a5c", "#9a4a3a", "#4a6a8a", "#b08d4f", "#6a5a7a"),
        radius="3px",
    ),
    6: Palette(
        "Asphalt", "#16181a", "#20242a", "#e8eaec", "#8a9096",
        "#f5c518", "#4a3f14", "Oswald", ("#f5c518", "#5ac8fa", "#ff6b6b", "#7ed957", "#c084fc", "#ffffff"),
        tracking="0.02em", radius="0px",
    ),
    7: Palette(
        "Forensic", "#0d0f14", "#161a22", "#dfe4ec", "#79808f",
        "#ff4d6d", "#4a1626", "IBM Plex Mono", ("#ff4d6d", "#4dd4ff", "#ffd166", "#06d6a0", "#b085f5", "#e8e8e8"),
        radius="2px",
    ),
    8: Palette(
        "Handheld", "#1b1d21", "#25282e", "#eceef1", "#8d939b",
        "#64d2a3", "#1c4a3a", "Chivo", ("#64d2a3", "#f2a65a", "#7aa5d2", "#e06c75", "#c8a2d8", "#d8d8d8"),
        radius="8px",
    ),
    9: Palette(
        "Brass on felt", "#12301f", "#19402a", "#f2f0e4", "#93a68f",
        "#d4a132", "#4a3d16", "Bitter", ("#d4a132", "#e8e0c0", "#a8c090", "#c47a4a", "#7aa8b8", "#b8869a"),
        radius="24px",
    ),
    10: Palette(
        "Cut and fold", "#faf7f2", "#ffffff", "#20232a", "#70757f",
        "#6a4c93", "#ddd3ea", "Archivo", ("#6a4c93", "#1982c4", "#8ac926", "#ff924c", "#ff595e", "#4a4e69"),
        radius="6px",
    ),
    11: Palette(
        "Exposure stack", "#101014", "#1a1a20", "#f4f4f8", "#8b8b96",
        "#ff9f1c", "#4a3310", "Manrope", ("#ff9f1c", "#2ec4b6", "#e71d36", "#8093f1", "#ffe066", "#e0e0e0"),
        radius="12px",
    ),
    12: Palette(
        "Depth map", "#0b1220", "#131d30", "#e4ebf7", "#78869c",
        "#7cb9ff", "#1a3555", "Rajdhani", ("#7cb9ff", "#ffd166", "#ef476f", "#06d6a0", "#c77dff", "#f0f0f0"),
        tracking="0.01em", radius="4px",
    ),
}

#: The look for a project with no explicit palette yet, derived from the project
#: number.
#:
#: The three list lengths are **pairwise co-prime — 13, 11 and 7** — on purpose.
#: With equal-length lists the combination repeats every 12 projects, which put
#: 20 and 44 in identical clothes. Co-prime lengths give a period of
#: 13 x 11 x 7 = 1001, so across the 46 auto-themed projects no two share the
#: same accent *and* font *and* radius.
_FALLBACK_ACCENTS = (
    "#e07a5f", "#3d8361", "#5a6fa8", "#b5838d", "#7f8c5a", "#9a6fb0",
    "#c9843e", "#4a8f9c", "#a6567a", "#6b8f4e", "#8a6ad0", "#d0785a",
    "#5f9ea0",
)
_FALLBACK_FONTS = (
    "Outfit", "Sora", "Work Sans", "Lexend", "Urbanist", "Figtree",
    "Public Sans", "Epilogue", "Karla", "Mulish", "Asap",
)
_FALLBACK_RADII = ("0px", "3px", "6px", "10px", "14px", "18px", "24px")


def palette(project: int) -> Palette:
    """The palette for a project number, generating one if none is declared.

    A missing entry is not an error: projects 13-58 get a look derived from
    their number, so every app is visually distinct from its neighbours from the
    first commit, and a hand-picked palette can replace it later without
    touching the app.
    """
    if project in PALETTES:
        return PALETTES[project]
    i = (project - 13) % len(_FALLBACK_ACCENTS)
    accent = _FALLBACK_ACCENTS[i]
    dark = (project % 2) == 1
    return Palette(
        name=f"Auto {project:02d}",
        bg="#101216" if dark else "#f6f5f3",
        surface="#191c22" if dark else "#ffffff",
        ink="#e9ecf1" if dark else "#1d2027",
        muted="#848b96" if dark else "#6c727c",
        accent=accent,
        accent_soft=_mix(accent, "#101216" if dark else "#f6f5f3", 0.78),
        font=_FALLBACK_FONTS[(project - 13) % len(_FALLBACK_FONTS)],
        cycle=(accent, "#7aa5d2", "#e8b44a", "#69ba8f", "#d4737f", "#a48bc8"),
        radius=_FALLBACK_RADII[(project - 13) % len(_FALLBACK_RADII)],
    )


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Blend two ``#rrggbb`` colours, ``t`` of the way from a to b."""
    a = [int(hex_a[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(hex_b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def css(project: int) -> str:
    """The project's stylesheet, as a ``<style>`` block ready for st.markdown.

    Written against Streamlit's ``data-testid`` hooks rather than its generated
    class names, which change between releases. Where a testid is not available
    the selector is attribute-based and deliberately loose, so a Streamlit
    upgrade degrades the styling rather than breaking the app.
    """
    p = palette(project)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family={p.font.replace(" ", "+")}:wght@500;700&display=swap');

:root {{
  --cv-bg: {p.bg};
  --cv-surface: {p.surface};
  --cv-ink: {p.ink};
  --cv-muted: {p.muted};
  --cv-accent: {p.accent};
  --cv-accent-soft: {p.accent_soft};
  --cv-radius: {p.radius};
}}

.stApp, [data-testid="stAppViewContainer"] {{ background: var(--cv-bg); }}
[data-testid="stHeader"] {{ background: transparent; }}
.stApp, .stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"] {{ color: var(--cv-ink); }}

h1, h2, h3, h4 {{
  font-family: '{p.font}', ui-serif, Georgia, serif !important;
  letter-spacing: {p.tracking};
  color: var(--cv-ink) !important;
}}
h1 {{ font-weight: 700; }}

/* The identity stripe. Cheapest possible way to make two screenshots
   unmistakably different software at a glance. */
h1::after {{
  content: "";
  display: block;
  width: 78px; height: 4px;
  margin-top: .45rem;
  background: var(--cv-accent);
  border-radius: 2px;
}}

/* metric tiles */
[data-testid="stMetric"] {{
  background: var(--cv-surface);
  border: 1px solid var(--cv-accent-soft);
  border-left: 3px solid var(--cv-accent);
  border-radius: var(--cv-radius);
  padding: .7rem .9rem;
}}
[data-testid="stMetricValue"] {{ color: var(--cv-ink); }}
[data-testid="stMetricLabel"] {{ color: var(--cv-muted); }}

/* tabs */
[data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid var(--cv-accent-soft); }}
[data-baseweb="tab"] {{ color: var(--cv-muted); }}
[data-baseweb="tab"][aria-selected="true"] {{ color: var(--cv-accent); }}
[data-baseweb="tab-highlight"] {{ background: var(--cv-accent); }}

/* controls */
.stButton > button, .stDownloadButton > button {{
  background: var(--cv-accent);
  color: {_readable_on(p.accent)};
  border: none;
  border-radius: var(--cv-radius);
  font-weight: 600;
}}
[data-testid="stSlider"] [role="slider"] {{ background: var(--cv-accent) !important; }}
[data-baseweb="select"] > div {{
  background: var(--cv-surface);
  border-color: var(--cv-accent-soft);
  border-radius: var(--cv-radius);
}}
[data-testid="stExpander"] {{
  background: var(--cv-surface);
  border: 1px solid var(--cv-accent-soft);
  border-radius: var(--cv-radius);
}}
[data-testid="stImage"] img {{ border-radius: var(--cv-radius); }}
hr, [data-testid="stDivider"] {{ border-color: var(--cv-accent-soft); }}

/* Captions under the pipeline panels. Small, muted, and never the accent --
   four coloured captions in a row would compete with the images. */
[data-testid="stCaptionContainer"], .stCaption {{ color: var(--cv-muted) !important; }}
</style>
"""


def _readable_on(bg: str) -> str:
    """Black or white, whichever is readable on ``bg``.

    Uses relative luminance rather than a naive average: a saturated yellow and
    a saturated blue can share a mean RGB and need opposite text colours.
    """
    r, g, b = (int(bg[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#11131a" if lum > 0.45 else "#ffffff"


def rc_params(project: int) -> dict:
    """Matplotlib rcParams matching the project's palette.

    Applied so a chart inside the app sits in the same design as the app, rather
    than punching a white rectangle through a dark page.
    """
    p = palette(project)
    return {
        "figure.facecolor": p.surface,
        "axes.facecolor": p.surface,
        "axes.edgecolor": p.muted,
        "axes.labelcolor": p.ink,
        "text.color": p.ink,
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "grid.color": _mix(p.muted, p.surface, 0.72),
        "axes.prop_cycle": __import__("cycler").cycler(color=list(p.cycle)),
        "savefig.facecolor": p.surface,
    }


def config_toml(project: int) -> str:
    """The project's ``.streamlit/config.toml``.

    CSS alone is not enough. Streamlit renders its *widgets* — selectboxes,
    radios, checkboxes — through BaseWeb, which reads its own theme rather than
    the page stylesheet. Overriding those with CSS means chasing generated class
    names, and the first miss shows as a dark navy dropdown sitting on a cream
    page. Setting Streamlit's native theme fixes the widgets at the source, and
    the CSS in :func:`css` then handles only what Streamlit has no setting for
    (fonts, the accent stripe, the metric tiles, corner radius).

    Written from the same :class:`Palette`, so the two can never disagree.
    Streamlit reads ``.streamlit/config.toml`` from the **current working
    directory**, which is why each project carries its own and every README says
    to launch from inside the project folder.
    """
    p = palette(project)
    light = _luminance(p.bg) > 0.5
    return f"""# Generated by shared/theme.py — do not edit by hand.
# Project {project:02d}: "{p.name}"
#
# Streamlit's own theme, so BaseWeb widgets (selectbox, radio, checkbox) match
# the page instead of falling back to the stock dark palette. The matching CSS
# lives in shared/theme.py and is applied by the app at runtime.
[theme]
base = "{"light" if light else "dark"}"
primaryColor = "{p.accent}"
backgroundColor = "{p.bg}"
secondaryBackgroundColor = "{p.surface}"
textColor = "{p.ink}"

[server]
headless = true

[browser]
gatherUsageStats = false
"""


def _luminance(hex_colour: str) -> float:
    """Relative luminance of a ``#rrggbb`` colour, 0 (black) to 1 (white)."""
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def apply(project: int) -> Palette:
    """Apply the project's theme to the running Streamlit app.

    Imports streamlit locally rather than at module scope, so the rest of this
    module — and therefore the test suite — stays importable without a Streamlit
    runtime, matching the convention in :mod:`shared.ui`.
    """
    import matplotlib.pyplot as plt
    import streamlit as st

    p = palette(project)
    st.markdown(css(project), unsafe_allow_html=True)
    plt.rcParams.update(rc_params(project))
    return p
