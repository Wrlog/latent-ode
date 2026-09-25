"""Colours for the dashboard, in a light and a dark variant.

The light values match the matplotlib figures in figures/ (scripts/make_figures.py)
so the static PNGs and the interactive charts look like the same figure. The dark
values are the same hues stepped for a dark surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_2: str
    muted: str
    grid: str
    axis: str
    methods: dict = field(default_factory=dict)


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    methods={
        "latent ODE": "#2a78d6",
        "popPK, no covariates": "#eb6834",
        "popPK with covariates": "#1baf7a",
        "popPK, time-varying CL": "#eda100",
        "true model (ceiling)": "#4a3aa7",
    },
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    methods={
        "latent ODE": "#3987e5",
        "popPK, no covariates": "#d95926",
        "popPK with covariates": "#199e70",
        "popPK, time-varying CL": "#c98500",
        "true model (ceiling)": "#9085e9",
    },
)

THEMES = (LIGHT, DARK)

# Method order, marker and dash. Shape and dash back up colour, so no series is
# told apart by colour alone.
METHOD_ORDER = [
    "latent ODE",
    "popPK, no covariates",
    "popPK with covariates",
    "popPK, time-varying CL",
    "true model (ceiling)",
]
SYMBOLS = {
    "latent ODE": "circle",
    "popPK, no covariates": "square",
    "popPK with covariates": "diamond",
    "popPK, time-varying CL": "triangle-up",
    "true model (ceiling)": "triangle-down",
}
DASH = {"true model (ceiling)": "dash"}
