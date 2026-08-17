"""Self-contained matplotlib styling for the machine-learning notebooks.

Kept local --- a sibling of the notebooks --- so each notebook is self-contained
and does NOT depend on the repository's shared ``teaching_utils`` package. It just
sets a clean, colorblind-safe matplotlib look that matches the lecture figures.

Usage inside a notebook (run from this directory):

    from nb_utils import use_style
    use_style("notes")     # or "slides"
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib as mpl
from cycler import cycler

# Colorblind-safe qualitative palette (Wong, 2011).
COLORBLIND_CYCLE = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Per-context overrides: "notes" for a printed page, "slides" for a projector.
CONTEXTS: Dict[str, Dict[str, Any]] = {
    "notes": {
        "figure.figsize": (5.5, 3.4), "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "lines.linewidth": 1.5, "lines.markersize": 5,
    },
    "slides": {
        "figure.figsize": (7.0, 4.2), "font.size": 14, "axes.titlesize": 16,
        "axes.labelsize": 14, "legend.fontsize": 12, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "lines.linewidth": 2.2, "lines.markersize": 7,
    },
}


def use_style(context: str = "notes") -> Dict[str, Any]:
    """Apply the notebook matplotlib style for ``context`` ("notes" or "slides")."""
    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}; expected one of {sorted(CONTEXTS)}")
    params: Dict[str, Any] = {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif",
                       "Times New Roman", "serif"],
        "mathtext.fontset": "cm",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=COLORBLIND_CYCLE),
        "figure.constrained_layout.use": True,
        "figure.dpi": 100, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    }
    params.update(CONTEXTS[context])
    mpl.rcParams.update(params)
    return params
