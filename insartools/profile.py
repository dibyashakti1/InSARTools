"""
===============================================================================

InSARTools

profile.py

Extract and visualize profiles from raster products.

Supports

- Radar coordinates
- Geographic coordinates
- Pixel sampling
- CSV export
- MAT export
- Publication-quality figures

Author
------
Dibyashakti Panda

License
-------
MIT

===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import read_raster
from . import geocode

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]


DEFAULT_TITLE = "Profile"

DEFAULT_EXPORT = (
    "png",
)

def plot(
    input_file: str | Path,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    coordinates: str = "pixel",
    geometry_dir: str | Path | None = None,
    output: str | Path |None = None,
    figsize: tuple[float, float] = (7.0, 4.0),
    dpi: int = 300,
    title: str = DEFAULT_TITLE,
    show: bool = False,
):

def _read_data(...):
    ...


def _convert_coordinates(...):
    ...


def _extract_profile(...):
    ...


def _compute_distance(...):
    ...


def _plot_profile(...):
    ...


