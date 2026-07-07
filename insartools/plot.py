"""
===============================================================================

InSARTools

plot.py

Low-level plotting engine used throughout InSARTools.

This module provides:

    • Figure creation
    • Image plotting
    • Geographic axes
    • Colorbars
    • Scale bars
    • Consistent publication-quality defaults

Higher-level plotting routines should be implemented in wrapped.py,
unwrapped.py and amplitude.py.

Author
------
Dibyashakti Panda

License
-------
MIT

===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import logging

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

logger = logging.getLogger(__name__)

###############################################################################
# Public API
###############################################################################

__all__ = [
    "PlotConfig",
    "imshow",
    "save",
    "show",
]

###############################################################################
# Type aliases
###############################################################################

Interpolation = Literal[
    "nearest",
    "bilinear",
    "bicubic",
]

Origin = Literal[
    "upper",
    "lower",
]

Aspect = Literal[
    "equal",
    "auto",
]

###############################################################################
# Plot configuration
###############################################################################


@dataclass(slots=True)
class PlotConfig:
    """
    Configuration for image plotting.
    """

    # Figure

    figsize: tuple[float, float] = (8.0, 8.0)

    dpi: int = 300

    title: str = ""

    # Image

    cmap: str = "viridis"

    interpolation: Interpolation = "nearest"

    origin: Origin = "upper"

    aspect: Aspect = "equal"

    vmin: float | None = None

    vmax: float | None = None

    # Labels

    xlabel: str = "Longitude"

    ylabel: str = "Latitude"

    fontsize: int = 12

    ticksize: int = 10

    # Colorbar

    show_colorbar: bool = True

    colorbar_label: str = ""

    # NaN handling

    transparent_nan: bool = True

    nan_color: str = "white"

    # Grid

    show_grid: bool = False

    # Scale bar

    show_scalebar: bool = False

    scalebar_length: float | None = None

    scalebar_label: str | None = None

###############################################################################
# Validation
###############################################################################


def _validate(
    data: np.ndarray,
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
) -> None:
    """
    Validate plotting inputs.
    """

    if data.ndim != 2:
        raise ValueError(
            "Input data must be a 2-D array."
        )

    if latitude is not None:

        if latitude.ndim != 1:
            raise ValueError(
                "Latitude must be a 1-D vector."
            )

        if latitude.size != data.shape[0]:
            raise ValueError(
                "Latitude length does not match image rows."
            )

    if longitude is not None:

        if longitude.ndim != 1:
            raise ValueError(
                "Longitude must be a 1-D vector."
            )

        if longitude.size != data.shape[1]:
            raise ValueError(
                "Longitude length does not match image columns."
            )

            ###############################################################################
# Internal helper functions
###############################################################################


def _create_axes(
    config: PlotConfig,
) -> tuple[Figure, Axes]:
    """
    Create a matplotlib figure and axes.
    """

    fig, ax = plt.subplots(
        figsize=config.figsize,
        dpi=config.dpi,
    )

    return fig, ax


def _get_extent(
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
) -> tuple[float, float, float, float] | None:
    """
    Return image extent for geographic coordinates.
    """

    if latitude is None or longitude is None:
        return None

    return (
        float(longitude.min()),
        float(longitude.max()),
        float(latitude.min()),
        float(latitude.max()),
    )


def _apply_axes(
    ax: Axes,
    config: PlotConfig,
) -> None:
    """
    Apply common axis formatting.
    """

    ax.set_title(
        config.title,
        fontsize=config.fontsize,
    )

    ax.set_xlabel(
        config.xlabel,
        fontsize=config.fontsize,
    )

    ax.set_ylabel(
        config.ylabel,
        fontsize=config.fontsize,
    )

    ax.tick_params(
        labelsize=config.ticksize,
    )

    if config.show_grid:

        ax.grid(

            True,

            linestyle=":",

            linewidth=0.5,

            alpha=0.5,
        )

def _add_colorbar(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    config: PlotConfig,
):
    """
    Add a colorbar if requested.
    """

    if not config.show_colorbar:
        return None

    cb = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    if config.colorbar_label:
        cb.set_label(
            config.colorbar_label,
            fontsize=config.fontsize,
        )

    cb.ax.tick_params(
        labelsize=config.ticksize,
    )

    return cb


def _add_scalebar(
    ax: Axes,
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
    config: PlotConfig,
) -> None:
    """
    Draw a simple geographic scale bar.

    Assumes geographic coordinates in degrees.
    """

    if not config.show_scalebar:
        return

    if latitude is None or longitude is None:
        return

    length_km = (
        10.0
        if config.scalebar_length is None
        else config.scalebar_length
    )

    mean_lat = float(
        np.mean(latitude)
    )

    deg = length_km / (
        111.32
        * np.cos(
            np.deg2rad(mean_lat)
        )
    )

    xmin = float(longitude.min())
    xmax = float(longitude.max())

    ymin = float(latitude.min())
    ymax = float(latitude.max())

    x0 = xmin + 0.05 * (xmax - xmin)
    y0 = ymin + 0.05 * (ymax - ymin)

    ax.plot(
        [x0, x0 + deg],
        [y0, y0],
        color="black",
        linewidth=3,
    )

    label = (
        config.scalebar_label
        if config.scalebar_label is not None
        else f"{length_km:g} km"
    )

    ax.text(
        x0 + deg / 2,
        y0,
        label,
        ha="center",
        va="bottom",
        fontsize=config.ticksize,
    )

    ###############################################################################
# Public plotting functions
###############################################################################


def imshow(
    data: np.ndarray,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes, AxesImage]:
    """
    Display a 2-D raster.

    Parameters
    ----------
    data : ndarray
        Image to display.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    config : PlotConfig, optional
        Plot configuration.

    Returns
    -------
    fig : Figure

    ax : Axes

    image : AxesImage
    """

    if config is None:
        config = PlotConfig()

    data = np.asarray(data)

    _validate(
        data=data,
        latitude=latitude,
        longitude=longitude,
    )

    fig, ax = _create_axes(config)

    extent = _get_extent(
        latitude=latitude,
        longitude=longitude,
    )

    # ------------------------------------------------------------------
    # Prepare image
    # ------------------------------------------------------------------

    plot_data = np.array(
        data,
        copy=True,
    )

    cmap = plt.get_cmap(
        config.cmap,
    ).copy()

    if config.transparent_nan:

        cmap.set_bad(
            color=config.nan_color,
            alpha=0.0,
        )

    plot_data = np.ma.masked_invalid(
        plot_data,
    )

    image = ax.imshow(
        plot_data,
        cmap=cmap,
        interpolation=config.interpolation,
        origin=config.origin,
        aspect=config.aspect,
        extent=extent,
        vmin=config.vmin,
        vmax=config.vmax,
    )

    _apply_axes(
        ax=ax,
        config=config,
    )

    _add_colorbar(
        fig=fig,
        ax=ax,
        image=image,
        config=config,
    )

    _add_scalebar(
        ax=ax,
        latitude=latitude,
        longitude=longitude,
        config=config,
    )

    return fig, ax, image


###############################################################################
# Figure utilities
###############################################################################


def save(
    figure: Figure,
    filename: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """
    Save a matplotlib figure.
    """

    filename = Path(filename)

    figure.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )

    logger.info(
        "Saved figure to %s",
        filename,
    )

    return filename


def show(
    figure: Figure | None = None,
) -> None:
    """
    Display a figure.
    """

    plt.show()


