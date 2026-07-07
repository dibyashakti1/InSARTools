"""
insartools.plot
===============

Core plotting utilities for InSARTools.

This module provides a generic scientific raster plotting engine used by
all visualization modules (wrapped, unwrapped, amplitude, phase).

The functions in this module are intentionally data-agnostic and operate
on any two-dimensional raster.

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

logger = logging.getLogger(__name__)

__all__ = [
    "PlotConfig",
    "imshow",
    "save",
    "show",
]


###############################################################################
# Configuration
###############################################################################

Interpolation = Literal[
    "nearest",
    "bilinear",
    "bicubic",
    "none",
]


@dataclass(slots=True)
class PlotConfig:
    """
    Configuration options for raster plotting.

    Parameters
    ----------
    cmap : str
        Matplotlib colormap.

    figsize : tuple
        Figure size in inches.

    dpi : int
        Figure resolution.

    interpolation : str
        Image interpolation method.

    colorbar : bool
        Draw a colorbar.

    colorbar_label : str
        Colorbar label.

    title : str
        Figure title.

    xlabel : str
        X-axis label.

    ylabel : str
        Y-axis label.

    aspect : str
        Image aspect ratio.

    origin : str


    """
@dataclass(slots=True)
class PlotConfig:

    cmap: str = "viridis"

    figsize: tuple[float, float] = (8,8)

    dpi: int = 300

    interpolation: Interpolation = "nearest"

    show_colorbar: bool = True

    colorbar_label: str = ""

    title: str = ""

    xlabel: str = "Longitude"

    ylabel: str = "Latitude"

    aspect: str = "equal"

    origin: str = "upper"

    vmin: float | None = None

    vmax: float | None = None

    fontsize: int = 12

    ticksize: int = 10

    transparent_nan: bool = True

    nan_color: str = "white"

    show_scalebar: bool = False

    scalebar_length: float | None = None

    scalebar_label: str | None = None

    show_grid: bool = False

    cmap: str = "viridis"

    figsize: tuple[float, float] = (8.0, 8.0)

    dpi: int = 300

    interpolation: Interpolation = "nearest"

    show_colorbar: bool = True

    colorbar_label: str = ""

    title: str = ""

    xlabel: str = "Longitude"

    ylabel: str = "Latitude"

    aspect: str = "equal"

    origin: str = "upper"

    vmin: float | None = None

    vmax: float | None = None


###############################################################################
# Validation
###############################################################################


def _validate(
    data: np.ndarray,
    lat: np.ndarray | None,
    lon: np.ndarray | None,
    config: PlotConfig,
) -> None:
    """
    Validate plotting inputs.
    """

    if data.ndim != 2:
        raise ValueError(
            "Input raster must be two-dimensional."
        )

    if lat is not None:

        if lat.ndim != 1:
            raise ValueError(
                "Latitude must be a one-dimensional vector."
            )

        if lat.size != data.shape[0]:
            raise ValueError(
                "Latitude length must equal raster rows."
            )

    if lon is not None:

        if lon.ndim != 1:
            raise ValueError(
                "Longitude must be a one-dimensional vector."
            )

        if lon.size != data.shape[1]:
            raise ValueError(
                "Longitude length must equal raster columns."
            )

    if config.dpi <= 0:
        raise ValueError(
            "DPI must be positive."
        )


###############################################################################
# Helper functions
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
        constrained_layout=True,
    )

    return fig, ax


def _get_extent(
    lat: np.ndarray | None,
    lon: np.ndarray | None,
) -> tuple[float, float, float, float] | None:
    """
    Build matplotlib extent from coordinate vectors.

    Returns
    -------
    tuple or None
    """

    if lat is None or lon is None:
        return None

    return (
        float(lon.min()),
        float(lon.max()),
        float(lat.min()),
        float(lat.max()),
    )


def _apply_axes(
    ax: Axes,
    config: PlotConfig,
) -> None:
    """
    Apply axes formatting.
    """

    ax.set_title(config.title)

    ax.set_xlabel(config.xlabel)

    ax.set_ylabel(config.ylabel)

    ax.set_aspect(config.aspect)

    ax.tick_params(
        direction="out",
        length=4,
        width=0.8,
    )

    ax.grid(config.show_grid)

    ax.title.set_fontsize(config.fontsize)

    ax.xaxis.label.set_fontsize(config.fontsize)

    ax.yaxis.label.set_fontsize(config.fontsize)

    ax.tick_params(
    labelsize=config.ticksize,
    )

def _add_colorbar(
    fig: Figure,
    ax: Axes,
    image: AxesImage,
    config: PlotConfig,
) -> None:
    """
    Add a colorbar.
    """

    if not config.colorbar:
        return

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        config.colorbar_label,
    )

    ###############################################################################
# Public plotting API
###############################################################################


def imshow(
    data: np.ndarray,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    config: PlotConfig | None = None,
) -> tuple[Figure, Axes, AxesImage]:
    """
    Display a two-dimensional raster.

    Parameters
    ----------
    data : ndarray
        Input raster.

    lat : ndarray, optional
        Latitude vector.

    lon : ndarray, optional
        Longitude vector.

    config : PlotConfig, optional
        Plot configuration.

    Returns
    -------
    fig : Figure

    ax : Axes

    image : AxesImage
    """

    logger.info("Creating raster plot.")

    if config is None:
        config = PlotConfig()

    _validate(
        data=data,
        lat=lat,
        lon=lon,
        config=config,
    )

    fig, ax = _create_axes(config)

    extent = _get_extent(
        lat=lat,
        lon=lon,
    )

plot_data = np.array(
    data,
    copy=True,
)

cmap = plt.get_cmap(config.cmap).copy()

if config.transparent_nan:
    cmap.set_bad(
        color=config.nan_color,
        alpha=0.0,
    )

plot_data = np.ma.masked_invalid(plot_data)

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
        def _add_scalebar(
    ax: Axes,
    latitude: np.ndarray | None,
    longitude: np.ndarray | None,
    config: PlotConfig,
) -> None:
    """
    Add a simple scale bar.

    Currently supports geographic coordinates.
    """

    if not config.scalebar:
        return

    if latitude is None or longitude is None:
        return

    if config.scalebar_length is None:
        length_km = 10.0
    else:
        length_km = config.scalebar_length

    mean_lat = float(np.mean(latitude))

    deg = length_km / (
        111.32 * np.cos(
            np.deg2rad(mean_lat)
        )
    )

    x0 = longitude.min() + 0.05 * (
        longitude.max() - longitude.min()
    )

    y0 = latitude.min() + 0.05 * (
        latitude.max() - latitude.min()
    )

    ax.plot(
        [x0, x0 + deg],
        [y0, y0],
        color="black",
        linewidth=3,
    )

    label = (
        config.scalebar_label
        or
        f"{length_km:g} km"
    )

    ax.text(
        x0 + deg / 2,
        y0,
        label,
        ha="center",
        va="bottom",
        fontsize=config.ticksize,
    )
    )

    logger.info("Plot created successfully.")

    return (
        fig,
        ax,
        image,
    )


###############################################################################
# Figure utilities
###############################################################################


def save(
    fig: Figure,
    filename: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = "tight",
) -> Path:
    """
    Save a matplotlib figure.

    Parameters
    ----------
    fig : Figure
        Figure object.

    filename : str or Path
        Output filename.

    dpi : int, default=300
        Output resolution.

    transparent : bool, default=False
        Save with transparent background.

    bbox_inches : str
        Bounding box mode.

    Returns
    -------
    Path
        Saved filename.
    """

    filename = Path(filename)

    logger.info(
        "Saving figure to %s",
        filename,
    )

    fig.savefig(
        filename,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
    )

    logger.info("Figure saved.")

    return filename


def show() -> None:
    """
    Display all pending figures.
    """

    plt.show()


