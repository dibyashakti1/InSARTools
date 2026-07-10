"""
===============================================================================

Private raster plotting backend used by InSARTools.

This module implements the common plotting workflow shared by all
raster products, including wrapped phase, unwrapped phase,
coherence, amplitude, LOS displacement, DEM and other
geocoded/radar-coordinate rasters.

This module is intended for internal use only.

===============================================================================
"""


from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import export
from . import geocode

from .plot import PlotConfig
from .plot import imshow

logger = logging.getLogger(__name__)

__all__ = [
    "RasterStyle",
    "_plot_raster",
]

###############################################################################
# Raster plotting style
###############################################################################


@dataclass(slots=True)
class RasterStyle:
    """
    Default plotting style for raster products.
    """

    title: str

    cmap: str

    colorbar_label: str

    variable_name: str = "data"

    vmin: float | None = None

    vmax: float | None = None

###############################################################################
# Geographic preparation
###############################################################################

def _prepare_geographic_view(
    *,
    data: np.ndarray,
    view: str,
    geometry_dir: str | Path | None,
    title: str,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    """
    Prepare raster for radar or geographic visualization.

    Parameters
    ----------
    data
        Radar-coordinate raster.

    view
        Either "radar" or "geo".

    geometry_dir
        ISCE geometry directory.

    title
        Figure title.

    Returns
    -------
    data
        Raster to plot.

    latitude
        Latitude vector.

    longitude
        Longitude vector.

    title
        Updated figure title.
    """

    if view not in ("radar", "geo"):
        raise ValueError(
            "view must be either 'radar' or 'geo'."
        )

    latitude = None
    longitude = None

    if view == "radar":

        logger.info(
            "Using radar coordinates."
        )

        return (
            data,
            latitude,
            longitude,
            title,
        )

    ###########################################################################
    # Geographic coordinates
    ###########################################################################

    if geometry_dir is None:
        raise ValueError(
            "geometry_dir is required when view='geo'."
        )

    logger.info(
        "Preparing geographic raster."
    )

    data, latitude, longitude = geocode.geocode_for_plot(
        data=data,
        geometry_dir=geometry_dir,
    )

    title = f"{title} (Geocoded)"

    return (
        data,
        latitude,
        longitude,
        title,
    )

###############################################################################
# Plot configuration
###############################################################################

def _create_plot_config(
    *,
    style: RasterStyle,
    figsize: tuple[float, float],
    dpi: int,
    cmap: str | None = None,
    title: str | None = None,
) -> PlotConfig:
    """
    Create a PlotConfig object from a RasterStyle.

    Parameters
    ----------
    style
        Raster plotting style.

    figsize
        Figure size.

    dpi
        Figure resolution.

    cmap
        Optional colormap override.

    title
        Optional title override.

    Returns
    -------
    PlotConfig
    """

    return PlotConfig(
        cmap=cmap if cmap is not None else style.cmap,
        figsize=figsize,
        dpi=dpi,
        title=title if title is not None else style.title,
        show_colorbar=True,
        colorbar_label=style.colorbar_label,
        vmin=style.vmin,
        vmax=style.vmax,
    )


###############################################################################
# Main plotting helper
###############################################################################

def _plot_raster(
    *,
    data: np.ndarray,
    style: RasterStyle,
    geometry_dir: str | Path | None = None,
    output: str | Path | None = None,
    processor: str = "auto",
    view: str = "radar",
    export_formats: tuple[str, ...] = ("png",),
    figsize: tuple[float, float] = (6.5, 5.5),
    dpi: int = 300,
    cmap: str | None = None,
    title: str | None = None,
    variable_name: str = "data",
    metadata: dict | None = None,
    show: bool = False,
):

    """
    Generic raster plotting routine used throughout InSARTools.

    Parameters
    ----------
    data
        Raster to display.

    style
        Raster plotting style.

    geometry_dir
        Geometry directory for geographic plotting.

    output
        Output filename (without extension).

    processor
        Processor name.

    view
        "radar" or "geo".

    export_formats
        Output formats.

    figsize
        Figure size.

    dpi
        Figure DPI.

    cmap
        Optional colormap override.

    title
        Optional title override.

    variable_name
        Variable name used in MAT export.

    metadata
        Additional metadata.

    show
        Display the figure.

    Returns
    -------
    fig
    ax
    image
    """

    ###########################################################################
    # Geographic preparation
    ###########################################################################

    plot_title = title if title is not None else style.title

    data, lat, lon, plot_title = _prepare_geographic_view(
        data=data,
        view=view,
        geometry_dir=geometry_dir,
        title=plot_title,
    )

    ###########################################################################
    # Plot configuration
    ###########################################################################

    config = _create_plot_config(
        style=style,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=plot_title,
    )

    ###########################################################################
    # Plot image
    ###########################################################################
    logger.info("Rendering raster.")
    fig, ax, image = imshow(
        data,
        latitude=lat,
        longitude=lon,
        config=config,
    )

    ###########################################################################
    # Metadata
    ###########################################################################

    info = {
        "processor": processor,
        "view": view,
        "rows": int(data.shape[0]),
        "cols": int(data.shape[1]),
        "dtype": str(data.dtype),
    }

    if metadata is not None:
        info.update(metadata)

    ###########################################################################
    # Export
    ###########################################################################

    if output is not None:
        logger.info("Exporting raster.")
        export.save(
            figure=fig,
            data=data,
            latitude=lat,
            longitude=lon,
            output=output,
            formats=list(export_formats),
            metadata=info,
            variable_name=variable_name,
        )

    ###########################################################################
    # Display
    ###########################################################################

    if show:
        plt.show()
    logger.info("Raster visualization complete.")
    return fig, ax, image
