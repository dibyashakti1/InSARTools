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
import rasterio

from . import export
from . import geocode

from .plot import PlotConfig
from .plot import imshow

logger = logging.getLogger(__name__)

__all__ = [
    "RasterStyle",
    "_plot_raster",
    "DISPLACEMENT_STYLE",
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

def _geographic_coordinates(
    *,
    input_file: str | Path,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read geographic coordinate vectors from an already-geocoded raster.

    The input may be a GeoTIFF or an ISCE binary raster accompanied by
    a GDAL VRT, such as ``product.geo`` and ``product.geo.vrt``.

    Returns
    -------
    tuple of ndarray or None
        Latitude and longitude vectors, or None when the source is not
        a suitable north-up geographic raster.
    """

    source = Path(input_file)

    candidates = []
    if source.suffix.lower() == ".vrt":
        candidates.append(source)
    else:
        candidates.extend([
            Path(str(source) + ".vrt"),
            source,
        ])

    for candidate in candidates:
        if not candidate.is_file():
            continue

        try:
            with rasterio.open(candidate) as ds:
                if ds.crs is None or not ds.crs.is_geographic:
                    continue

                transform = ds.transform

# Reject identity/pixel-coordinate transforms even if a CRS is set.
                if (
                    abs(transform.a - 1.0) < 1e-12
                    and abs(transform.e - 1.0) < 1e-12
                    and abs(transform.b) < 1e-12
                    and abs(transform.d) < 1e-12
                    and abs(transform.c) < 1e-12
                    and abs(transform.f) < 1e-12
                ):
                    logger.debug(
                        "Skipping identity transform falsely labelled geographic: %s",
                        candidate,
                    )
                    continue

                # The plotting interface uses 1-D coordinate vectors.
                # Reject rotated/sheared rasters rather than misplot them.
                if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
                    logger.debug(
                        "Skipping rotated geographic raster: %s",
                        candidate,
                    )
                    continue

                if (ds.height, ds.width) != data.shape[:2]:
                    logger.debug(
                        "Raster dimensions do not match input data: %s",
                        candidate,
                    )
                    continue

                longitude = (
                    transform.c
                    + (np.arange(ds.width) + 0.5) * transform.a
                )
                latitude = (
                    transform.f
                    + (np.arange(ds.height) + 0.5) * transform.e
                )

                logger.info(
                    "Using geographic coordinates from %s",
                    candidate,
                )
                return latitude, longitude

                # Reject pixel-coordinate transforms incorrectly labelled
                # as geographic coordinates (e.g. an identity transform).
                if (
                    np.any(np.abs(latitude) > 90)
                    or np.any(np.abs(longitude) > 180)
                ):
                    logger.debug(
                        "Skipping invalid geographic coordinate ranges: %s",
                        candidate,
                    )
                    continue

        except (rasterio.errors.RasterioError, OSError) as exc:
            logger.debug(
                "Could not read georeferencing from %s: %s",
                candidate,
                exc,
            )

    return None


def _prepare_geographic_view(
    *,
    data: np.ndarray,
    input_file: str | Path | None,
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

    # First handle inputs that are already geocoded, including ISCE
    # rasters accompanied by a GDAL VRT.
    if input_file is not None:
        coordinates = _geographic_coordinates(
            input_file=input_file,
            data=data,
        )

        if coordinates is not None:
            latitude, longitude = coordinates
            title = f"{title} (Geocoded)"
            return data, latitude, longitude, title

    # Otherwise retain the existing radar-coordinate geocoding route.
    if geometry_dir is None:
        raise ValueError(
            "Could not obtain geographic coordinates from the input raster. "
            "For radar-coordinate data, provide geometry_dir."
        )

    logger.info("Preparing geographic raster using geometry directory.")

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
    input_file: str | Path | None = None,
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
        input_file=input_file,
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


    formats = list(export_formats)

    if lat is None or lon is None:
        if "tif" in formats:
            logger.warning(
                "Skipping GeoTIFF export for radar-coordinate raster."
            )
            formats.remove("tif")

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
            formats=formats,
            metadata=info,
            variable_name=variable_name,
            cmap=config.cmap,
            vmin=config.vmin,
            vmax=config.vmax,
        )

    ###########################################################################
    # Display
    ###########################################################################

    if show:
        plt.show()
    logger.info("Raster visualization complete.")
    return fig, ax, image
