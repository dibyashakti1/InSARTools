"""
===============================================================================

InSARTools
Coherence Visualization

===============================================================================

Author
------
Dibyashakti Panda

Description
-----------
Visualization utilities for interferometric coherence rasters.

Features
--------
- Radar-coordinate plotting
- Geographic plotting using GDAL geocoding
- MATLAB export
- GeoTIFF export
- PNG/PDF/SVG export
- JSON metadata export

===============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .io import read_raster

from ._raster import _plot_raster
from ._styles import COHERENCE_STYLE

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]

###############################################################################
# Defaults
###############################################################################

#DEFAULT_CMAP = "viridis"

#DEFAULT_TITLE = "Interferometric Coherence"

#DEFAULT_COLORBAR = "Coherence"

DEFAULT_EXPORT = (
    "png",
    "pdf",
    "svg",
    "mat",
    "json",
    "tif",
    "kml",
)

###############################################################################
# Internal helper
###############################################################################



def _prepare_coherence(
    *,
    input_file: str | Path,
    processor: str = "auto",
) -> np.ndarray:
    """
    Read and validate an interferometric coherence raster.

    Supports single-band coherence rasters and multi-band rasters
    where normalized coherence is stored in band 2.

    Band 1 is preferred when it contains valid normalized coherence.
    Otherwise, band 2 is checked.

    Parameters
    ----------
    input_file
        Path to the coherence raster.

    processor
        Reserved for future processor-specific handling.

    Returns
    -------
    numpy.ndarray
        Float32 coherence array with values in [0, 1].

    Raises
    ------
    ValueError
        If no suitable normalized coherence band is found.
    """

    from osgeo import gdal

    logger.info("Reading coherence raster: %s", input_file)

    dataset = gdal.Open(str(input_file), gdal.GA_ReadOnly)

    if dataset is None:
        raise ValueError(
            f"Cannot open coherence raster: {input_file}"
        )

    try:
        band_count = dataset.RasterCount

        if band_count < 1:
            raise ValueError(
                f"Raster contains no bands: {input_file}"
            )

        def is_normalized_coherence(data: np.ndarray) -> bool:
            """Check whether finite, valid pixels lie in [0, 1]."""
            valid = np.isfinite(data)

            if not np.any(valid):
                return False

            values = data[valid]

            return (
                float(np.min(values)) >= 0.0
                and float(np.max(values)) <= 1.0
            )

        selected_band = None
        coherence = None

        # Prefer band 1 when it contains normalized coherence.
        for band_number in (1, 2):
            if band_number > band_count:
                continue

            data = dataset.GetRasterBand(
                band_number
            ).ReadAsArray()

            if data is None:
                continue

            data = data.astype(np.float32, copy=False)

            if is_normalized_coherence(data):
                selected_band = band_number
                coherence = data
                break

        if coherence is None:
            raise ValueError(
                "Could not identify a normalized coherence band "
                f"in {input_file}. Checked bands 1 and 2; "
                "expected finite values within [0, 1]."
            )

        logger.info(
            "Using coherence band %d from %s",
            selected_band,
            input_file,
        )

        # Preserve NaN and Inf as invalid pixels.
        coherence = np.where(
            np.isfinite(coherence),
            coherence,
            np.nan,
        ).astype(np.float32, copy=False)

        return coherence

    finally:
        dataset = None



###############################################################################
# Public API
###############################################################################


def plot(
    input_file: str | Path,
    *,
    geometry_dir: str | Path | None = None,
    output: str | Path | None = None,
    processor: str = "auto",
    view: str = "radar",
    export_formats: tuple[str, ...] = DEFAULT_EXPORT,
    figsize: tuple[float, float] = (6.5, 5.5),
    dpi: int = 300,
    cmap: str | None = None,
    title: str | None = None,
    show: bool = False,
):
    """
    Plot an interferometric coherence raster.

    Parameters
    ----------
    input_file : str or Path
        Coherence raster.

    geometry_dir : str or Path, optional
        Geometry directory used for geographic plotting.

    output : str or Path, optional
        Output filename without extension.

    processor : str, default="auto"
        InSAR processor.

    view : {"radar", "geo"}, default="radar"
        Radar-coordinate or geographic view.

    export_formats : tuple of str
        Output formats.

    figsize : tuple
        Figure size.

    dpi : int
        Figure resolution.

    cmap : str, optional
        Override the default colormap.

    title : str, optional
        Override the default title.

    show : bool, default=False
        Display the figure.

    Returns
    -------
    fig
        Matplotlib Figure.

    ax
        Matplotlib Axes.

    image
        Matplotlib AxesImage.
    """

    logger.info("Interferometric coherence")

    coherence = _prepare_coherence(
        input_file=input_file,
        processor=processor,
    )

    return _plot_raster(
        input_file=input_file,
        data=coherence,
        style=COHERENCE_STYLE,
        geometry_dir=geometry_dir,
        output=output,
        processor=processor,
        view=view,
        export_formats=export_formats,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
        variable_name="coherence",
        metadata={
            "minimum": float(np.nanmin(coherence)),
            "maximum": float(np.nanmax(coherence)),
            "mean": float(np.nanmean(coherence)),
        },
        show=show,
    )
