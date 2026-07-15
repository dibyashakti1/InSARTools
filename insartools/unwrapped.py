"""
insartools.unwrapped
====================

Visualization utilities for unwrapped InSAR interferograms.

This module provides publication-quality visualization of
unwrapped phase interferograms in both radar and geographic
coordinates.

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .io import read_raster

from ._raster import _plot_raster
from ._styles import UNWRAPPED_STYLE

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]

###############################################################################
# Defaults
###############################################################################

#DEFAULT_TITLE = "Unwrapped Interferogram"

#DEFAULT_CMAP = "RdBu_r"

#DEFAULT_COLORBAR = "Unwrapped Phase (radians)"

DEFAULT_EXPORT = (
    "png",
    "svg",
    "mat",
    "pdf",
    "tif",
    "kml",
)

###############################################################################
# Internal helpers
###############################################################################

def _prepare_unwrapped(
    input_file: str | Path,
    processor: str = "auto",
) -> np.ndarray:
    """
    Read an unwrapped interferogram.

    Parameters
    ----------
    input_file : str or Path
        Unwrapped interferogram.

    processor : str, default="auto"
        InSAR processor.

    Returns
    -------
    ndarray
        Unwrapped phase in radians.
    """

    logger.info(
        "Reading unwrapped interferogram: %s",
        input_file,
    )

    processor = processor.upper()

    if processor in ("AUTO", "ISCE2"):

        # ISCE2/ROI_PAC convention:
        # Band 1 -> amplitude
        # Band 2 -> unwrapped phase
        phase = read_raster(
            str(input_file),
            band=2,
        )

    else:
        raise NotImplementedError(
            f"Processor '{processor}' is not yet supported."
        )

    phase = phase.astype(
        np.float32,
        copy=False,
    )

    logger.info(
        "Raster size: %d × %d",
        phase.shape[0],
        phase.shape[1],
    )

    return phase

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
    Plot an unwrapped interferogram.

    Parameters
    ----------
    input_file : str or Path
        Unwrapped interferogram.

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

    logger.info("Unwrapped interferogram")

    phase = _prepare_unwrapped(
        input_file=input_file,
        processor=processor,
    )

    return _plot_raster(
        data=phase,
        style=UNWRAPPED_STYLE,
        geometry_dir=geometry_dir,
        output=output,
        processor=processor,
        view=view,
        export_formats=export_formats,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
        variable_name="unwrapped_phase",
        show=show,
    )
