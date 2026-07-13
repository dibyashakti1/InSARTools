"""
===============================================================================

InSARTools

amplitude.py

Visualization of SAR amplitude images.

Supports

- ISCE2
- Amplitude rasters
- Radar coordinates
- Geographic coordinates
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

from dataclasses import replace

from pathlib import Path

import numpy as np

from .io import read_raster

from ._raster import _plot_raster

from ._styles import AMPLITUDE_STYLE

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]


###############################################################################
# Defaults
###############################################################################

DEFAULT_EXPORT = (
    "png",
    "svg",
    "pdf",
    "mat",
    "tif",
)


def _prepare_amplitude(
    input_file: str | Path,
) -> np.ndarray:
    """
    Read an amplitude image.

    Complex rasters are converted to amplitude using their magnitude.
    """

    logger.info("Reading amplitude image.")

    amplitude = read_raster(
        str(input_file),
    )

    if np.iscomplexobj(amplitude):

        logger.info(
            "Complex raster detected. Computing amplitude."
        )

        amplitude = np.abs(amplitude)

    return amplitude.astype(
        np.float32,
    )

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
    Plot an amplitude image.

    Parameters
    ----------
    input_file
        Input raster (complex interferogram, SLC or amplitude image).

    geometry_dir
        Geometry directory for geographic plotting.

    output
        Output filename without extension.

    processor
        InSAR processor.

    view
        "radar" or "geo".

    export_formats
        Export formats.

    figsize
        Figure size.

    dpi
        Figure resolution.

    cmap
        Optional colormap override.

    title
        Optional title override.

    show
        Display figure.

    Returns
    -------
    fig
    ax
    image
    """

    logger.info("Preparing amplitude image.")

    amplitude = _prepare_amplitude(
        input_file,
    )

    
###############################################################################
# Display scaling
###############################################################################

    display = 20.0 * np.log10(
        np.maximum(amplitude, 1.0)
    )

    vmin = float(
        np.nanpercentile(display, 2)
    )

    vmax = float(
        np.nanpercentile(display, 98)
    )

    style = replace(
        AMPLITUDE_STYLE,
        colorbar_label="Amplitude (dB)",
        vmin=vmin,
        vmax=vmax,
    )

###############################################################################
# Display image
###############################################################################

    display = 20.0 * np.log10(
        np.maximum(amplitude, 1.0)
    )
    style = replace(
        AMPLITUDE_STYLE,
        colorbar_label="Amplitude (dB)",
        vmin=vmin,
        vmax=vmax,
    )
    vmin = float(np.nanpercentile(display, 2))
    vmax = float(np.nanpercentile(display, 98))


    return _plot_raster(
        data=display,
        style=style,
        geometry_dir=geometry_dir,
        output=output,
        processor=processor,
        view=view,
        export_formats=export_formats,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
        variable_name="amplitude",
        metadata={
            "minimum": float(np.nanmin(amplitude)),
            "maximum": float(np.nanmax(amplitude)),
            "mean": float(np.nanmean(amplitude)),
        },
        show=show,
    )
