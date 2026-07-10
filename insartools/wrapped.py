"""
===============================================================================

InSARTools

wrapped.py

Visualization of wrapped interferograms.

Supports

- ISCE2
- Complex interferograms
- Wrapped phase
- Publication-quality figures
- PNG
- GeoTIFF
- MATLAB
- KMZ

Author
------
Dibyashakti Panda

License
-------
MIT

===============================================================================
"""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from . import geocode
from .io import read_raster

from ._raster import _plot_raster
from ._styles import WRAPPED_STYLE

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]

###############################################################################
# Constants
###############################################################################

#DEFAULT_CMAP = "twilight"

#DEFAULT_TITLE = "Wrapped Interferogram"

#DEFAULT_COLORBAR = "Wrapped Phase (rad)"

DEFAULT_EXPORT = (
    "png",
)

###############################################################################
# Helper functions
###############################################################################


def _read_input(
    input_file: str | Path,
) -> np.ndarray:
    """
    Read an interferogram.
    """

    logger.info(
        "Reading %s",
        input_file,
    )

    return read_raster(
        str(input_file),
    )


def _to_wrapped_phase(
    data: np.ndarray,
) -> np.ndarray:
    """
    Convert complex interferogram
    to wrapped phase.
    """

    if np.iscomplexobj(data):
        logger.info("Complex interferogram detected. Computing wrapped phase.")
        logger.info(
            "Complex interferogram detected."
        )

        return np.angle(
            data,
        ).astype(np.float32)
    logger.info("Input is already a real-valued phase array.")
    return data.astype(
        np.float32,
    )


def _flip_if_required(
    phase: np.ndarray,
    flip: bool,
) -> np.ndarray:
    """
    Flip image vertically.
    """

    if flip:

        logger.info(
            "Applying flipud()."
        )

        return np.flipud(
            phase,
        )

    return phase

###############################################################################
# Processor utilities
###############################################################################

def _detect_processor(
    input_file: str | Path,
    processor: str,
) -> str:
    """
    Detect the processor that generated the interferogram.

    Parameters
    ----------
    input_file : str or Path
        Input interferogram.

    processor : str
        User-supplied processor name or "auto".

    Returns
    -------
    str
        Processor name.
    """

    if processor.lower() != "auto":
        return processor.upper()

    filename = Path(input_file).name.lower()

    if filename.endswith(".int"):
        return "ISCE2"

    return "UNKNOWN"


def _prepare_phase(
    input_file: str | Path,
    *,
    processor: str = "auto",
    flip: str | bool = "auto",
) -> np.ndarray:
    """
    Read an interferogram and prepare wrapped phase.

    This routine

    1. reads the raster,
    2. converts complex interferograms to wrapped phase,
    3. applies the correct orientation.
    """

    proc = _detect_processor(
        input_file,
        processor,
    )

    logger.info("Processor : %s", proc)

    data = _read_input(input_file)

    phase = _to_wrapped_phase(data)

    if flip == "auto":

        # Current convention:
        # ISCE2 radar products are displayed upside-down
        # relative to geographic north.

        flip_required = proc == "ISCE2"

    else:

        flip_required = bool(flip)

    phase = _flip_if_required(
        phase,
        flip_required,
    )

    return phase

def plot(
    input_file: str | Path,
    *,
    geometry_dir: str | Path | None = None,
    output: str | Path | None = None,
    processor: str = "auto",
    view: str = "radar",
    flip: str | bool = "auto",
    export_formats: tuple[str, ...] = DEFAULT_EXPORT,
    figsize: tuple[float, float] = (6.5, 5.5),
    dpi: int = 300,
    cmap: str | None = None,
    title: str | None = None,
    show: bool = False,
):
    """
    Plot a wrapped interferogram.

    Parameters
    ----------
    input_file : str or Path
        Complex interferogram (.int).

    geometry_dir : str or Path, optional
        Geometry directory used for geographic plotting.

    output : str or Path, optional
        Output filename without extension.

    processor : str, default="auto"
        InSAR processor.

    view : {"radar", "geo"}, default="radar"
        Radar-coordinate or geographic view.

    flip : bool or "auto"
        Flip the raster vertically.

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

    logger.info("Wrapped interferogram")

    phase = _prepare_phase(
        input_file=input_file,
        processor=processor,
        flip=flip,
    )

    return _plot_raster(
        data=phase,
        style=WRAPPED_STYLE,
        geometry_dir=geometry_dir,
        output=output,
        processor=processor,
        view=view,
        export_formats=export_formats,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
        variable_name="wrapped_phase",
        metadata=None,
        show=show,
    )
