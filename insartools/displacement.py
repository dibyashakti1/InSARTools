"""
===============================================================================

InSARTools

displacement.py

Visualization of line-of-sight (LOS) displacement derived from
unwrapped interferograms.

Supports

- ISCE2
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

from pathlib import Path

import numpy as np

from .io import read_raster

from ._raster import _plot_raster

from ._styles import DISPLACEMENT_STYLE

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

DEFAULT_WAVELENGTH = 0.05546576


def _prepare_phase(
    input_file: str | Path,
) -> np.ndarray:
    """
    Read an unwrapped interferogram.
    """

    logger.info(
        "Reading unwrapped phase."
    )

    phase = read_raster(
        str(input_file),
    )

    return phase.astype(
        np.float32,
    )

def _phase_to_displacement(
    phase: np.ndarray,
    *,
    wavelength: float,
) -> np.ndarray:
    """
    Convert unwrapped phase to LOS displacement.
    """

    logger.info(
        "Converting phase to LOS displacement."
    )

    displacement = -(
        wavelength
        * phase
    ) / (
        4.0 * np.pi
    )

    return displacement.astype(
        np.float32,
    
    )
def plot(
    input_file: str | Path,
    *,
    wavelength: float = DEFAULT_WAVELENGTH,
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
    Plot LOS displacement derived from an unwrapped interferogram.
    """

    logger.info("Preparing LOS displacement.")

    phase = _prepare_phase(
        input_file,
    )

    displacement = _phase_to_displacement(
        phase,
        wavelength=wavelength,
    )

    return _plot_raster(
        data=displacement,
        style=DISPLACEMENT_STYLE,
        geometry_dir=geometry_dir,
        output=output,
        processor=processor,
        view=view,
        export_formats=export_formats,
        figsize=figsize,
        dpi=dpi,
        cmap=cmap,
        title=title,
        variable_name="displacement",
        metadata={
            "wavelength": wavelength,
            "minimum": float(np.nanmin(displacement)),
            "maximum": float(np.nanmax(displacement)),
            "mean": float(np.nanmean(displacement)),
        },
        show=show,
    )
