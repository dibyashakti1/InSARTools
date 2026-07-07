"""
insartools.unwrapped
====================

Visualization utilities for unwrapped InSAR interferograms.

This module provides publication-quality plotting of unwrapped phase
interferograms together with optional export to common scientific formats.

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .plot import PlotConfig
from .plot import imshow
from . import export

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]


DEFAULT_CMAP = "viridis"
DEFAULT_COLORBAR = "Unwrapped Phase (radians)"


def plot(
    phase: np.ndarray,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    title: str = "Unwrapped Interferogram",
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 300,
    cmap: str = DEFAULT_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    output: str | Path | None = None,
    save: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    show: bool = True,
):
    """
    Plot an unwrapped interferogram.

    Parameters
    ----------
    phase : ndarray
        Unwrapped phase in radians.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    title : str, default="Unwrapped Interferogram"
        Figure title.

    figsize : tuple, default=(8.0, 8.0)
        Figure size.

    dpi : int, default=300
        Figure resolution.

    cmap : str, default="viridis"
        Matplotlib colormap.

    vmin, vmax : float, optional
        Color limits.

    output : str or Path, optional
        Output filename without extension.

    save : list[str], optional
        Export formats.

    metadata : dict, optional
        Additional metadata.

    show : bool, default=True
        Display the figure.

    Returns
    -------
    fig : Figure

    ax : Axes

    image : AxesImage

    files : dict
        Dictionary containing exported files.
    """

    logger.info("Plotting unwrapped interferogram.")

    config = PlotConfig(
        cmap=cmap,
        figsize=figsize,
        dpi=dpi,
        title=title,
        colorbar=True,
        colorbar_label=DEFAULT_COLORBAR,
        vmin=vmin,
        vmax=vmax,
    )

    fig, ax, image = imshow(
        phase,
        latitude=latitude,
        longitude=longitude,
        config=config,
    )

    files = {}

    if save is None:
        save = []

    if output is not None:

        metadata_out = {
            "product": "unwrapped_phase",
            "rows": int(phase.shape[0]),
            "cols": int(phase.shape[1]),
            "dtype": str(phase.dtype),
            "unit": "radians",
        }

        if metadata is not None:
            metadata_out.update(metadata)

        files = export.save(
            figure=fig,
            data=phase,
            latitude=latitude,
            longitude=longitude,
            output=output,
            formats=save,
            metadata=metadata_out,
            variable_name="unwrapped_phase",
        )

        logger.info(
            "Exported unwrapped interferogram to %s",
            output,
        )

    if show:
        plt.show()

    return fig, ax, image, files
