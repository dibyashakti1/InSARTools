"""
insartools.wrapped
==================

Visualization utilities for wrapped InSAR interferograms.

This module provides publication-quality plotting of wrapped phase
interferograms together with optional export to common scientific formats.

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .plot import PlotConfig
from .plot import imshow
from . import export

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]

DEFAULT_CMAP = "twilight"

DEFAULT_VMIN = -np.pi

DEFAULT_VMAX = np.pi

DEFAULT_COLORBAR = "Wrapped Phase (radians)"

def plot(
    phase: np.ndarray,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    title: str = "Wrapped Interferogram",
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 300,
    cmap: str = DEFAULT_CMAP,
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    output: str | Path | None = None,
    save: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    show: bool = True,
):
    """
    Plot a wrapped interferogram.

    Parameters
    ----------
    phase : ndarray
        Wrapped phase in radians.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    title : str
        Figure title.

    output : str or Path, optional
        Output filename without extension.

    save : list[str], optional
        Export formats.

    Returns
    -------
    Figure
    Axes
    AxesImage
    """

    logger.info("Plotting wrapped interferogram.")

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
        lat=latitude,
        lon=longitude,
        config=config,
    )
    if save is None:
        save = []

    if output is not None:
        export.save(
            figure=fig,
            data=phase,
            latitude=latitude,
            longitude=longitude,
            output=output,
            formats=save,
            metadata=metadata,
            variable_name="wrapped_phase",
        )

    if show:
        fig.show()

    logger.info("Wrapped interferogram completed.")

    return fig, ax, image
