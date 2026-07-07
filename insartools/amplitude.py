"""
insartools.amplitude
====================

Visualization utilities for SAR amplitude and intensity images.

This module provides publication-quality visualization of SAR amplitude
images with optional preprocessing including logarithmic scaling,
decibel conversion, gamma correction and percentile clipping.

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from . import export
from .plot import PlotConfig
from .plot import imshow

logger = logging.getLogger(__name__)

__all__ = [
    "plot",
]

DisplayMode = Literal[
    "linear",
    "log",
    "db",
]

DEFAULT_CMAP = "gray"

DEFAULT_COLORBAR = "Amplitude"


###############################################################################
# Internal preprocessing
###############################################################################


def _prepare_amplitude(
    amplitude: np.ndarray,
    *,
    mode: DisplayMode,
    gamma: float,
    clip_percentile: tuple[float, float] | None,
) -> np.ndarray:
    """
    Prepare an amplitude image for visualization.
    """

    image = np.asarray(
        amplitude,
        dtype=np.float32,
    ).copy()

    image[image < 0] = np.nan

    if mode == "log":
        image = np.log10(
            image + 1e-6,
        )

    elif mode == "db":
        image = 20.0 * np.log10(
            image + 1e-6,
        )

    if gamma != 1.0:
        image = np.power(
            image - np.nanmin(image),
            gamma,
        )

    if clip_percentile is not None:

        low, high = np.nanpercentile(
            image,
            clip_percentile,
        )

        image = np.clip(
            image,
            low,
            high,
        )

    return image

def plot(
    amplitude: np.ndarray,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    mode: DisplayMode = "db",
    gamma: float = 1.0,
    clip_percentile: tuple[float, float] | None = (2, 98),
    title: str = "SAR Amplitude",
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: int = 300,
    cmap: str = DEFAULT_CMAP,
    output: str | Path | None = None,
    save: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    show: bool = True,
):
    """
    Plot a SAR amplitude image.

    Parameters
    ----------
    amplitude : ndarray
        SAR amplitude image.

    mode : {"linear","log","db"}
        Display mode.

    gamma : float
        Gamma correction.

    clip_percentile : tuple, optional
        Percentile stretch.

    Returns
    -------
    fig
    ax
    image
    files
    """

    logger.info(
        "Preparing amplitude image."
    )

    image_data = _prepare_amplitude(
        amplitude,
        mode=mode,
        gamma=gamma,
        clip_percentile=clip_percentile,
    )

    if mode == "db":
        colorbar = "Amplitude (dB)"
    elif mode == "log":
        colorbar = "Log Amplitude"
    else:
        colorbar = DEFAULT_COLORBAR

    config = PlotConfig(
        cmap=cmap,
        figsize=figsize,
        dpi=dpi,
        title=title,
        colorbar_label=colorbar,
    )

    fig, ax, image = imshow(
        image_data,
        latitude=latitude,
        longitude=longitude,
        config=config,
    )

    files = {}

    if save is None:
        save = []

    if output is not None:

        metadata_out = {
            "product": "amplitude",
            "display_mode": mode,
            "gamma": gamma,
            "rows": int(amplitude.shape[0]),
            "cols": int(amplitude.shape[1]),
            "dtype": str(amplitude.dtype),
        }

        if metadata is not None:
            metadata_out.update(metadata)

        files = export.save(
            figure=fig,
            data=image_data,
            latitude=latitude,
            longitude=longitude,
            output=output,
            formats=save,
            metadata=metadata_out,
            variable_name="amplitude",
        )

    if show:
        plt.show()

    return fig, ax, image, files
