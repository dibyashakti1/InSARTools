"""
===============================================================================
InSARTools

phase.py

Phase → Line-of-Sight (LOS) displacement conversion.

Author : Dibyashakti Panda
License: MIT
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import logging
import numpy as np
from osgeo import gdal

from .io import (
    read_raster,
    raster_info,
    write_envi,
    copy_header,
)

from .plotting import (
    quicklook,
    histogram,
    print_statistics,
    print_statistics_console,
)

from .sensors import get_sensor

from .exceptions import (
    UnsupportedProcessorError,
)

gdal.UseExceptions()


def detect_processor(driver: str) -> str:
    """
    Detect the raster processor from the GDAL driver.
    """

    supported = {
        "ROI_PAC": "ROI_PAC",
        "ENVI": "ENVI",
        "GTiff": "GeoTIFF",
    }

    if driver not in supported:
        raise UnsupportedProcessorError(
            f"Unsupported raster driver: {driver}"
        )

    return supported[driver]



def detect_phase_band(
    processor: str,
    bands: int,
) -> int:
    """
    Determine which raster band contains
    the unwrapped phase.
    """

    if processor == "ROI_PAC":

        if bands != 2:
            raise RuntimeError(
                "ROI_PAC unwrapped files should contain two bands."
            )

        return 2

    return 1

def get_wavelength(
    sensor: str,
    custom: float | None = None,
) -> float:
    """
    Return radar wavelength in metres.
    """

    if custom is not None:
        return custom

    return get_sensor(sensor).wavelength

def phase_to_los(
    phase: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """
    Convert phase (radians) to LOS displacement (metres).
    """

    return (
        phase * wavelength
        / (4.0 * np.pi)
    ).astype(np.float32)
