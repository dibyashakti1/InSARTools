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
    save_raster,
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



    # =============================================================================
def compute_statistics(
    los: np.ndarray,
) -> dict:
    """
    Compute LOS displacement statistics.

    Parameters
    ----------
    los : ndarray
        LOS displacement array.

    Returns
    -------
    dict
        Dictionary containing min, max, mean and std.
    """

    return print_statistics(los)


# =============================================================================
def convert_file(
    input_file: str,
    output_file: str = "los_disp.rdr",
    sensor: str = "S1",
    wavelength: float | None = None,
    make_quicklook: bool = True,
    make_histogram: bool = True,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Convert an unwrapped interferogram to LOS displacement.

    Parameters
    ----------
    input_file : str
        Input interferogram.
    output_file : str
        Output ENVI raster.
    sensor : str
        SAR sensor.
    wavelength : float, optional
        Override radar wavelength.
    make_quicklook : bool
        Save quicklook PNG.
    make_histogram : bool
        Save histogram PNG.

    Returns
    -------
    dict
        Statistics dictionary.
    """

    info = raster_info(input_file)

    processor = detect_processor(
        info["driver"]
    )

    phase_band = detect_phase_band(
        processor,
        info["bands"]
    )

    wl = get_wavelength(
        sensor,
        wavelength
    )

    print("\nReading phase ...")

    phase = read_raster(
        input_file,
        band=phase_band
    )

    print("Converting phase → LOS ...")

    los = phase_to_los(
        phase,
        wl
    )

    print("Saving raster ...")

    save_raster(
        output_file,
        los
    )

    copy_header(
        input_file,
        output_file
    )

    stats = compute_statistics(
        los
    )

    print_statistics_console(
        stats
    )

    stem = Path(output_file).with_suffix("")

    if make_quicklook:

        print("Creating quicklook ...")

        quicklook(
            los,
            str(stem) + "_quicklook.png",
            title="LOS Displacement",
            unit="m"
        )

    if make_histogram:

        print("Creating histogram ...")

        histogram(
            los,
            str(stem) + "_histogram.png",
            xlabel="LOS displacement (m)"
        )

    if logger:

        logger.info(
            "Input : %s",
            input_file
        )

        logger.info(
            "Output : %s",
            output_file
        )

        logger.info(
            "Sensor : %s",
            sensor
        )

        logger.info(
            "Wavelength : %.9f",
            wl
        )

        for key, value in stats.items():

            logger.info(
                "%s : %.6f",
                key,
                value
            )

    return stats


# =============================================================================
def run(
    input_file: str,
    output_file: str = "los_disp.rdr",
    sensor: str = "S1",
    wavelength: float | None = None,
):
    """
    Convenience wrapper.
    """

    return convert_file(
        input_file=input_file,
        output_file=output_file,
        sensor=sensor,
        wavelength=wavelength,
    )
