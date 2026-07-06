"""
===============================================================================
InSARTools

io.py

Raster I/O utilities

Author : Dibyashakti Panda
License: MIT
===============================================================================
"""

from __future__ import annotations

import os
import shutil

import numpy as np
from osgeo import gdal

from .exceptions import InvalidRasterError

gdal.UseExceptions()


def read_raster(filename: str, band: int = 1) -> np.ndarray:
    """
    Read a raster band into a NumPy array.

    Parameters
    ----------
    filename : str
        Input raster.
    band : int
        Band number (1-based).

    Returns
    -------
    numpy.ndarray
    """

    ds = gdal.Open(filename)

    if ds is None:
        raise InvalidRasterError(
            f"Cannot open raster: {filename}"
        )

    arr = ds.GetRasterBand(band).ReadAsArray()

    ds = None

    return arr


def raster_info(filename: str) -> dict:
    """
    Read raster metadata.

    Returns
    -------
    dict
    """

    ds = gdal.Open(filename)

    if ds is None:
        raise InvalidRasterError(
            f"Cannot open raster: {filename}"
        )

    info = {

        "driver": ds.GetDriver().ShortName,
        "rows": ds.RasterYSize,
        "cols": ds.RasterXSize,
        "bands": ds.RasterCount

    }

    ds = None

    return info


def write_envi(
    filename: str,
    array: np.ndarray,
    dtype=gdal.GDT_Float32,
) -> None:
    """
    Write an ENVI raster.
    """

    rows, cols = array.shape

    driver = gdal.GetDriverByName("ENVI")

    ds = driver.Create(
        filename,
        cols,
        rows,
        1,
        dtype
    )

    ds.GetRasterBand(1).WriteArray(array)

    ds.FlushCache()

    ds = None


def copy_header(
    input_file: str,
    output_file: str,
) -> None:
    """
    Copy ENVI header if present.
    """

    hdr = os.path.splitext(input_file)[0] + ".hdr"

    if os.path.isfile(hdr):

        out_hdr = os.path.splitext(output_file)[0] + ".hdr"

        shutil.copy(hdr, out_hdr)


def save_numpy(
    filename: str,
    array: np.ndarray,
) -> None:
    """
    Save NumPy array.
    """

    np.save(filename, array)
