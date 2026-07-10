"""
insartools.geocode
==================

Geocoding utilities for radar-coded InSAR products.

This module provides functions to interpolate radar-coordinate rasters onto
a regular geographic (latitude/longitude) grid using external latitude and
longitude lookup tables.

The primary public interfaces are

    geocode_array()
    geocode_raster()

which operate on in-memory NumPy arrays and raster files, respectively.

Supported interpolation methods
-------------------------------

- nearest
- linear
- cubic

The implementation is intentionally backend-independent except for the
interpolation routine (SciPy) and raster I/O (Rasterio).

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from osgeo import gdal

gdal.UseExceptions()

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None

try:
    from scipy.interpolate import griddata
except ImportError:
    griddata = None


logger = logging.getLogger(__name__)


InterpolationMethod = Literal[
    "nearest",
    "linear",
    "cubic",
]


__all__ = [
    "geocode_gdal",
    "load_geometry",
    "geocode_array",
    "geocode_raster",
]


###############################################################################
# Validation
###############################################################################


def _validate_inputs(
    data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    resolution: float,
    method: InterpolationMethod,
    mask: np.ndarray | None = None,
    resolution_unit: Literal["meters", "degrees"] = "meters",
) -> None:
    """
    Validate input arrays before interpolation.

    Parameters
    ----------
    data : ndarray
        Input raster.

    latitude : ndarray
        Latitude lookup table.

    longitude : ndarray
        Longitude lookup table.

    resolution : float
        Desired output grid resolution.

    method : {"nearest", "linear", "cubic"}
        Interpolation method.

    mask : ndarray, optional
        Boolean validity mask.

    resolution_unit : {"meters", "degrees"}, default="meters"
        Unit of the requested output resolution.

    Raises
    ------
    ValueError
        If any input is invalid.
    """

    if data.ndim != 2:
        raise ValueError("Input data must be two-dimensional.")

    if latitude.ndim != 2:
        raise ValueError("Latitude array must be two-dimensional.")

    if longitude.ndim != 2:
        raise ValueError("Longitude array must be two-dimensional.")

    if data.shape != latitude.shape:
        raise ValueError(
            "Data and latitude arrays must have identical shapes."
        )

    if data.shape != longitude.shape:
        raise ValueError(
            "Data and longitude arrays must have identical shapes."
        )

    if mask is not None:
        if mask.shape != data.shape:
            raise ValueError(
                "Mask must have the same shape as the input raster."
            )

        if mask.dtype != bool:
            raise ValueError("Mask must have boolean dtype.")

    if resolution <= 0:
        raise ValueError("Resolution must be positive.")

    if resolution_unit not in ("meters", "degrees"):
        raise ValueError(
            "resolution_unit must be either 'meters' or 'degrees'."
        )

    if method not in ("nearest", "linear", "cubic"):
        raise ValueError(
            f"Unsupported interpolation method '{method}'."
        )

    valid = np.isfinite(latitude) & np.isfinite(longitude)

    if not np.any(valid):
        raise ValueError(
            "Latitude/longitude lookup tables contain no valid coordinates."
        )

###############################################################################
# Output grid construction
###############################################################################

def _build_output_grid(
    latitude: np.ndarray,
    longitude: np.ndarray,
    resolution: float,
    resolution_unit: Literal["meters", "degrees"] = "meters",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a regular geographic output grid.

    Parameters
    ----------
    latitude : ndarray
        Latitude lookup table.

    longitude : ndarray
        Longitude lookup table.

    resolution : float
        Output resolution.

    resolution_unit : {"meters", "degrees"}, default="meters"
        Unit of the requested output resolution.

    Returns
    -------
    geo_lat : ndarray
        Latitude vector.

    geo_lon : ndarray
        Longitude vector.

    mesh_lon : ndarray
        Longitude meshgrid.

    mesh_lat : ndarray
        Latitude meshgrid.
    """

    valid = np.isfinite(latitude) & np.isfinite(longitude)

    lat_min = float(np.min(latitude[valid]))
    lat_max = float(np.max(latitude[valid]))

    lon_min = float(np.min(longitude[valid]))
    lon_max = float(np.max(longitude[valid]))

    mean_lat = float(np.mean(latitude[valid]))

    if resolution_unit == "degrees":
        lat_spacing = resolution
        lon_spacing = resolution
    else:
        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = (
            111_320.0 * np.cos(np.deg2rad(mean_lat))
        )

        lat_spacing = resolution / meters_per_degree_lat
        lon_spacing = resolution / meters_per_degree_lon

    logger.info(
        (
            "Output grid spacing: "
            "%.8f° latitude, %.8f° longitude"
        ),
        lat_spacing,
        lon_spacing,
    )

    geo_lat = np.arange(
        lat_min,
        lat_max + lat_spacing,
        lat_spacing,
        dtype=np.float64,
    )

    geo_lon = np.arange(
        lon_min,
        lon_max + lon_spacing,
        lon_spacing,
        dtype=np.float64,
    )

    mesh_lon, mesh_lat = np.meshgrid(
        geo_lon,
        geo_lat,
    )

    logger.info(
        "Output grid size: %d rows × %d columns",
        mesh_lat.shape[0],
        mesh_lat.shape[1],
    )

    return (
        geo_lat,
        geo_lon,
        mesh_lon,
        mesh_lat,
    )

###############################################################################
# Point preparation
###############################################################################


def _prepare_points(
    data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare interpolation points and values.

    Parameters
    ----------
    data : ndarray
        Radar-coordinate raster.

    latitude : ndarray
        Latitude lookup table.

    longitude : ndarray
        Longitude lookup table.

    mask : ndarray, optional
        Boolean mask indicating valid pixels.

    Returns
    -------
    points : ndarray of shape (N, 2)
        Coordinate pairs in (longitude, latitude) order.

    values : ndarray of shape (N,)
        Raster values corresponding to each point.
    """

    valid = (
        np.isfinite(data)
        & np.isfinite(latitude)
        & np.isfinite(longitude)
    )

    if mask is not None:
        valid &= mask

    if np.count_nonzero(valid) == 0:
        raise ValueError(
            "No valid pixels remain after applying the validity mask."
        )

    points = np.column_stack(
        (
            longitude[valid].ravel(),
            latitude[valid].ravel(),
        )
    )

    values = data[valid].ravel()

    logger.info(
        "Prepared %d valid interpolation points.",
        values.size,
    )

    return points, values


###############################################################################
# Interpolation
###############################################################################


def _interpolate(
    points: np.ndarray,
    values: np.ndarray,
    mesh_lon: np.ndarray,
    mesh_lat: np.ndarray,
    *,
    method: InterpolationMethod,
    fill_value: float,
) -> np.ndarray:
    """
    Interpolate scattered radar-coordinate samples onto a regular
    geographic grid.

    Parameters
    ----------
    points : ndarray
        Coordinate pairs in (longitude, latitude) order.

    values : ndarray
        Data values.

    mesh_lon : ndarray
        Longitude mesh.

    mesh_lat : ndarray
        Latitude mesh.

    method : {"nearest", "linear", "cubic"}
        Interpolation method.

    fill_value : float
        Value assigned outside the convex hull.

    Returns
    -------
    ndarray
        Geocoded raster.
    """

    if griddata is None:
        raise ImportError(
            "SciPy is required for geocoding. "
            "Install scipy to use this functionality."
        )

    logger.info(
        "Interpolating using '%s' method...",
        method,
    )

    geo_data = griddata(
        points,
        values,
        (mesh_lon, mesh_lat),
        method=method,
        fill_value=fill_value,
    )

    if geo_data is None:
        raise RuntimeError("Interpolation failed.")

    logger.info("Interpolation complete.")

    return geo_data.astype(np.float32, copy=False)

###############################################################################
# GeoTIFF writer
###############################################################################


def _write_geotiff(
    output_file: str | Path,
    data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    crs: str = "EPSG:4326",
    nodata: float | None = np.nan,
    overwrite: bool = False,
) -> Path:
    """
    Write a geocoded raster to a GeoTIFF file.

    Parameters
    ----------
    output_file : str or Path
        Output GeoTIFF path.

    data : ndarray
        Geocoded raster.

    latitude : ndarray
        Latitude vector (ascending).

    longitude : ndarray
        Longitude vector (ascending).

    crs : str, default="EPSG:4326"
        Coordinate reference system.

    nodata : float, optional
        NoData value written to the GeoTIFF.

    overwrite : bool, default=False
        Overwrite an existing file.

    Returns
    -------
    Path
        Path to the written GeoTIFF.
    """

    if rasterio is None:
        raise ImportError(
            "Rasterio is required to write GeoTIFF files."
        )

    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}"
        )

    if latitude.ndim != 1:
        raise ValueError("Latitude must be a one-dimensional vector.")

    if longitude.ndim != 1:
        raise ValueError("Longitude must be a one-dimensional vector.")

    if data.shape != (latitude.size, longitude.size):
        raise ValueError(
            "Raster dimensions do not match the coordinate vectors."
        )

    if latitude.size < 2 or longitude.size < 2:
        raise ValueError(
            "Coordinate vectors must contain at least two samples."
        )

    lat_res = float(np.mean(np.diff(latitude)))
    lon_res = float(np.mean(np.diff(longitude)))

    transform = from_origin(
        west=float(longitude.min()),
        north=float(latitude.max()),
        xsize=lon_res,
        ysize=abs(lat_res),
    )

    logger.info("Writing GeoTIFF: %s", output_file)

    with rasterio.open(
        output_file,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    logger.info("GeoTIFF successfully written.")

    return output_file

###############################################################################
# Geometry loader
###############################################################################

from .io import read_raster


def load_geometry(
    geometry_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ISCE2 latitude and longitude lookup tables.

    Parameters
    ----------
    geometry_dir : str or Path
        Directory containing lat.rdr and lon.rdr.

    Returns
    -------
    latitude : ndarray
    longitude : ndarray
    """

    geometry_dir = Path(geometry_dir)

    lat_file = geometry_dir / "lat.rdr"
    lon_file = geometry_dir / "lon.rdr"

    if not lat_file.exists():
        raise FileNotFoundError(lat_file)

    if not lon_file.exists():
        raise FileNotFoundError(lon_file)

    logger.info("Reading geometry lookup tables.")

    latitude = read_raster(str(lat_file))
    longitude = read_raster(str(lon_file))

    # ISCE commonly uses 0 outside the valid footprint.
    latitude = latitude.astype(np.float64)
    longitude = longitude.astype(np.float64)

    latitude[latitude == 0] = np.nan
    longitude[longitude == 0] = np.nan

    return latitude, longitude


def geocode_gdal(
    input_file: str | Path,
    geometry_dir: str | Path,
    output_file: str | Path,
    *,
    target_epsg: str = "EPSG:4326",
    resampling: str = "near",
    overwrite: bool = True,
) -> Path:
    """
    Geocode a radar-coordinate raster using GDAL geolocation arrays.

    Parameters
    ----------
    input_file : str or Path
        Radar-coordinate raster.

    geometry_dir : str or Path
        Directory containing lat.rdr and lon.rdr.

    output_file : str or Path
        Output GeoTIFF.

    target_epsg : str
        Output CRS.

    resampling : str
        GDAL resampling method.

    overwrite : bool
        Overwrite existing output.

    Returns
    -------
    Path
        Output GeoTIFF.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    logger.info("Starting GDAL geocoding.")
    logger.info("Input : %s", input_file)
    logger.info("Output: %s", output_file)

    if output_file.exists():

        if overwrite:
            output_file.unlink()

        else:
            raise FileExistsError(output_file)

    vrt_file = _create_geoloc_vrt(
        input_file,
        geometry_dir,
    )

    logger.info("Running GDAL Warp...")

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=target_epsg,
        geoloc=True,
        resampleAlg=resampling,
        creationOptions=[
            "COMPRESS=LZW",
            "TILED=YES",
        ],
    )

    result = gdal.Warp(
        str(output_file),
        str(vrt_file),
        options=warp_options,
    )

    if result is None:
        raise RuntimeError(
            "GDAL Warp failed."
        )

    result = None

    logger.info("Geocoding complete.")

    return output_file


def _create_geoloc_vrt(
    input_file: str | Path,
    geometry_dir: str | Path,
) -> Path:
    """
    Create a VRT with ISCE2 geolocation arrays attached.

    Parameters
    ----------
    input_file : str or Path
        Radar-coordinate raster.

    geometry_dir : str or Path
        Directory containing lat.rdr and lon.rdr.

    Returns
    -------
    Path
        Path to the generated VRT.
    """

    input_file = Path(input_file)
    geometry_dir = Path(geometry_dir)

    vrt_file = input_file.with_suffix(".vrt")

    ds = gdal.Open(str(input_file))

    if ds is None:
        raise RuntimeError(f"Cannot open {input_file}")

    driver = gdal.GetDriverByName("VRT")

    vrt = driver.CreateCopy(
        str(vrt_file),
        ds,
        0,
    )

    vrt.SetMetadata(
        {
            "X_DATASET": str(geometry_dir / "lon.rdr"),
            "X_BAND": "1",
            "Y_DATASET": str(geometry_dir / "lat.rdr"),
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "PIXEL_STEP": "1",
            "LINE_STEP": "1",
        },
        "GEOLOCATION",
    )

    vrt.FlushCache()

    vrt = None
    ds = None

    logger.info("Created VRT: %s", vrt_file)

    return vrt_file


###############################################################################
# Public API
###############################################################################

def geocode_array(
    data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    resolution: float = 30.0,
    resolution_unit: Literal["meters", "degrees"] = "meters",
    method: InterpolationMethod = "linear",
    mask: np.ndarray | None = None,
    fill_value: float = np.nan,
    output_file: str | Path | None = None,
    overwrite: bool = False,
):
    """
    Geocode an in-memory radar-coordinate array.

    Parameters
    ----------
    data : ndarray
        Radar-coordinate raster.

    latitude : ndarray
        Latitude lookup table.

    longitude : ndarray
        Longitude lookup table.

    resolution : float
        Output grid spacing.

    resolution_unit : {"meters", "degrees"}

    method : {"nearest", "linear", "cubic"}

    output_file : str or Path, optional
        If supplied, write a GeoTIFF.

    Returns
    -------
    geo_data : ndarray

    geo_lat : ndarray

    geo_lon : ndarray
    """

    logger.info("Starting geocoding.")

    _validate_inputs(
        data=data,
        latitude=latitude,
        longitude=longitude,
        resolution=resolution,
        resolution_unit=resolution_unit,
        method=method,
        mask=mask,
    )

    geo_lat, geo_lon, mesh_lon, mesh_lat = _build_output_grid(
        latitude,
        longitude,
        resolution,
        resolution_unit,
    )

    points, values = _prepare_points(
        data,
        latitude,
        longitude,
        mask,
    )

    geo_data = _interpolate(
        points,
        values,
        mesh_lon,
        mesh_lat,
        method=method,
        fill_value=fill_value,
    )

    if output_file is not None:

        _write_geotiff(
            output_file=output_file,
            data=geo_data,
            latitude=geo_lat,
            longitude=geo_lon,
            overwrite=overwrite,
        )

    logger.info("Geocoding complete.")

    return geo_data, geo_lat, geo_lon

###############################################################################
# Plot helper
###############################################################################

import tempfile

from .io import (
    read_raster,
    save_raster,
)



###############################################################################
# Geocode for visualization
###############################################################################

def geocode_for_plot(
    data: np.ndarray,
    geometry_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Geocode an in-memory raster for visualization.

    Parameters
    ----------
    data : ndarray
        Radar-coordinate raster.

    geometry_dir : str or Path
        Directory containing lat.rdr and lon.rdr.

    Returns
    -------
    geo_data : ndarray
        Geocoded raster.

    latitude : ndarray
        Latitude vector.

    longitude : ndarray
        Longitude vector.
    """

    from tempfile import TemporaryDirectory

    from osgeo import gdal

    from .io import save_raster

    with TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        radar_file = tmpdir / "temp.rdr"
        geo_file = tmpdir / "temp_geo.tif"

        save_raster(
            str(radar_file),
            data,
        )

        geocode_gdal(
            input_file=radar_file,
            geometry_dir=geometry_dir,
            output_file=geo_file,
        )

        ds = gdal.Open(str(geo_file))

        if ds is None:
            raise RuntimeError(
                f"Unable to open geocoded raster: {geo_file}"
            )

        gt = ds.GetGeoTransform()

        cols = ds.RasterXSize
        rows = ds.RasterYSize

        longitude = gt[0] + np.arange(cols) * gt[1]
        latitude = gt[3] + np.arange(rows) * gt[5]

        geo_data = ds.GetRasterBand(1).ReadAsArray()

        ds = None

        return (
            geo_data,
            latitude,
            longitude,
        )
