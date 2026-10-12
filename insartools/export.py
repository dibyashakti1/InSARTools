"""
insartools.export
=================

Export utilities for InSARTools.

This module provides functions for exporting figures and raster datasets
to common scientific formats including:

- PNG
- PDF
- SVG
- MATLAB (.mat)
- NumPy (.npz)
- GeoTIFF (.tif)

Author
------
InSARTools Development Team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import zipfile
from xml.etree.ElementTree import Element, SubElement, ElementTree

from matplotlib.figure import Figure

try:
    from scipy.io import savemat
except ImportError:
    savemat = None

try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio = None


logger = logging.getLogger(__name__)

__all__ = [
    "save",
    "save_figure",
    "save_mat",
    "save_npz",
    "save_geotiff",
    "save_kml",
    "save_kmz",
]

###############################################################################
# Helpers
###############################################################################


def _ensure_parent_directory(path: Path) -> None:
    """
    Create the parent directory if it does not exist.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _validate_coordinates(
    data: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> None:
    """
    Validate raster dimensions.
    """

    if latitude.ndim != 1:
        raise ValueError(
            "Latitude must be one-dimensional."
        )

    if longitude.ndim != 1:
        raise ValueError(
            "Longitude must be one-dimensional."
        )

    if latitude.size != data.shape[0]:
        raise ValueError(
            "Latitude size does not match raster."
        )

    if longitude.size != data.shape[1]:
        raise ValueError(
            "Longitude size does not match raster."
        )

###############################################################################
# Master export function
###############################################################################



###############################################################################
# Figure export
###############################################################################


def save_figure(
    figure: Figure,
    filename: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """
    Save a matplotlib figure.

    Parameters
    ----------
    figure : Figure
        Figure object.

    filename : str or Path
        Output filename.

    dpi : int
        Output resolution.

    transparent : bool
        Transparent background.

    Returns
    -------
    Path
    """

    filename = Path(filename)

    _ensure_parent_directory(filename)

    logger.info(
        "Saving figure: %s",
        filename,
    )

    figure.savefig(
        filename,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )

    return filename

###############################################################################
# MATLAB export
###############################################################################


def save_mat(
    data: np.ndarray,
    filename: str | Path,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    variable_name: str = "data",
) -> Path:
    """
    Save a raster dataset as a MATLAB MAT file.

    Parameters
    ----------
    data : ndarray
        Raster data.

    filename : str or Path
        Output MAT filename.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    metadata : dict, optional
        Metadata dictionary.

    variable_name : str
        Variable name used inside MATLAB.

    Returns
    -------
    Path
        Output filename.
    """

    if savemat is None:
        raise ImportError(
            "scipy is required for MATLAB export."
        )

    filename = Path(filename)

    _ensure_parent_directory(filename)

    output: dict[str, Any] = {
        variable_name: np.asarray(data),
    }

    if latitude is not None:
        output["latitude"] = np.asarray(latitude)

    if longitude is not None:
        output["longitude"] = np.asarray(longitude)

    if metadata is not None:
        output["metadata"] = metadata

    logger.info(
        "Saving MATLAB file: %s",
        filename,
    )

    savemat(
        filename,
        output,
        do_compression=True,
    )

    logger.info("MATLAB export complete.")

    return filename


###############################################################################
# NumPy export
###############################################################################


def save_npz(
    data: np.ndarray,
    filename: str | Path,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Save a compressed NumPy archive.

    Parameters
    ----------
    data : ndarray
        Raster data.

    filename : str or Path
        Output filename.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    metadata : dict, optional
        Metadata dictionary.

    Returns
    -------
    Path
    """

    filename = Path(filename)

    _ensure_parent_directory(filename)

    logger.info(
        "Saving NumPy archive: %s",
        filename,
    )

    output: dict[str, Any] = {
        "data": np.asarray(data),
    }

    if latitude is not None:
        output["latitude"] = np.asarray(latitude)

    if longitude is not None:
        output["longitude"] = np.asarray(longitude)

    if metadata is not None:
        output["metadata"] = np.array(
            metadata,
            dtype=object,
        )

    np.savez_compressed(
        filename,
        **output,
    )

    logger.info("NumPy export complete.")

    return filename
###############################################################################
# GeoTIFF export
###############################################################################


def save_geotiff(
    data: np.ndarray,
    filename: str | Path,
    *,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    transform=None,
    crs: str = "EPSG:4326",
    nodata: float | None = np.nan,
    compress: str = "lzw",
) -> Path:
    """
    Save a raster as a GeoTIFF.

    Parameters
    ----------
    data : ndarray
        Raster data.

    filename : str or Path
        Output GeoTIFF filename.

    latitude : ndarray, optional
        Latitude vector.

    longitude : ndarray, optional
        Longitude vector.

    transform : Affine, optional
        Rasterio affine transform.

    crs : str
        Coordinate reference system.

    nodata : float, optional
        NoData value.

    compress : str
        Compression method.

    Returns
    -------
    Path
    """

    if rasterio is None:
        raise ImportError(
            "Rasterio is required for GeoTIFF export."
        )

    filename = Path(filename)

    _ensure_parent_directory(filename)

    if transform is None:

        if latitude is None or longitude is None:
            raise ValueError(
                "Either transform or latitude/longitude "
                "must be supplied."
            )

        _validate_coordinates(
            data,
            latitude,
            longitude,
        )

        if latitude.size < 2 or longitude.size < 2:
            raise ValueError(
                "Coordinate vectors must contain "
                "at least two values."
            )

        dx = float(np.mean(np.diff(longitude)))
        dy = float(np.mean(np.diff(latitude)))

        transform = from_origin(
            west=float(longitude.min()),
            north=float(latitude.max()),
            xsize=dx,
            ysize=abs(dy),
        )

    logger.info(
        "Saving GeoTIFF: %s",
        filename,
    )

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        transform=transform,
        crs=crs,
        nodata=nodata,
        compress=compress,
    ) as dst:

        dst.write(
            np.asarray(data),
            1,
        )

    logger.info("GeoTIFF export complete.")

    return filename

###############################################################################
# Latitude and longitude GeoTIFF export
###############################################################################


def save_coordinate_geotiffs(
    latitude: np.ndarray,
    longitude: np.ndarray,
    output: str | Path,
    *,
    transform,
    crs: str = "EPSG:4326",
) -> dict[str, Path]:
    """
    Save latitude and longitude coordinate grids as GeoTIFF files.

    The coordinate rasters use the same dimensions and grid transform
    convention as the corresponding data GeoTIFF.

    Parameters
    ----------
    latitude : ndarray
        One-dimensional latitude vector.

    longitude : ndarray
        One-dimensional longitude vector.

    output : str or Path
        Output filename prefix without extension.

    Returns
    -------
    dict[str, Path]
        Paths to the latitude and longitude GeoTIFF files.
    """

    if rasterio is None:
        raise ImportError(
            "Rasterio is required for coordinate GeoTIFF export."
        )

    latitude = np.asarray(latitude, dtype=np.float64).squeeze()
    longitude = np.asarray(longitude, dtype=np.float64).squeeze()

    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError(
            "Latitude and longitude must be one-dimensional vectors."
        )

    if latitude.size < 2 or longitude.size < 2:
        raise ValueError(
            "Latitude and longitude vectors must contain "
            "at least two values."
        )

    if not np.all(np.isfinite(latitude)):
        raise ValueError("Latitude contains non-finite values.")

    if not np.all(np.isfinite(longitude)):
        raise ValueError("Longitude contains non-finite values.")

    if np.any(np.abs(latitude) > 90):
        raise ValueError("Latitude values must be between -90 and 90.")

    if np.any(np.abs(longitude) > 180):
        raise ValueError("Longitude values must be between -180 and 180.")

    dlat = np.diff(latitude)
    dlon = np.diff(longitude)

    if not (
        np.all(dlat > 0) or np.all(dlat < 0)
    ):
        raise ValueError("Latitude must be strictly monotonic.")

    if not np.all(dlon > 0):
        raise ValueError(
            "Longitude must increase from left to right "
            "for GeoTIFF export."
        )

    if not np.allclose(dlat, dlat[0], rtol=1e-5, atol=1e-10):
        raise ValueError("Latitude coordinates are not regularly spaced.")

    if not np.allclose(dlon, dlon[0], rtol=1e-5, atol=1e-10):
        raise ValueError("Longitude coordinates are not regularly spaced.")

    output = Path(output).with_suffix("")
    _ensure_parent_directory(output)

    output = Path(output).with_suffix("")
    _ensure_parent_directory(output)

    # Generate pixel-center coordinates from the exact affine transform
    # of the corresponding data GeoTIFF.
    height = latitude.size
    width = longitude.size

    col_grid, row_grid = np.meshgrid(
        np.arange(width, dtype=np.float64) + 0.5,
        np.arange(height, dtype=np.float64) + 0.5,
    )

    lon_grid = (
        transform.a * col_grid
        + transform.b * row_grid
        + transform.c
    )

    lat_grid = (
        transform.d * col_grid
        + transform.e * row_grid
        + transform.f
    )

    coordinate_grids = {
        "latitude": lat_grid.astype(np.float32),
        "longitude": lon_grid.astype(np.float32),
    }

    files: dict[str, Path] = {}

    for name, grid in coordinate_grids.items():

        filename = output.with_name(
            f"{output.name}_{name}"
        ).with_suffix(".tif")

        with rasterio.open(
            filename,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=np.nan,
            compress="lzw",
        ) as dst:

            dst.write(grid, 1)

        files[name] = filename

        logger.info(
            "Saved %s coordinate GeoTIFF: %s",
            name,
            filename,
        )

    return files

###############################################################################
# Metadata export
###############################################################################


def save_json(
    metadata: dict[str, Any],
    filename: str | Path,
) -> Path:
    """
    Save metadata as a JSON file.
    """

    filename = Path(filename)

    _ensure_parent_directory(filename)

    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(
            metadata,
            fp,
            indent=4,
        )

    logger.info("Metadata written to %s", filename)

    return filename


###############################################################################
# Google Earth export
###############################################################################


def save_kml(
    image_file: str | Path,
    filename: str | Path,
    *,
    west: float,
    east: float,
    south: float,
    north: float,
    name: str = "InSARTools Overlay",
    draw_order: int = 1,
) -> Path:
    """
    Create a Google Earth GroundOverlay KML.
    """

    filename = Path(filename)

    _ensure_parent_directory(filename)

    root = Element("kml")
    root.set("xmlns", "http://www.opengis.net/kml/2.2")

    document = SubElement(root, "Document")

    overlay = SubElement(document, "GroundOverlay")

    SubElement(overlay, "name").text = name

    SubElement(overlay, "drawOrder").text = str(draw_order)

    icon = SubElement(overlay, "Icon")

    SubElement(icon, "href").text = Path(image_file).name

    box = SubElement(overlay, "LatLonBox")

    SubElement(box, "north").text = str(north)

    SubElement(box, "south").text = str(south)

    SubElement(box, "east").text = str(east)

    SubElement(box, "west").text = str(west)

    tree = ElementTree(root)

    tree.write(
        filename,
        encoding="utf-8",
        xml_declaration=True,
    )

    logger.info("KML written to %s", filename)

    return filename


def save_kmz(
    image_file: str | Path,
    kmz_file: str | Path,
    *,
    west: float,
    east: float,
    south: float,
    north: float,
    name: str = "InSARTools Overlay",
) -> Path:
    """
    Create a KMZ archive containing a GroundOverlay.
    """

    kmz_file = Path(kmz_file)

    _ensure_parent_directory(kmz_file)

    temp_kml = kmz_file.with_suffix(".kml")

    save_kml(
        image_file=image_file,
        filename=temp_kml,
        west=west,
        east=east,
        south=south,
        north=north,
        name=name,
    )

    with zipfile.ZipFile(
        kmz_file,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.write(
            temp_kml,
            arcname="doc.kml",
        )

        archive.write(
            image_file,
            arcname=Path(image_file).name,
        )

    temp_kml.unlink(missing_ok=True)

    logger.info("KMZ written to %s", kmz_file)

    return kmz_file

###############################################################################
# Google Earth overlay image
###############################################################################

def save_overlay_png(

    data: np.ndarray,

    filename: str | Path,

    *,

    cmap="viridis",

    vmin=None,

    vmax=None,

):

    """

    Save a borderless raster image for Google Earth overlays.

    """

    import matplotlib.pyplot as plt

    filename = Path(filename)

    _ensure_parent_directory(filename)

    plt.imsave(

        filename,

        data,

        cmap=cmap,

        vmin=vmin,

        vmax=vmax,

    )

    logger.info(

        "Overlay PNG written to %s",

        filename,

    )

    return filename


###############################################################################
# Google Earth overlay PNG
###############################################################################

def save_overlay_png(
    data: np.ndarray,
    filename: str | Path,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """
    Save a borderless PNG for Google Earth overlays.
    """

    import matplotlib.pyplot as plt

    filename = Path(filename)

    _ensure_parent_directory(filename)

    masked = np.ma.masked_invalid(data)

    plt.imsave(
        filename,
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    logger.info(
        "Overlay PNG written to %s",
        filename,
    )

    return filename


###############################################################################
# Unified export dispatcher
###############################################################################


def save(
    *,
    figure: Figure | None = None,
    data: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    output: str | Path,
    formats: list[str],
    metadata: dict[str, Any] | None = None,
    variable_name: str = "data",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Path]:
    """
    Export data and/or figures to one or more formats.

    Parameters
    ----------
    figure : Figure, optional
        Matplotlib figure.

    data : ndarray, optional
        Raster data.

    latitude, longitude : ndarray, optional
        Coordinate vectors.

    output : str or Path
        Output filename without extension.

    formats : list[str]
        Export formats.

    metadata : dict, optional
        Metadata dictionary.

    variable_name : str
        Variable name used in MATLAB export.

    Returns
    -------
    dict
        Dictionary mapping format names to output files.
    """

    output = Path(output).with_suffix("")

    files: dict[str, Path] = {}

    if metadata is None and data is not None:
        metadata = {
            "rows": int(data.shape[0]),
            "cols": int(data.shape[1]),
            "dtype": str(data.dtype),
        }
    png = None
    overlay_png = None
    if figure is not None:

        # Publication PNG
        if "png" in formats:
            png = save_figure(
                figure,
                output.with_suffix(".png"),
            )
            files["png"] = png

        # Google Earth overlay PNG (no axes, no labels)
        if (
            data is not None
            and (
                "kml" in formats
                or "kmz" in formats
            )
        ):

            overlay_png = save_overlay_png(
                data,
                output.with_name(
                    output.name + "_overlay"
                ).with_suffix(".png"),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        if "pdf" in formats:
            files["pdf"] = save_figure(
                figure,
                output.with_suffix(".pdf"),
            )

        if "svg" in formats:
            files["svg"] = save_figure(
                figure,
                output.with_suffix(".svg"),
            )

    if data is not None:

        if "mat" in formats:
            files["mat"] = save_mat(
                data,
                output.with_suffix(".mat"),
                latitude=latitude,
                longitude=longitude,
                metadata=metadata,
                variable_name=variable_name,
            )

        if "npz" in formats:
            files["npz"] = save_npz(
                data,
                output.with_suffix(".npz"),
                latitude=latitude,
                longitude=longitude,
                metadata=metadata,
            )

        if "tif" in formats:

            if latitude is None or longitude is None:
                raise ValueError(
                    "GeoTIFF export requires latitude and longitude."
                )

            print(">>> Writing GeoTIFF...")

            data_tif = save_geotiff(
                data,
                output.with_suffix(".tif"),
                latitude=latitude,
                longitude=longitude,
            )

            files["tif"] = data_tif

            print(">>> GeoTIFF finished.")

            coordinate_products = {
                "coherence",
                "wrapped_phase",
                "unwrapped_phase",
            }

            if variable_name in coordinate_products:

                if rasterio is None:
                    raise ImportError(
                        "Rasterio is required for coordinate GeoTIFF export."
                    )

                with rasterio.open(data_tif) as src:
                    data_transform = src.transform
                    data_crs = src.crs

                print(">>> Writing latitude and longitude GeoTIFFs...")

                coordinate_files = save_coordinate_geotiffs(
                    latitude=latitude,
                    longitude=longitude,
                    output=output,
                    transform=data_transform,
                    crs=str(data_crs),
                )

                files.update(
                    {
                        f"{name}_tif": path
                        for name, path in coordinate_files.items()
                    }
                )

                print(">>> Coordinate GeoTIFFs finished.")



    if metadata is not None and "json" in formats:
        files["json"] = save_json(
            metadata,
            output.with_suffix(".json"),
        )

    if ("kml" in formats or "kmz" in formats):

        if latitude is None or longitude is None:
            raise ValueError(
                "Google Earth export requires latitude and longitude."
            )

        png = output.with_suffix(".png")

        if not png.exists():

            if figure is None:
                raise ValueError(
                    "PNG export requires a matplotlib Figure."
                )

            save_figure(
                figure,
                png,
                transparent=True,
            )

        west = float(longitude.min())
        east = float(longitude.max())
        south = float(latitude.min())
        north = float(latitude.max())

        if "kml" in formats:
            files["kml"] = save_kml(
                image_file=overlay_png,
                filename=output.with_suffix(".kml"),
                west=west,
                east=east,
                south=south,
                north=north,
            )

        if "kmz" in formats:
            files["kmz"] = save_kmz(
                image_file=overlay_png,
                kmz_file=output.with_suffix(".kmz"),
                west=west,
                east=east,
                south=south,
                north=north,
            )

    return files
