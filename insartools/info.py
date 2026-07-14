from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from .io import (
    raster_info,
    raster_dtype,
    read_raster,
)

#from .processor import detect_processor
from .sensors import get_sensor


def _band_names(
    filename: str,
    bands: int,
) -> list[str]:
    """
    Return descriptive names for raster bands.
    """

    suffix = Path(
        filename,
    ).suffix.lower()

    if suffix == ".unw" and bands >= 2:

        names = [
            "Amplitude",
            "Unwrapped Phase",
        ]

    elif suffix == ".cor":

        names = [
            "Coherence",
        ]

    elif suffix == ".int":

        names = [
            "Complex Interferogram",
        ]

    elif suffix == ".slc":

        names = [
            "Complex SLC",
        ]

    elif suffix == ".dem":

        names = [
            "Elevation",
        ]

    else:

        names = [
            f"Band {i}"
            for i in range(
                1,
                bands + 1,
            )
        ]

    while len(names) < bands:

        names.append(
            f"Band {len(names)+1}"
        )

    return names


def show(
    input_file: str | Path,
    *,
    sensor: str = "S1",
    json_output: bool = False,
):
    """
    Display raster information.
    """

    filename = str(input_file)

    info = raster_info(
        filename,
    )

    dtype = raster_dtype(
        filename,
    )

    sensor_info = get_sensor(
        sensor,
    )

    summary = {

        "file": Path(filename).name,

        "driver": info["driver"],

        "rows": info["rows"],

        "cols": info["cols"],

        "bands": info["bands"],

        "dtype": dtype,

        "sensor": sensor_info.full_name,

        "wavelength": sensor_info.wavelength,

    }

    band_names = _band_names(
        filename,
        info["bands"],
    )

    band_statistics = []

    for band in range(
        1,
        info["bands"] + 1,
    ):

        array = read_raster(
            filename,
            band=band,
        )

        if np.iscomplexobj(array):

            band_statistics.append({

                "band": band,

                "type": "Complex",

                "amplitude_min":
                    float(np.nanmin(np.abs(array))),

                "amplitude_max":
                    float(np.nanmax(np.abs(array))),

                "phase_min":
                    float(np.nanmin(np.angle(array))),

                "phase_max":
                    float(np.nanmax(np.angle(array))),

            })

        else:

            band_statistics.append({

                "band": band,

                "minimum":
                    float(np.nanmin(array)),

                "maximum":
                    float(np.nanmax(array)),

                "mean":
                    float(np.nanmean(array)),

            })


    if json_output:

        summary["statistics"] = band_statistics

        print(
            json.dumps(
                summary,
                indent=4,
            )
        )

        return

    print()

    print("=" * 60)
    print("               InSARTools Information")
    print("=" * 60)

    print("\nFile")
    print("-" * 60)

    print(f"Input file : {summary['file']}")
    print(f"Driver     : {summary['driver']}")

    print("\nRaster")
    print("-" * 60)

    print(f"Rows        : {summary['rows']}")
    print(f"Columns     : {summary['cols']}")
    print(f"Bands       : {summary['bands']}")
    print(f"Data Type   : {summary['dtype']}")

    print("\nSensor")
    print("-" * 60)

    print(f"Sensor      : {summary['sensor']}")
    print(f"Wavelength  : {summary['wavelength']} m")

    print("\nBand Statistics")
    print("-" * 60)

    for band in band_statistics:

        print(
            f"\nBand {band['band']} "
            f"({band_names[band['band'] - 1]})"
        )

        for key, value in band.items():

            if key == "band":
                continue

            label = key.replace(
                "_",
                " ",
            ).title()

            if isinstance(value, float):

                print(
                    f"    {label:16s}: {value:.3f}"
                )

            else:

                print(
                    f"    {label:16s}: {value}"
                )
