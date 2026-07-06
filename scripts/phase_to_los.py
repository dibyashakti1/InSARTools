#!/usr/bin/env python3
# =============================================================================
# InSARTools
#
# phase_to_los.py
#
# Convert unwrapped interferometric phase (radians)
# to Line-of-Sight (LOS) displacement (meters)
#
# Version : 2.0
# Author  : Dibyashakti Panda
#
# =============================================================================

import argparse
import logging
import os
import shutil
import sys
import time

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

VERSION = "2.0"

# ---------------------------------------------------------------------
# Supported Sensors
# ---------------------------------------------------------------------

SENSORS = {
    "S1": {
        "name": "Sentinel-1",
        "wavelength": 0.05546576
    },
    "ALOS2": {
        "name": "ALOS-2 PALSAR-2",
        "wavelength": 0.238403545
    }
}


# ---------------------------------------------------------------------
def banner():

    print("\n")
    print("=" * 75)
    print(" InSARTools : Phase → LOS Displacement")
    print(" Version :", VERSION)
    print("=" * 75)


# ---------------------------------------------------------------------
def setup_logger(logfile):

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        filemode="w"
    )


# ---------------------------------------------------------------------
def detect_processor(ds):

    driver = ds.GetDriver().ShortName

    if driver == "ROI_PAC":
        return "ROI_PAC"

    elif driver == "ENVI":
        return "ENVI"

    elif driver == "GTiff":
        return "GeoTIFF"

    else:
        return driver


# ---------------------------------------------------------------------
def detect_phase_band(processor, bands):

    if processor == "ROI_PAC":

        if bands != 2:
            raise RuntimeError(
                "ROI_PAC .unw should contain two bands."
            )

        return 2

    return 1


# ---------------------------------------------------------------------
def get_wavelength(sensor, custom):

    if custom is not None:
        return custom

    sensor = sensor.upper()

    if sensor not in SENSORS:

        raise RuntimeError(
            f"Unknown sensor : {sensor}"
        )

    return SENSORS[sensor]["wavelength"]


# ---------------------------------------------------------------------
def copy_header(input_file, output_file):

    hdr = os.path.splitext(input_file)[0] + ".hdr"

    if os.path.isfile(hdr):

        out_hdr = os.path.splitext(output_file)[0] + ".hdr"

        shutil.copy(hdr, out_hdr)


# ---------------------------------------------------------------------
def phase_to_los(phase, wavelength):

    return phase * wavelength / (4.0 * np.pi)


# ---------------------------------------------------------------------
def print_stats(arr):

    print("\nOutput Statistics")
    print("-" * 30)

    print(f"Minimum : {np.nanmin(arr):12.5f} m")
    print(f"Maximum : {np.nanmax(arr):12.5f} m")
    print(f"Mean    : {np.nanmean(arr):12.5f} m")
    print(f"Std Dev : {np.nanstd(arr):12.5f} m")


# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description="Convert phase to LOS displacement"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output",
        default="los_disp.rdr"
    )

    parser.add_argument(
        "--sensor",
        default="S1",
        choices=["S1", "ALOS2"]
    )

    parser.add_argument(
        "--wavelength",
        type=float,
        default=None,
        help="Override wavelength (meters)"
    )

    parser.add_argument(
        "--log",
        default="phase_to_los.log"
    )

    args = parser.parse_args()

    banner()

    setup_logger(args.log)

    t0 = time.time()

    if not os.path.isfile(args.input):

        sys.exit("Input file not found.")

    ds = gdal.Open(args.input)

    if ds is None:

        sys.exit("Cannot open input image.")

    processor = detect_processor(ds)

    wavelength = get_wavelength(
        args.sensor,
        args.wavelength
    )

    phase_band = detect_phase_band(
        processor,
        ds.RasterCount
    )

    print(f"Input file      : {args.input}")
    print(f"Output file     : {args.output}")
    print(f"Processor       : {processor}")
    print(f"Sensor          : {args.sensor}")
    print(f"Wavelength      : {wavelength:.9f} m")
    print(f"Phase Band      : {phase_band}")
    print(f"Image Size      : {ds.RasterXSize} x {ds.RasterYSize}")

    logging.info("Input : %s", args.input)
    logging.info("Output : %s", args.output)
    logging.info("Sensor : %s", args.sensor)
    logging.info("Processor : %s", processor)

    print("\nReading phase...")

    phase = ds.GetRasterBand(phase_band).ReadAsArray()

    print("Converting...")

    los = phase_to_los(
        phase,
        wavelength
    ).astype(np.float32)

    driver = gdal.GetDriverByName("ENVI")

    out = driver.Create(
        args.output,
        ds.RasterXSize,
        ds.RasterYSize,
        1,
        gdal.GDT_Float32
    )

    out.GetRasterBand(1).WriteArray(los)

    out.FlushCache()

    out = None
    ds = None

    copy_header(
        args.input,
        args.output
    )

    print_stats(los)

    logging.info(
        "Minimum %.6f",
        np.nanmin(los)
    )

    logging.info(
        "Maximum %.6f",
        np.nanmax(los)
    )

    logging.info(
        "Mean %.6f",
        np.nanmean(los)
    )

    logging.info(
        "Std %.6f",
        np.nanstd(los)
    )

    print("\nFinished Successfully.")
    print(f"Elapsed Time : {time.time()-t0:.2f} sec")
    print("=" * 75)


if __name__ == "__main__":

    main()
