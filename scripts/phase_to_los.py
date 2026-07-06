#!/usr/bin/env python3

"""
Command-line interface for Phase → LOS conversion.
"""

import argparse
import logging
import time

from insartools.phase import convert_file


def main():

    parser = argparse.ArgumentParser(
        description="Convert unwrapped phase to LOS displacement"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input unwrapped interferogram"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="los_disp.rdr",
        help="Output LOS raster"
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
        help="Override radar wavelength (m)"
    )

    parser.add_argument(
        "--no-quicklook",
        action="store_true",
        help="Disable quicklook PNG"
    )

    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Disable histogram PNG"
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="phase_to_los.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    t0 = time.time()

    convert_file(
        input_file=args.input,
        output_file=args.output,
        sensor=args.sensor,
        wavelength=args.wavelength,
        make_quicklook=not args.no_quicklook,
        make_histogram=not args.no_histogram,
        logger=logging.getLogger(__name__),
    )

    print(f"\nFinished in {time.time()-t0:.2f} seconds")


if __name__ == "__main__":
    main()
