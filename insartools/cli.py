"""
===============================================================================

InSARTools

Command Line Interface

Author
------
Dibyashakti Panda

License
-------
MIT

===============================================================================
"""

from __future__ import annotations

import argparse
import logging

from . import amplitude
from . import coherence
from . import displacement
from . import unwrapped
from . import wrapped
from . import info
from . import info


logger = logging.getLogger(__name__)


def main() -> None:
    """
    InSARTools command-line interface.
    """

    parser = argparse.ArgumentParser(
        prog="insar",
        description="InSARTools command-line interface.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="InSARTools 0.3.0",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    wrapped_parser = subparsers.add_parser(
        "wrapped",
        help="Plot wrapped interferogram.",
    )

    wrapped_parser.add_argument(
        "input_file",
    )

    wrapped_parser.add_argument(
        "--geometry-dir",
    )

    wrapped_parser.add_argument(
        "--view",
        default="radar",
        choices=("radar", "geo"),
    )

    wrapped_parser.add_argument(
        "--output",
    )

    unwrapped_parser = subparsers.add_parser(
        "unwrapped",
        help="Plot unwrapped interferogram.",
    )

    unwrapped_parser.add_argument(
        "input_file",
    )

    unwrapped_parser.add_argument(
        "--geometry-dir",
    )

    unwrapped_parser.add_argument(
        "--view",
        default="radar",
        choices=("radar", "geo"),
    )

    unwrapped_parser.add_argument(
        "--output",
    )

    coherence_parser = subparsers.add_parser(
        "coherence",
        help="Plot coherence.",
    )

    coherence_parser.add_argument(
        "input_file",
    )

    coherence_parser.add_argument(
        "--geometry-dir",
    )

    coherence_parser.add_argument(
        "--view",
        default="radar",
        choices=("radar", "geo"),
    )

    coherence_parser.add_argument(
        "--output",
    )


    amplitude_parser = subparsers.add_parser(
        "amplitude",
        help="Plot amplitude image.",
    )

    amplitude_parser.add_argument(
        "input_file",
    )

    amplitude_parser.add_argument(
        "--geometry-dir",
    )

    amplitude_parser.add_argument(
        "--view",
        default="radar",
        choices=("radar", "geo"),
    )

    amplitude_parser.add_argument(
        "--output",
    )


    displacement_parser = subparsers.add_parser(
        "displacement",
        help="Plot LOS displacement.",
    )

    displacement_parser.add_argument(
        "input_file",
    )

    displacement_parser.add_argument(
        "--geometry-dir",
    )

    displacement_parser.add_argument(
        "--view",
        default="radar",
        choices=("radar", "geo"),
    )

    displacement_parser.add_argument(
        "--sensor",
        default="S1",
    )

    displacement_parser.add_argument(
        "--wavelength",
        type=float,
    )

    displacement_parser.add_argument(
        "--output",
    )


    info_parser = subparsers.add_parser(
        "info",
        help="Display raster information.",
    )

    info_parser.add_argument(
        "input_file",
    )

    info_parser.add_argument(
        "--sensor",
        default="S1",
    )

    info_parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if args.command == "wrapped":

        wrapped.plot(
            input_file=args.input_file,
            geometry_dir=args.geometry_dir,
            view=args.view,
            output=args.output,
            show=True,
        )

    elif args.command == "unwrapped":

        unwrapped.plot(
            input_file=args.input_file,
            geometry_dir=args.geometry_dir,
            view=args.view,
            output=args.output,
            show=True,
        )

    elif args.command == "coherence":

        coherence.plot(
            input_file=args.input_file,
            geometry_dir=args.geometry_dir,
            view=args.view,
            output=args.output,
            show=True,
        )

    elif args.command == "amplitude":

        amplitude.plot(
            input_file=args.input_file,
            geometry_dir=args.geometry_dir,
            view=args.view,
            output=args.output,
            show=True,
        )

    elif args.command == "displacement":

        displacement.plot(
            input_file=args.input_file,
            geometry_dir=args.geometry_dir,
            view=args.view,
            output=args.output,
            sensor=args.sensor,
            wavelength=args.wavelength,
            show=True,
        )
    elif args.command == "info":

        info.show(
            input_file=args.input_file,
            sensor=args.sensor,
            json_output=args.json,
        )



if __name__ == "__main__":
    main()
