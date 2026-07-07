"""
===============================================================================
InSARTools

plotting.py

High-level plotting utilities built on top of plot.py.

Author : Dibyashakti Panda
License: MIT
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plot import PlotConfig, imshow, save


def quicklook(
    array: np.ndarray,
    output: str,
    cmap: str = "RdBu_r",
    title: str | None = None,
    unit: str = "m",
    dpi: int = 300,
) -> None:
    """
    Save a publication-quality quicklook PNG.
    """

    data = np.asarray(array, dtype=float)

    data[~np.isfinite(data)] = np.nan

    try:
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)
    except Exception:
        vmin = None
        vmax = None

    config = PlotConfig(
        cmap=cmap,
        title=title or "",
        colorbar_label=unit,
        vmin=vmin,
        vmax=vmax,
    )

    fig, ax, image = imshow(
        data,
        config=config,
    )

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save(
        fig,
        output,
        dpi=dpi,
    )

    plt.close(fig)


def histogram(
    array: np.ndarray,
    output: str,
    bins: int = 100,
    title: str = "Histogram",
    xlabel: str = "Value",
    dpi: int = 300,
) -> None:
    """
    Save a histogram of raster values.
    """

    values = np.asarray(array)

    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values,
        bins=bins,
    )

    ax.set_title(title)

    ax.set_xlabel(xlabel)

    ax.set_ylabel("Frequency")

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_statistics(
    array: np.ndarray,
) -> dict:
    """
    Compute raster statistics.
    """

    array = np.asarray(array)

    array = array[np.isfinite(array)]

    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def print_statistics_console(
    stats: dict,
) -> None:
    """
    Print statistics to the console.
    """

    print("\nOutput Statistics")
    print("-" * 35)

    print(f"Minimum : {stats['min']:12.5f} m")
    print(f"Maximum : {stats['max']:12.5f} m")
    print(f"Mean    : {stats['mean']:12.5f} m")
    print(f"Std Dev : {stats['std']:12.5f} m")
