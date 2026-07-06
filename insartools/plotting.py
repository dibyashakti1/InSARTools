"""
===============================================================================
InSARTools

plotting.py

Plotting utilities for InSAR products.

Author : Dibyashakti Panda
License: MIT
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def quicklook(
    array: np.ndarray,
    output: str,
    cmap: str = "RdBu_r",
    title: str | None = None,
    unit: str = "m",
    dpi: int = 300,
) -> None:
    """
    Save a quicklook PNG of a raster.

    Parameters
    ----------
    array : ndarray
        Input raster.
    output : str
        Output PNG filename.
    cmap : str
        Matplotlib colormap.
    title : str, optional
        Figure title.
    unit : str
        Colorbar label.
    dpi : int
        Output resolution.
    """

    data = np.array(array, dtype=float)

    data[~np.isfinite(data)] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    # Robust display range (ignore extreme outliers)
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)

    im = ax.imshow(
        data,
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.75,
        pad=0.02
    )

    cbar.set_label(unit)

    fig.tight_layout()

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight"
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
    Save histogram of raster values.
    """

    values = array[np.isfinite(array)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        values,
        bins=bins
    )

    ax.set_title(title)

    ax.set_xlabel(xlabel)

    ax.set_ylabel("Frequency")

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close(fig)


def print_statistics(array: np.ndarray) -> dict:
    """
    Compute raster statistics.

    Returns
    -------
    dict
    """

    stats = {

        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array))

    }

    return stats


def print_statistics_console(stats: dict) -> None:
    """
    Pretty-print statistics.
    """

    print("\nOutput Statistics")
    print("-" * 35)

    print(f"Minimum : {stats['min']:12.5f} m")
    print(f"Maximum : {stats['max']:12.5f} m")
    print(f"Mean    : {stats['mean']:12.5f} m")
    print(f"Std Dev : {stats['std']:12.5f} m")
