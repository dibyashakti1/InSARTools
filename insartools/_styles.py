"""
Default visualization styles for raster products.

Each RasterStyle defines the default title, colormap,
colorbar label and display limits for a raster product.

This module centralizes all visualization defaults.

"""

from __future__ import annotations

import numpy as np

from ._raster import RasterStyle

###############################################################################
# Wrapped phase
###############################################################################

WRAPPED_STYLE = RasterStyle(
    title="Wrapped Interferogram",
    cmap="twilight",
    colorbar_label="Wrapped Phase (rad)",
    variable_name="wrapped_phase",
    vmin=-np.pi,
    vmax=np.pi,
)

###############################################################################
# Unwrapped phase
###############################################################################

UNWRAPPED_STYLE = RasterStyle(
    title="Unwrapped Interferogram",
    cmap="RdYlBu",
    colorbar_label="LOS Displacement (m)",
    variable_name="unwrapped_phase",
)

###############################################################################
# Coherence
###############################################################################

COHERENCE_STYLE = RasterStyle(
    title="Interferometric Coherence",
    cmap="viridis",
    colorbar_label="Coherence",
    variable_name="coherence",
    vmin=0.0,
    vmax=1.0,
)

###############################################################################
# SAR Amplitude
###############################################################################

AMPLITUDE_STYLE = RasterStyle(
    title="SAR Amplitude",
    cmap="gray",
    colorbar_label="Amplitude",
    variable_name="amplitude",
)

###############################################################################
# LOS Displacement
###############################################################################

LOS_STYLE = RasterStyle(
    title="Line-of-Sight Displacement",
    cmap="RdYlBu",
    colorbar_label="LOS Displacement (m)",
    variable_name="los_displacement",
)

###############################################################################
# DEM
###############################################################################

DEM_STYLE = RasterStyle(
    title="Digital Elevation Model",
    cmap="terrain",
    colorbar_label="Elevation (m)",
    variable_name="dem",
)

###############################################################################
# Incidence Angle
###############################################################################

INCIDENCE_STYLE = RasterStyle(
    title="Incidence Angle",
    cmap="viridis",
    colorbar_label="Incidence Angle (°)",
    variable_name="incidence_angle",
)

###############################################################################
# Azimuth Angle
###############################################################################

AZIMUTH_STYLE = RasterStyle(
    title="Azimuth Angle",
    cmap="twilight",
    colorbar_label="Azimuth Angle (°)",
    variable_name="azimuth_angle",
)

###############################################################################
# Height
###############################################################################

HEIGHT_STYLE = RasterStyle(
    title="Height",
    cmap="terrain",
    colorbar_label="Height (m)",
    variable_name="height",
)
