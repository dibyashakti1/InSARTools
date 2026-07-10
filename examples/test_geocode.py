"""
Test geocoding on an ISCE2 interferogram.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import insartools as insar

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

ifg = Path(
    "/mnt/d/Venezuela_Eq_2026/merged/interferograms/20260618_20260625/filt_fine.int"
)

geometry = Path(
    "/mnt/d/Venezuela_Eq_2026/merged/geom_reference"
)

# ------------------------------------------------------------
# Read interferogram
# ------------------------------------------------------------

print("Reading interferogram...")

complex_ifg = insar.io.read_raster(str(ifg))

phase = np.flipud(
    np.angle(complex_ifg)
).astype(np.float32)

# ------------------------------------------------------------
# Read geometry
# ------------------------------------------------------------

print("Reading geometry...")

lat, lon = insar.geocode.load_geometry(
    geometry
)

print("Latitude :", lat.shape)
print("Longitude:", lon.shape)

# ------------------------------------------------------------
# Geocode
# ------------------------------------------------------------

print("Geocoding...")

geo_phase, geo_lat, geo_lon = insar.geocode.geocode_array(
    data=phase,
    latitude=lat,
    longitude=lon,
    resolution=30,
    method="linear",
    output_file="wrapped_geo.tif",
)

print("Output shape:", geo_phase.shape)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

plt.figure(figsize=(7,6))

plt.imshow(
    geo_phase,
    origin="upper",
    cmap="twilight",
)

plt.title("Geocoded Wrapped Phase")

plt.colorbar()

plt.tight_layout()

plt.show()
