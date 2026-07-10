from pathlib import Path

import numpy as np
import insartools as insar
from insartools.io import read_raster

# -------------------------------------------------------
# Input ISCE2 interferogram
# -------------------------------------------------------

DATA = Path(
    "/mnt/d/Venezuela_Eq_2026/merged/interferograms/20260618_20260625"
)

wrapped_file = DATA / "filt_fine.int"

print("Reading:", wrapped_file)

# ISCE wrapped interferograms are complex.
complex_ifg = read_raster(
    str(wrapped_file)
)

phase = np.angle(complex_ifg).astype(np.float32)

print("Shape:", phase.shape)
print("dtype:", phase.dtype)

fig, ax, image = insar.wrapped.plot(
    phase,
    output=DATA / "wrapped_test",
    save=["png"],
)

print("Done.")
