from pathlib import Path

import insartools as insar

input_file = Path(
    "/mnt/d/Venezuela_Eq_2026/merged/interferograms/20260618_20260625/phase.tif"
)

geometry = Path(
    "/mnt/d/Venezuela_Eq_2026/merged/geom_reference"
)

output = Path("phase_geo.tif")

insar.geocode.geocode_gdal(
    input_file=input_file,
    geometry_dir=geometry,
    output_file=output,
)

print("Done!")
