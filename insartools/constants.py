"""
Physical constants used throughout InSARTools.
"""

import numpy as np

PI = np.pi

LIGHT_SPEED = 299792458.0

SENSORS = {

    "S1": {
        "name": "Sentinel-1",
        "wavelength": 0.05546576
    },

    "ALOS": {
        "name": "ALOS PALSAR",
        "wavelength": 0.236057
    },

    "ALOS2": {
        "name": "ALOS-2 PALSAR-2",
        "wavelength": 0.238403545
    }

}
