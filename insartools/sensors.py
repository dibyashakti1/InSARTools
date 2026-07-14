"""
insartools.sensors
==================

Definitions of supported SAR sensors and radar wavelengths.

Author
------
Dibyashakti Panda

License
-------
MIT
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Sensor:
    """
    Representation of a SAR sensor.

    Parameters
    ----------
    short_name : str
        Short sensor identifier.
    full_name : str
        Full sensor name.
    wavelength : float
        Radar wavelength in metres.
    frequency : float
        Radar frequency in Hz.
    """

    short_name: str
    full_name: str
    wavelength: float
    frequency: float


SENSORS: Dict[str, Sensor] = {

    "S1": Sensor(
        short_name="S1",
        full_name="Sentinel-1",
        wavelength=0.05546576,
        frequency=5.405e9,
    ),

    "ALOS": Sensor(
        short_name="ALOS",
        full_name="ALOS PALSAR",
        wavelength=0.236057,
        frequency=1.270e9,
    ),

    "ALOS2": Sensor(
        short_name="ALOS2",
        full_name="ALOS-2 PALSAR-2",
        wavelength=0.238403545,
        frequency=1.2575e9,
    ),

    "NISAR-L": Sensor(
        short_name="NISAR-L",
        full_name="NISAR L-band",
        wavelength=0.238,
        frequency=1.257e9,

    ),

    "NISAR-S": Sensor(
        short_name="NISAR-S",
        full_name="NISAR S-band",
        wavelength=0.094,
        frequency=3.20e9,
    ),
}


def get_sensor(sensor: str) -> Sensor:
    """
    Return a Sensor object.

    Parameters
    ----------
    sensor : str
        Sensor short name.

    Returns
    -------
    Sensor

    Raises
    ------
    ValueError
        If sensor is unknown.
    """

    sensor = sensor.upper()

    if sensor not in SENSORS:
        raise ValueError(f"Unsupported sensor: {sensor}")

    return SENSORS[sensor]

def get_wavelength(sensor: str) -> float:
    """
    Return radar wavelength in metres.
    """

    return get_sensor(sensor).wavelength
