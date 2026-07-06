import numpy as np

from insartools.phase import phase_to_los


def test_phase_conversion():

    wavelength = 0.05546576

    phase = np.array([0.0, np.pi])

    los = phase_to_los(
        phase,
        wavelength
    )

    expected = np.array([
        0.0,
        wavelength / 4.0
    ])

    assert np.allclose(
        los,
        expected
    )
