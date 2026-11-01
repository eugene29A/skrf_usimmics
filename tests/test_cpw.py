import pytest

from skrf_usimmics import Substrate


def test_known_impedance(substrate, frequency):
    assert substrate.z0(frequency, 9e9, 40e-6, 25e-6) == pytest.approx(51.03, abs=0.01)


@pytest.mark.parametrize('w, z', [(40e-6, 36.0), (40e-6, 50.0), (10e-6, 51.0), (640e-6, 51.0)])
def test_gap_for_z0_inverts_z0(substrate, frequency, w, z):
    s = substrate.gap_for_z0(frequency, 9e9, w, z)
    assert substrate.z0(frequency, 9e9, w, s) == pytest.approx(z, abs=1e-3)


def test_gap_for_z0_widens_bracket(substrate, frequency):
    # At W = 640 um, 51 ohm needs a gap above the default 200 um upper bound
    assert substrate.gap_for_z0(frequency, 9e9, 640e-6, 51.0) > 200e-6


def test_gap_for_z0_with_metal_thickness(frequency):
    # The model returns NaN for gaps small relative to t, including the default lower bound
    thick = Substrate(h=500e-6, ep_r=11.7, t=2e-6, rho=2.4e-8)
    s = thick.gap_for_z0(frequency, 9e9, 40e-6, 36.0)
    assert thick.z0(frequency, 9e9, 40e-6, s) == pytest.approx(36.0, abs=1e-3)


def test_gap_for_z0_rejects_impossible_impedance(substrate, frequency):
    with pytest.raises(ValueError, match='No gap'):
        substrate.gap_for_z0(frequency, 9e9, 40e-6, 1000.0)
