import numpy as np
import pytest
from skrf.mathFunctions import mag_2_db

from skrf_usimmics import constant_z_cpw, klopfenstein_cpw, klopfenstein_response

F = 9e9


@pytest.fixture(scope='module')
def klopfenstein(substrate, frequency):
    return klopfenstein_cpw(40e-6, 25e-6, 36.0, F, -30.0, 20, substrate, frequency)


def peak_s11_db(network, mask):
    return mag_2_db(np.abs(network.s[mask, 0, 0])).max()


def test_klopfenstein_design_values(klopfenstein):
    assert klopfenstein.length == pytest.approx(5.036e-3, abs=1e-6)
    assert klopfenstein.s[0] == pytest.approx(22.417e-6, abs=1e-9)
    assert klopfenstein.s[-1] == pytest.approx(7.186e-6, abs=1e-9)
    assert np.all(np.diff(klopfenstein.s) < 0)
    assert np.all(klopfenstein.w == 40e-6)


def test_klopfenstein_passband(klopfenstein, frequency):
    passband = frequency.f >= F
    assert peak_s11_db(klopfenstein.network(frequency), passband) == pytest.approx(-27.95, abs=0.01)
    assert peak_s11_db(klopfenstein.network(frequency, z_ports=(50, 36.0)), passband) == pytest.approx(-26.30, abs=0.01)


def test_klopfenstein_converges_to_analytic(substrate, frequency):
    line = klopfenstein_cpw(40e-6, 25e-6, 36.0, F, -30.0, 201, substrate, frequency)
    s11 = np.abs(line.network(frequency).s[:, 0, 0])
    assert np.abs(s11 - klopfenstein_response(line, frequency)).max() < 0.005


def test_constant_z_matched_endpoints(substrate, frequency):
    s2 = substrate.gap_for_z0(frequency, F, 10e-6, substrate.z0(frequency, F, 40e-6, 25e-6))
    line = constant_z_cpw(40e-6, 25e-6, 10e-6, s2, 500e-6, 20, substrate, frequency, F)
    assert (line.w[0], line.s[0], line.w[-1], line.s[-1]) == (40e-6, 25e-6, 10e-6, s2)
    assert np.ptp(line.z_sections) < 0.01
    assert line.length == pytest.approx(500e-6)
    assert abs(line.report['endpoint_mismatch_ohm']) < 1e-3
    assert line.report['s2_for_constant_z'] == pytest.approx(s2, abs=1e-9)
    assert peak_s11_db(line.network(frequency), slice(None)) < -60


def test_constant_z_mismatched_endpoints_ramp(substrate, frequency):
    line = constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 20, substrate, frequency, F)
    assert (line.s[0], line.s[-1]) == (25e-6, 10e-6)
    assert np.all(np.diff(line.z_sections) > 0)
    assert line.report['endpoint_mismatch_ohm'] == pytest.approx(58.436 - 51.027, abs=0.01)
    assert line.report['s2_for_constant_z'] == pytest.approx(6.262e-6, abs=1e-8)
    assert line.report['z_error_ohm'] < 1e-3


def test_constant_z_edge_angles(substrate, frequency):
    line = constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 20, substrate, frequency, F)
    dx = 500e-6 / 20
    assert line.report['inner_edge_angle_deg'] == pytest.approx(np.degrees(np.arctan((30e-6 / 19 / 2) / dx)), abs=0.01)
    assert line.report['outer_edge_angle_deg'] > line.report['inner_edge_angle_deg']
    assert line.report['min_gap'] == pytest.approx(line.s.min())


def test_constant_z_custom_width_profile(substrate, frequency):
    line = constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 21, substrate, frequency, F, w_profile=lambda t: t**2)
    assert line.w[10] == pytest.approx(40e-6 - 30e-6 * 0.25)
    assert (line.w[0], line.w[-1]) == (40e-6, 10e-6)


def test_constant_z_needs_two_sections(substrate, frequency):
    with pytest.raises(ValueError, match='at least 2'):
        constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 1, substrate, frequency, F)
