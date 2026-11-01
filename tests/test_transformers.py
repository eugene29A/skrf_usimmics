import os

import numpy as np
import pytest
from skrf.mathFunctions import mag_2_db

from skrf_usimmics import klopfenstein_cpw, transformer_for_length, transformer_for_reflection
from skrf_usimmics.usimmics import read_sp, run_simulation, simulator_path, write_schematic

F = 9e9
GEOMETRY = (40e-6, 25e-6, 20e-6, 4e-6)
needs_simulator = pytest.mark.skipif(not os.path.exists(simulator_path()), reason='uSimmics simulator not found')


@pytest.fixture(scope='module')
def designed(substrate, frequency):
    return transformer_for_reflection(*GEOMETRY, -30.0, F, substrate, frequency)


def peak_db(line, frequency, f_c=F, f_max=np.inf):
    band = (frequency.f >= f_c) & (frequency.f <= f_max)
    return mag_2_db(np.abs(line.network(frequency).s[band, 0, 0])).max()


def test_endpoints_and_ports(designed, substrate, frequency):
    assert (designed.w[0], designed.w[-1], designed.s[-1]) == (40e-6, 20e-6, 4e-6)
    z1, z2 = designed.z_ports
    assert z1 == pytest.approx(substrate.z0(frequency, F, 40e-6, 25e-6))
    assert z2 == pytest.approx(substrate.z0(frequency, F, 20e-6, 4e-6))
    assert designed.z_sections[-1] == z2


def test_section_impedances_are_achieved(designed, substrate, frequency):
    achieved = [substrate.z0(frequency, F, w, s) for w, s in zip(designed.w, designed.s)]
    assert np.abs(achieved - designed.z_sections).max() < 1e-3


def test_target_is_met_on_the_real_network(designed, frequency):
    assert designed.report['spec_met'] and designed.report['limited_by'] == 'reflection'
    assert peak_db(designed, frequency) <= -30.0
    assert designed.report['peak_s11_db'] == pytest.approx(peak_db(designed, frequency))


def test_result_is_close_to_shortest(designed, substrate, frequency):
    shorter = transformer_for_length(*GEOMETRY, designed.length * 0.97, F, substrate, frequency)
    assert shorter.report['peak_s11_db'] > -30.0


def test_length_is_near_the_analytic_estimate(designed):
    assert designed.length == pytest.approx(designed.report['estimated_length'], rel=0.1)


def test_optimised_ripple_beats_formula_at_same_length_and_sections(substrate, frequency):
    old = klopfenstein_cpw(40e-6, 25e-6, 36.0, F, -30.0, 20, substrate, frequency)
    s2 = old.report['s2']
    new = transformer_for_length(40e-6, 25e-6, 40e-6, s2, old.length, F, substrate, frequency, max_sections=20)
    assert new.report['n_sections'] == 20
    assert new.length == pytest.approx(old.length)
    assert new.report['peak_s11_db'] < peak_db(old, frequency) - 1.0


def test_longer_is_better(substrate, frequency):
    peaks = [transformer_for_length(*GEOMETRY, length, F, substrate, frequency).report['peak_s11_db']
             for length in (2e-3, 4e-3, 8e-3)]
    assert peaks[0] > peaks[1] > peaks[2]


def test_max_length_limit_warns_and_reports(substrate, frequency):
    with pytest.warns(UserWarning, match='not met'):
        line = transformer_for_reflection(*GEOMETRY, -30.0, F, substrate, frequency, max_length=2e-3)
    assert line.length == pytest.approx(2e-3)
    assert line.report['spec_met'] is False and line.report['limited_by'] == 'length limits'


def test_section_limits_cap_the_length(substrate, frequency):
    with pytest.warns(UserWarning, match='not met'):
        line = transformer_for_reflection(*GEOMETRY, -30.0, F, substrate, frequency, max_sections=10,
                                          max_section_length=200e-6)
    assert len(line.w) == 10
    assert line.length == pytest.approx(2e-3)


def test_max_sections_is_respected_and_target_still_met(substrate, frequency):
    line = transformer_for_reflection(*GEOMETRY, -30.0, F, substrate, frequency, max_sections=12)
    assert len(line.w) == 12
    assert peak_db(line, frequency) <= -30.0


def test_max_section_length_is_respected(substrate, frequency):
    line = transformer_for_length(*GEOMETRY, 3e-3, F, substrate, frequency, max_section_length=50e-6)
    assert line.lengths[0] <= 50e-6
    assert len(line.w) == 60


def test_impossible_section_limits_raise_for_fixed_length(substrate, frequency):
    with pytest.raises(ValueError, match='max_sections'):
        transformer_for_length(*GEOMETRY, 3e-3, F, substrate, frequency, max_sections=10, max_section_length=100e-6)


def test_matched_endpoints_use_the_edge_angle_limit(substrate, frequency):
    s2 = substrate.gap_for_z0(frequency, F, 10e-6, substrate.z0(frequency, F, 40e-6, 25e-6))
    line = transformer_for_reflection(40e-6, 25e-6, 10e-6, s2, -30.0, F, substrate, frequency, max_edge_angle_deg=5)
    outer_travel = abs(10e-6 / 2 + s2 - 40e-6 / 2 - 25e-6)
    assert line.report['limited_by'] == 'edge angle'
    assert line.length == pytest.approx(outer_travel / np.tan(np.radians(5)))
    assert np.ptp(line.z_sections) < 0.01


def test_band_limit_is_used(substrate, frequency):
    wide = transformer_for_reflection(*GEOMETRY, -30.0, F, substrate, frequency)
    narrow = transformer_for_reflection(*GEOMETRY, -30.0, 20e9, substrate, frequency, f_max=25e9)
    assert narrow.length < wide.length
    assert peak_db(narrow, frequency, 20e9, 25e9) <= -30.0


def test_empty_band_raises(substrate, frequency):
    with pytest.raises(ValueError, match='No frequency points'):
        transformer_for_length(*GEOMETRY, 3e-3, 40e9, substrate, frequency)


def test_custom_width_profile(substrate, frequency):
    line = transformer_for_length(*GEOMETRY, 3e-3, F, substrate, frequency, max_section_length=100e-6,
                                  w_profile=lambda t: t**2)
    t = np.linspace(0, 1, len(line.w))
    assert np.allclose(line.w, 40e-6 - 20e-6 * t**2)


@needs_simulator
def test_simulation_matches_skrf(tmp_path, designed, frequency):
    path = tmp_path / 'transformer.sch'
    write_schematic(path, designed, frequency)
    result = read_sp(run_simulation(path), list(designed.z_ports))
    assert np.abs(result.s - designed.network(frequency).s).max() < 1e-8
