import os
import re

import numpy as np
import pytest
import skrf as rf
from skrf.media import CPW

from skrf_usimmics import Substrate, constant_z_cpw, klopfenstein_cpw
from skrf_usimmics.usimmics import (netlist_text, read_sp, run_simulation, sch_components, sch_props, schematic_text,
                                    simulator_path, write_schematic)

F = 9e9
needs_simulator = pytest.mark.skipif(not os.path.exists(simulator_path()), reason='uSimmics simulator not found')


@pytest.fixture(scope='module')
def line(substrate, frequency):
    return klopfenstein_cpw(40e-6, 25e-6, 36.0, F, -30.0, 20, substrate, frequency)


@pytest.fixture(scope='module')
def varied_line(substrate, frequency):
    return constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 20, substrate, frequency, F)


@pytest.fixture(scope='module')
def sample_sch(data_dir):
    return (data_dir / 'cpw_sample.sch').read_text(encoding='utf-8')


def components(text):
    return text.split('<Components>\n')[1].split('</Components>')[0].splitlines()


def templates(text):
    """Component lines with values, coordinates and index numbers masked."""
    out = {}
    for row in components(text):
        row = re.sub(r'^(CLIN CL|Pac P)\d+', r'\1#', row)
        row = re.sub(r'^(\S+ \S+ \d) -?\d+ -?\d+', r'\1 X Y', row)
        out.setdefault(row.split()[0], set()).add(re.sub(r'"[^"]*"', '"V"', row))
    return out


def test_netlist_reproduces_gui_netlist(sample_sch, data_dir):
    gui = (data_dir / 'cpw_sample_netlist.txt').read_text(encoding='utf-8')
    assert netlist_text(sample_sch).splitlines()[1:] == gui.splitlines()[1:]


def test_schematic_matches_sample_format(line, frequency, sample_sch):
    new = schematic_text(line, frequency)
    assert templates(new) == templates(sample_sch)

    def structure(text):
        return [row for row in text.splitlines() if (row.startswith('<') or '=' in row) and not row.startswith('View=')]

    assert structure(new) == structure(sample_sch)
    raw = new.encode('utf-8')
    assert b'\r' not in raw and not raw.startswith(b'\xef\xbb\xbf')


def test_schematic_wiring(line, frequency):
    text = schematic_text(line, frequency)
    rows = components(text)
    wires = [tuple(map(int, row.split()[:4])) for row in text.split('<Wires>\n')[1].split('</Wires>')[0].splitlines()]
    ends = {w[:2] for w in wires} | {w[2:] for w in wires}
    position = lambda row: tuple(map(int, row.split()[3:5]))

    x_sections = [position(row)[0] for row in rows if row.startswith('CLIN')]
    assert len(x_sections) == len(line.w)
    assert all((x + 30, 150, x_next - 30, 150) in wires for x, x_next in zip(x_sections, x_sections[1:]))
    assert all({(x - 30, 150), (x + 30, 150)} <= ends for x in x_sections)

    ports = [position(row) for row in rows if row.startswith('Pac')]
    assert all((x, y - 30) in ends for x, y in ports)
    assert {position(row) for row in rows if row.startswith('GND')} == {(x, y + 30) for x, y in ports}
    assert all((w[0] == w[2] or w[1] == w[3]) and not any(v % 10 for v in w) for w in wires)


def test_schematic_writes_per_section_geometry(varied_line, frequency):
    parts = sch_components(schematic_text(varied_line, frequency))
    props = [sch_props(parts[f'CL{k + 1}']) for k in range(len(varied_line.w))]
    assert np.allclose([float(p[1].split()[0]) for p in props], varied_line.w * 1e6, rtol=1e-8)
    assert np.allclose([float(p[2].split()[0]) for p in props], varied_line.s * 1e6, rtol=1e-8)
    assert np.allclose([float(p[3].split()[0]) for p in props], varied_line.lengths * 1e6, rtol=1e-8)


def test_substrate_properties_are_written(frequency):
    substrate = Substrate(h=250e-6, ep_r=9.8, tand=1e-3, t=2e-6, rho=2.4e-8, metal_backside=True)
    line = constant_z_cpw(40e-6, 25e-6, 40e-6, 25e-6, 1e-3, 2, substrate, frequency, F)
    assert sch_props(sch_components(schematic_text(line, frequency))['Subst1'])[:7] == [
        '9.8', '250 um', '2 um', '0.001', '2.4e-08', '0', 'Metal']


def test_create_patch_regenerate(tmp_path, line, varied_line, frequency, capsys):
    path = tmp_path / 'deep' / 'taper.sch'
    assert write_schematic(path, line, frequency) == 'created'
    assert path.read_text(encoding='utf-8') == schematic_text(line, frequency)

    # Manual edits in the GUI: a diagram, a moved block, and a changed port
    edited = path.read_text(encoding='utf-8').replace('<Diagrams>\n', '<Diagrams>\n<Rect 1 2 3 4>\n</Rect>\n')
    edited = edited.replace('.SP SP1 1 90 490', '.SP SP1 1 100 500').replace('"36 Ω"', '"50 Ω"')
    path.write_text(edited, encoding='utf-8', newline='\n')

    assert write_schematic(path, varied_line, frequency) == 'patched'
    assert 'differs from the design: P2' in capsys.readouterr().out
    after = path.read_text(encoding='utf-8')
    changed = [(old, new) for old, new in zip(edited.splitlines(), after.splitlines()) if old != new]
    assert len(edited.splitlines()) == len(after.splitlines())
    assert len(changed) == len(line.w)
    for old, new in changed:
        assert old.startswith('CLIN') and old.split('"')[0] == new.split('"')[0]
        assert sch_props(old)[0] == sch_props(new)[0] and sch_props(old)[4:] == sch_props(new)[4:]
    assert '<Rect 1 2 3 4>' in after and '.SP SP1 1 100 500' in after and '"50 Ω"' in after
    new_parts = sch_components(schematic_text(varied_line, frequency))
    assert all(sch_props(sch_components(after)[k])[1:4] == sch_props(new_parts[k])[1:4] for k in new_parts if k.startswith('CL'))
    assert not path.with_name('taper.sch.bak').exists()

    shorter = constant_z_cpw(40e-6, 25e-6, 10e-6, 10e-6, 500e-6, 10, line.substrate, frequency, F)
    assert write_schematic(path, shorter, frequency) == 'regenerated'
    assert path.read_text(encoding='utf-8') == schematic_text(shorter, frequency)
    assert path.with_name('taper.sch.bak').read_text(encoding='utf-8') == after


def test_read_sp_matches_skrf_model(data_dir):
    # The sample has a metal backside and tand = 2e-4, and matches skrf's qucs compatibility mode
    network = read_sp(data_dir / 'cpw_sample.dat', 50)
    assert network.frequency.npoints == 100
    assert network.frequency.f[-1] == pytest.approx(30e9)
    assert np.abs(network.s[:, 0, 1] - network.s[:, 1, 0]).max() < 1e-12

    def media(s):
        return CPW(frequency=network.frequency, w=40e-6, s=s, h=500e-6, ep_r=11.7, t=None, rho=None, tand=2e-4,
                   diel='frequencyinvariant', has_metal_backside=True, compatibility_mode='qucs')

    model = media(25e-6).line(1e-3, 'm') ** media(10e-6).line(2e-3, 'm')
    model.renormalize(50)
    assert np.abs(model.s - network.s).max() < 1e-5


@needs_simulator
@pytest.mark.parametrize('fixture_name', ['line', 'varied_line'])
def test_simulation_matches_skrf(tmp_path, frequency, request, fixture_name):
    design = request.getfixturevalue(fixture_name)
    path = tmp_path / 'taper.sch'
    write_schematic(path, design, frequency)
    result = read_sp(run_simulation(path), list(design.z_ports))
    assert np.abs(result.s - design.network(frequency).s).max() < 1e-8


@needs_simulator
def test_backside_is_honoured(tmp_path, frequency, line):
    path = tmp_path / 'taper.sch'
    path.write_text(schematic_text(line, frequency).replace('"Air"', '"Metal"'), encoding='utf-8', newline='\n')
    result = read_sp(run_simulation(path), list(line.z_ports))
    assert np.abs(result.s - line.network(frequency).s).max() > 1e-4


@needs_simulator
def test_lossy_substrate_simulates(tmp_path, frequency):
    substrate = Substrate(h=500e-6, ep_r=11.7, tand=1e-3, t=1e-6, rho=2.4e-8)
    line = constant_z_cpw(40e-6, 25e-6, 40e-6, 25e-6, 5e-3, 2, substrate, frequency, F)
    path = tmp_path / 'lossy.sch'
    write_schematic(path, line, frequency)
    result = read_sp(run_simulation(path), list(line.z_ports))
    assert np.abs(result.s[-1, 1, 0]) < 0.99


@needs_simulator
def test_failed_simulation_raises_and_leaves_no_dataset(tmp_path, frequency, line):
    path = tmp_path / 'bad.sch'
    path.write_text(schematic_text(line, frequency).replace('"40 um"', '"abc"', 1), encoding='utf-8', newline='\n')
    path.with_suffix('.dat').write_bytes(b'stale')
    with pytest.raises(RuntimeError, match='simulation failed'):
        run_simulation(path)
    assert not path.with_suffix('.dat').exists()
