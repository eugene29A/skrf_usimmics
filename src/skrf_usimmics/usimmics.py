"""Write SectionedLines as uSimmics schematics, simulate them, and read the results.

uSimmics documents none of these formats. The schematic and netlist layouts are copied from files saved by
uSimmics 5.9, and the simulator call and dataset layout were worked out from its files, so a later version
could change them. tests/data holds the reference files.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import skrf as rf

DEFAULT_SIMULATOR = r'C:\Work\uSimmics\bin\simulator.exe'
SCH_PITCH = 230     # horizontal spacing of the line sections


def sch_components(text):
    block = text.split('<Components>\n')[1].split('</Components>')[0]
    return {line.split()[1]: line for line in block.splitlines() if not line.startswith('GND')}


def sch_props(line):
    return re.findall(r'"([^"]*)"', line)


def set_sch_props(line, new_values):
    parts = re.split(r'"([^"]*)"', line)
    for i, value in new_values.items():
        parts[2 * i + 1] = value
    return ''.join(f'"{p}"' if k % 2 else p for k, p in enumerate(parts))


def schematic_text(line, frequency):
    """Line sections in one horizontal row, between two ports, with an S-parameter simulation."""
    sub = line.substrate
    z_in, z_out = line.z_ports
    x_cl = [450 + SCH_PITCH * k for k in range(len(line.w))]
    x_p2 = x_cl[-1] + 250
    components = [
        f'CLIN CL{k + 1} 1 {x} 150 -26 28 0 "Subst1"1"{w*1e6:.9g} um"1"{s*1e6:.9g} um"1"{length*1e6:.9g} um"1"26.85"0'
        for k, (x, w, s, length) in enumerate(zip(x_cl, line.w, line.s, line.lengths))]
    # SUBST props: er, h, t, tand, rho, D, backside, microstrip models. t = rho = 0 is lossless PEC.
    thickness = f'{sub.t*1e6:.9g} um' if sub.t else '0'
    resistivity = f'{sub.rho:.9g}' if sub.rho else '0'
    backside = 'Metal' if sub.metal_backside else 'Air'
    components += [
        f'SUBST Subst1 1 520 490 -28 20 0 "{sub.ep_r:.9g}"1"{sub.h*1e6:.9g} um"1"{thickness}"1"{sub.tand:.9g}"1'
        f'"{resistivity}"1"0"0"{backside}"0"Hammerstad"0"Kirschning"0',
        f'Pac P1 1 220 230 18 -26 0 "1"1"{z_in:.9g} Ω"1"0 dBm"0"1 GHz"0"26.85"0"con_2"0',
        f'Pac P2 1 {x_p2} 230 18 -26 0 "2"1"{z_out:.9g} Ω"1"0 dBm"0"1 GHz"0"26.85"0"con_2"0',
        'GND * 1 220 260 0 0 0',
        f'GND * 1 {x_p2} 260 0 0 0',
        f'.SP SP1 1 90 490 0 9 0 "lin"1"{frequency.start/1e9:.9g} GHz"1"{frequency.stop/1e9:.9g} GHz"1'
        f'"{frequency.npoints}"1"no"0"1"0"2"0"none"0',
    ]
    wires = ['220 150 220 200', f'220 150 {x_cl[0] - 30} 150']
    wires += [f'{x + 30} 150 {x + SCH_PITCH - 30} 150' for x in x_cl[:-1]]
    wires += [f'{x_cl[-1] + 30} 150 {x_p2} 150', f'{x_p2} 150 {x_p2} 200']
    return '\n'.join([
        '<QucsStudio Schematic 5.9>', '<Properties>', f'View=0,0,{x_p2 + 110},800,1,0,0', 'Grid=10,10,1',
        'DataSet=*.dat', 'DataDisplay=*.dpl', 'OpenDisplay=1', 'showFrame=0', r'FrameText0=Title \n @PATH@@FILE@',
        'FrameText1=Drawn By:', 'FrameText2=Date: @DATE@', 'FrameText3=Revision:', '</Properties>',
        '<Symbol>', '</Symbol>', '<Components>', *components, '</Components>', '<Wires>', *wires, '</Wires>',
        '<Diagrams>', '</Diagrams>', '<Paintings>', '</Paintings>', '',
    ])


def write_schematic(path, line, frequency):
    """Create the schematic, or update an existing one. Returns 'created', 'patched' or 'regenerated'.

    With the same sections, only W, S and L change, so manual edits survive, and anything else that differs
    from the design is printed. With different sections, the file is rebuilt and the old one kept as .bak.
    """
    path = Path(path)
    new = schematic_text(line, frequency)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(new, encoding='utf-8', newline='\n')
        return 'created'

    old = path.read_text(encoding='utf-8')
    old_parts, new_parts = sch_components(old), sch_components(new)
    sections = {name for name in new_parts if name.startswith('CL')}
    if {name for name, text in old_parts.items() if text.startswith('CLIN')} != sections:
        shutil.copy2(path, path.with_name(path.name + '.bak'))
        path.write_text(new, encoding='utf-8', newline='\n')
        return 'regenerated'

    for name in sections:
        w_new, s_new, l_new = sch_props(new_parts[name])[1:4]
        old = old.replace(old_parts[name], set_sch_props(old_parts[name], {1: w_new, 2: s_new, 3: l_new}))
    path.write_text(old, encoding='utf-8', newline='\n')
    patched_parts = sch_components(old)
    for name, text in new_parts.items():
        if name not in patched_parts or sch_props(patched_parts[name]) != sch_props(text):
            print(f'Not updated, differs from the design: {name}')
    return 'patched'


def netlist_text(sch_text):
    """Netlist of a schematic from schematic_text. Sections are chained in name order, so GUI rewiring is ignored."""
    parts = sch_components(sch_text)
    n_lines = sum(text.startswith('CLIN') for text in parts.values())
    lines = ['# uSimmics 5.9', '']
    for name, text in parts.items():
        kind = text.split()[0]
        if kind == 'CLIN':
            nets = f' _net{int(name[2:]) - 1} _net{name[2:]}'
        elif kind == 'Pac':
            nets = ' _net0 gnd' if name == 'P1' else f' _net{n_lines} gnd'
        else:
            nets = ''
        lines.append(f'{kind}:{name}{nets} ' + ' '.join(f'"{v}"' for v in sch_props(text)))
    return '\n'.join(lines) + '\n'


def simulator_path():
    return os.environ.get('USIMMICS_SIM', DEFAULT_SIMULATOR)


def run_simulation(sch_path):
    """Simulate a schematic without the GUI, overwriting its .dat file. Returns the dataset path."""
    sch_path = Path(sch_path)
    netlist = sch_path.with_name(sch_path.stem + '_netlist.txt')
    dataset = sch_path.with_suffix('.dat')
    netlist.write_text(netlist_text(sch_path.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    dataset.unlink(missing_ok=True)
    result = subprocess.run([simulator_path(), str(netlist), str(dataset)], capture_output=True, text=True,
                            stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(f'uSimmics simulation failed: {result.stderr.strip()}')
    return dataset


def read_sp(dataset, z0):
    """Read a 2-port S-parameter dataset.

    Layout: 12-byte tag, u32 header size, header, then S[n_f] as complex128 for each port pair,
    then the frequencies as float64.
    """
    raw = Path(dataset).read_bytes()
    offset = 16 + int.from_bytes(raw[12:16], 'little')
    n_f = (len(raw) - offset) // (4 * 16 + 8)
    s = np.frombuffer(raw, '<c16', 4 * n_f, offset).reshape(2, 2, n_f).transpose(2, 0, 1)
    f = np.frombuffer(raw, '<f8', n_f, offset + 4 * 16 * n_f)
    return rf.Network(frequency=rf.Frequency.from_f(f / 1e9, unit='GHz'), s=s, z0=z0, name='uSimmics')
