"""CPW taper designs. Each function returns a SectionedLine."""
import warnings
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Callable

import numpy as np
from scipy.constants import c
from scipy.optimize import minimize_scalar
from skrf.mathFunctions import db_2_mag, mag_2_db
from skrf.media import CPW
from skrf.taper import Klopfenstein

from skrf_usimmics.cpw import Substrate
from skrf_usimmics.line import SectionedLine

MIN_SECTIONS = 5
SECTIONS_PER_WAVELENGTH = 20    # default section length is this fraction of the wavelength at f_max


def klopfenstein_cpw(w, s1, z2, f_c, gamma_m_db, n_sections, substrate, frequency):
    """Impedance transformer at constant W, from gap `s1` to the gap that gives `z2`.

    Gamma_0 = ln(Z2/Z1)/2, Gamma_m = Gamma_0/cosh(A), and the passband is beta*L >= A.
    skrf's rmax is the ratio Gamma_m/Gamma_0. The length is set by `f_c`, the lower passband edge.
    """
    z1 = substrate.z0(frequency, f_c, w, s1)
    gamma_0 = np.abs(np.log(z2 / z1)) / 2
    rmax = db_2_mag(gamma_m_db) / gamma_0
    a = np.arccosh(1 / rmax)
    ep_eff = substrate.ep_eff(frequency, f_c, w, s1)
    length = a * c / (2 * np.pi * f_c * np.sqrt(ep_eff))

    z_profile = Klopfenstein(med=CPW, param='z0', start=z1, stop=z2, length=length, n_sections=n_sections,
                             f_kw={'rmax': rmax}).value_vector
    s_profile = np.array([substrate.gap_for_z0(frequency, f_c, w, z) for z in z_profile])

    return SectionedLine(
        w=np.full(n_sections, float(w)), s=s_profile, lengths=np.full(n_sections, length / n_sections),
        z_sections=z_profile, z_ports=(z1, z2), substrate=substrate, name='Klopfenstein CPW taper',
        report={'gamma_0': gamma_0, 'rmax': rmax, 'A': a, 'ep_eff': ep_eff, 'z1': z1, 'z2': z2,
                's2': substrate.gap_for_z0(frequency, f_c, w, z2), 'f_c': f_c},
    )


def klopfenstein_response(line, frequency):
    """Analytic |Gamma| of a line from klopfenstein_cpw."""
    report = line.report
    beta = 2 * np.pi * frequency.f * np.sqrt(report['ep_eff']) / c
    return report['gamma_0'] * np.abs(np.cos(np.sqrt((beta * line.length)**2 - report['A']**2 + 0j))) / np.cosh(report['A'])


def constant_z_cpw(w1, s1, w2, s2, length, n_sections, substrate, frequency, f_design, w_profile=lambda t: t):
    """Transition from (w1, s1) to (w2, s2), solving the gap to hold the impedance.

    W follows `w_profile`, which maps position 0..1 to progress 0..1. The target impedance blends
    log-linearly from Z(w1, s1) to Z(w2, s2), so it is constant when those two match. Both end sections
    are exactly the requested geometry.

    The length is an input because the circuit model cannot choose it. The cascade only sees impedance
    steps, not step parasitics, mode conversion or radiation, so a constant-impedance line shows almost
    no reflection at any length. Use the edge angles in `report` as a guide, and a full-wave simulation
    to validate the length.
    """
    if n_sections < 2:
        raise ValueError('n_sections must be at least 2')

    z1 = substrate.z0(frequency, f_design, w1, s1)
    z2 = substrate.z0(frequency, f_design, w2, s2)
    t = np.linspace(0, 1, n_sections)
    w = w1 + (w2 - w1) * np.array([w_profile(ti) for ti in t])
    w[[0, -1]] = w1, w2
    z_target = z1 * (z2 / z1)**t
    s = np.array([s1, *(substrate.gap_for_z0(frequency, f_design, wi, zi) for wi, zi in zip(w[1:-1], z_target[1:-1])), s2])
    z_sections = np.array([substrate.z0(frequency, f_design, wi, si) for wi, si in zip(w, s)])

    section_length = length / n_sections
    inner_edge = w / 2
    outer_edge = w / 2 + s
    return SectionedLine(
        w=w, s=s, lengths=np.full(n_sections, section_length), z_sections=z_sections, z_ports=(z1, z2),
        substrate=substrate, name='Constant-impedance CPW taper',
        report={
            'z1': z1,
            'z2': z2,
            'endpoint_mismatch_ohm': z2 - z1,
            's2_for_constant_z': substrate.gap_for_z0(frequency, f_design, w2, z1),
            'z_error_ohm': np.abs(z_sections - z_target).max(),
            'min_gap': s.min(),
            'inner_edge_angle_deg': _steepest_angle(inner_edge, section_length),
            'outer_edge_angle_deg': _steepest_angle(outer_edge, section_length),
        },
    )


def transformer_for_reflection(w1, s1, w2, s2, gamma_m_db, f_c, substrate, frequency, f_max=None, max_length=None,
                               max_sections=200, max_section_length=None, max_edge_angle_deg=30, w_profile=lambda t: t):
    """Shortest Klopfenstein transition from (w1, s1) to (w2, s2) with |S11| <= gamma_m_db from f_c to f_max.

    The result is checked on the cascaded network, not the continuous-taper formula, and the length is found to
    about 1%. The length is limited by `max_length` and by `max_sections` * `max_section_length`. If the target
    can't be met, this warns and returns the best line at the limit, with report['spec_met'] False.

    `max_edge_angle_deg` sets a minimum length, from the straight end-to-end slope of the conductor edges. The
    circuit model cannot do this itself, so nearly matched endpoints would otherwise give a near-zero length.
    The 30 degree default is a starting assumption, not a result: validate it with a full-wave simulation.
    See `transformer_for_length` for how the line is built.
    """
    problem = _Transition(w1, s1, w2, s2, f_c, f_max or frequency.stop, substrate, frequency, w_profile, max_sections,
                          max_section_length)
    gamma_m = db_2_mag(gamma_m_db)
    a_needed = np.arccosh(problem.gamma_0 / gamma_m) if problem.gamma_0 > gamma_m else 0.0
    estimate = a_needed / problem.beta_c

    cap = min(max_length or np.inf, max_sections * max_section_length if max_section_length else np.inf)
    floor = min(problem.min_length(max_edge_angle_deg), cap)
    if np.isinf(cap):
        cap = max(10 * estimate, 2 * floor)

    best = problem.best_at(floor)
    limited_by = 'edge angle'
    if _peak_db(best) > gamma_m_db:
        low, high = floor, min(max(estimate, floor * 1.1), cap)
        best = problem.best_at(high)
        while _peak_db(best) > gamma_m_db and high < cap:
            low, high = high, min(high * 1.1, cap)
            best = problem.best_at(high)
        limited_by = 'reflection' if _peak_db(best) <= gamma_m_db else 'length limits'
        while limited_by == 'reflection' and high / low > 1.01:
            middle = (low + high) / 2
            candidate = problem.best_at(middle)
            if _peak_db(candidate) <= gamma_m_db:
                high, best = middle, candidate
            else:
                low = middle

    spec_met = bool(_peak_db(best) <= gamma_m_db)
    if not spec_met:
        warnings.warn(f'Target {gamma_m_db:.1f} dB not met: best is {_peak_db(best):.2f} dB at '
                      f'L = {best.length*1e3:.3f} mm. Relax the length limits or the target.', stacklevel=2)
    return replace(best, report={**best.report, 'target_db': gamma_m_db, 'spec_met': spec_met,
                                 'estimated_length': estimate, 'limited_by': limited_by})


def transformer_for_length(w1, s1, w2, s2, length, f_c, substrate, frequency, f_max=None, max_sections=200,
                           max_section_length=None, w_profile=lambda t: t):
    """Lowest peak |S11| from f_c to f_max for a transition from (w1, s1) to (w2, s2) of a given length.

    The impedance follows a Klopfenstein profile from Z(w1, s1) to Z(w2, s2), W follows `w_profile` (position 0..1
    to progress 0..1), and the gap is solved at each section. The last section is exactly (w2, s2). The first has
    width w1, but a gap set by the Klopfenstein end step, as in klopfenstein_cpw.

    The continuous-taper formula sets the ripple from the length, A = beta*L. A staircase of sections behaves
    slightly differently, so A is optimised on the cascaded network instead.

    The sections are as long as `max_section_length` allows, or wavelength/20 at f_max, and at most `max_sections`.
    The circuit model has no step parasitics, mode conversion or radiation, so check short designs and the
    end steps with a full-wave simulation.
    """
    if max_section_length and length > max_sections * max_section_length:
        raise ValueError(f'L = {length*1e3:.3f} mm needs more than max_sections = {max_sections} sections '
                         f'of at most {max_section_length*1e6:.3f} um')
    problem = _Transition(w1, s1, w2, s2, f_c, f_max or frequency.stop, substrate, frequency, w_profile, max_sections,
                          max_section_length)
    return problem.best_at(length)


@dataclass(frozen=True)
class _Transition:
    """The fixed inputs of a Klopfenstein transition, for designs that vary its length and ripple."""
    w1: float
    s1: float
    w2: float
    s2: float
    f_c: float
    f_max: float
    substrate: Substrate
    frequency: object
    w_profile: Callable
    max_sections: int
    max_section_length: float | None

    @cached_property
    def z1(self):
        return self.substrate.z0(self.frequency, self.f_c, self.w1, self.s1)

    @cached_property
    def z2(self):
        return self.substrate.z0(self.frequency, self.f_c, self.w2, self.s2)

    @cached_property
    def gamma_0(self):
        return np.abs(np.log(self.z2 / self.z1)) / 2

    @cached_property
    def ep_eff(self):
        return (self.substrate.ep_eff(self.frequency, self.f_c, self.w1, self.s1)
                + self.substrate.ep_eff(self.frequency, self.f_c, self.w2, self.s2)) / 2

    @cached_property
    def beta_c(self):
        return 2 * np.pi * self.f_c * np.sqrt(self.ep_eff) / c

    @cached_property
    def passband(self):
        passband = (self.frequency.f >= self.f_c) & (self.frequency.f <= self.f_max)
        if not passband.any():
            raise ValueError(f'No frequency points between f_c = {self.f_c:g} Hz and f_max = {self.f_max:g} Hz')
        return passband

    def min_length(self, max_edge_angle_deg):
        inner = np.abs(self.w2 - self.w1) / 2
        outer = np.abs(self.w2 / 2 + self.s2 - self.w1 / 2 - self.s1)
        return max(max(inner, outer) / np.tan(np.radians(max_edge_angle_deg)), 1e-6)

    def n_sections(self, length):
        wavelength = c / (self.f_max * np.sqrt(self.ep_eff))
        limit = self.max_section_length or wavelength / SECTIONS_PER_WAVELENGTH
        return min(max(int(np.ceil(length / limit)), MIN_SECTIONS), self.max_sections)

    def build(self, length, a):
        n = self.n_sections(length)
        z = Klopfenstein(med=CPW, param='z0', start=self.z1, stop=self.z2, length=length, n_sections=n,
                         f_kw={'rmax': 1 / np.cosh(a)}).value_vector
        w = self.w1 + (self.w2 - self.w1) * np.array([self.w_profile(t) for t in np.linspace(0, 1, n)])
        w[[0, -1]] = self.w1, self.w2
        s = np.array([*(self.substrate.gap_for_z0(self.frequency, self.f_c, wi, zi) for wi, zi in zip(w[:-1], z[:-1])),
                      self.s2])
        z[-1] = self.z2
        line = SectionedLine(w=w, s=s, lengths=np.full(n, length / n), z_sections=z, z_ports=(self.z1, self.z2),
                             substrate=self.substrate, name='Klopfenstein CPW transition')
        peak = mag_2_db(np.abs(line.network(self.frequency).s[self.passband, 0, 0])).max()
        return replace(line, report={
            'z1': self.z1,
            'z2': self.z2,
            'gamma_0': self.gamma_0,
            'A': a,
            'f_c': self.f_c,
            'f_max': self.f_max,
            'ep_eff': self.ep_eff,
            'peak_s11_db': peak,
            'analytic_peak_db': mag_2_db(max(self.gamma_0 / np.cosh(self.beta_c * length), 1e-15)),
            'n_sections': n,
            'section_length': length / n,
            'min_gap': s.min(),
            'inner_edge_angle_deg': _steepest_angle(w / 2, length / n),
            'outer_edge_angle_deg': _steepest_angle(w / 2 + s, length / n),
        })

    def best_at(self, length):
        a_formula = max(self.beta_c * length, 1e-3)
        best = self.build(length, a_formula)
        if self.gamma_0 < 1e-6:     # matched endpoints: the ripple has nothing to act on
            return best
        optimum = minimize_scalar(lambda a: _peak_db(self.build(length, a)), bounds=(0.5 * a_formula, 1.5 * a_formula),
                                  method='bounded', options={'xatol': 0.01})
        candidate = self.build(length, optimum.x)
        return candidate if _peak_db(candidate) < _peak_db(best) else best


def _peak_db(line):
    return line.report['peak_s11_db']


def _steepest_angle(edge, section_length):
    return np.degrees(np.arctan(np.abs(np.diff(edge)).max() / section_length))
