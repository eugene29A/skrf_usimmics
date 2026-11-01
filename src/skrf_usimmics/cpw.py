"""CPW media model: a substrate, and impedance as a function of geometry."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from skrf.media import CPW


@dataclass(frozen=True)
class Substrate:
    """CPW substrate and metal. Lengths in m. The defaults are lossless zero-thickness PEC, air backside."""
    h: float
    ep_r: float
    tand: float = 0.0
    t: float | None = None
    rho: float | None = None
    metal_backside: bool = False

    def media(self, frequency, w, s):
        return CPW(frequency=frequency, w=w, s=s, h=self.h, ep_r=self.ep_r, t=self.t, rho=self.rho, tand=self.tand,
                   diel='frequencyinvariant', has_metal_backside=self.metal_backside)

    def z0(self, frequency, f_design, w, s):
        """Characteristic impedance at the point of `frequency` nearest to `f_design`."""
        return self.media(frequency, w, s).z0_characteristic[_nearest(frequency, f_design)].real

    def ep_eff(self, frequency, f_design, w, s):
        """Effective permittivity at the point of `frequency` nearest to `f_design`."""
        return self.media(frequency, w, s).ep_reff_f[_nearest(frequency, f_design)].real

    def gap_for_z0(self, frequency, f_design, w, z, bracket=(0.5e-6, 200e-6)):
        """Gap S that gives impedance `z` at width `w`. The bracket widens if it doesn't contain the solution."""
        def error(s):
            return self.z0(frequency, f_design, w, s) - z

        lo, hi = bracket
        while error(lo) > 0 and lo > 1e-9:
            lo /= 10
        # With finite metal thickness, the model has no solution for gaps that are small relative to t
        while np.isnan(error(lo)) and lo < hi:
            lo *= 1.5
        while error(hi) < 0 and hi < 1:
            hi *= 10
        if not error(lo) * error(hi) <= 0:
            raise ValueError(f'No gap gives Z0 = {z:.2f} ohm at W = {w*1e6:.3f} um (tried S = {lo:.3g} to {hi:.3g} m)')
        return brentq(error, lo, hi, xtol=1e-10)


def _nearest(frequency, f):
    return np.argmin(np.abs(frequency.f - f))
