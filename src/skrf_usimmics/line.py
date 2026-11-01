"""SectionedLine: the description every design returns and every output consumes."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from skrf.network import cascade_list

from skrf_usimmics.cpw import Substrate


@dataclass(frozen=True)
class SectionedLine:
    """A chain of uniform CPW sections. Lengths in m, impedances in ohm.

    `z_sections` is the impedance of each section, `z_ports` the feed line impedance at each end,
    and `report` holds whatever figures the design that made this line wants to expose.
    """
    w: np.ndarray
    s: np.ndarray
    lengths: np.ndarray
    z_sections: np.ndarray
    z_ports: tuple[float, float]
    substrate: Substrate
    name: str = 'CPW line'
    report: dict = field(default_factory=dict)

    @property
    def length(self):
        return self.lengths.sum()

    @property
    def edges(self):
        """Section boundaries along the line."""
        return np.concatenate([[0.0], np.cumsum(self.lengths)])

    @property
    def centers(self):
        """Section midpoints along the line."""
        return (self.edges[:-1] + self.edges[1:]) / 2

    def network(self, frequency, z_ports=None):
        """Cascade the sections. skrf inserts the impedance steps between them, and at the ports."""
        sections = [self.substrate.media(frequency, w, s).line(length, 'm')
                    for w, s, length in zip(self.w, self.s, self.lengths)]
        network = cascade_list(sections)
        network.renormalize(list(self.z_ports if z_ports is None else z_ports))
        network.name = self.name
        return network

    def table(self):
        return pd.DataFrame({
            'Section': np.arange(1, len(self.w) + 1),
            'Start (mm)': self.edges[:-1] * 1e3,
            'End (mm)': self.edges[1:] * 1e3,
            'W (um)': self.w * 1e6,
            'S (um)': self.s * 1e6,
            'Z0 (ohm)': self.z_sections,
        }).set_index('Section')
