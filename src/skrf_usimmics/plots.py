"""Plots shared between notebooks."""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
GUIDE_COLOR = '#52514e'


def plot_profile(line):
    """Impedance and gap along the line, plus width if it varies."""
    panels = [(line.z_sections, '$Z_0$ (ohm)', 'Characteristic impedance along the taper')]
    if np.ptp(line.w) > 0:
        panels.append((line.w * 1e6, 'W (um)', 'CPW width along the taper'))
    panels.append((line.s * 1e6, 'S (um)', 'CPW gap along the taper'))

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.5 * len(panels)), sharex=True, layout='constrained')
    for ax, (values, ylabel, title) in zip(axes, panels):
        ax.plot(line.centers * 1e3, values, color=COLORS[0], linewidth=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    # Keep a minimum span so that solver noise on a constant impedance doesn't look like ripple
    if np.ptp(line.z_sections) < 2:
        axes[0].set_ylim(line.z_sections.mean() - 1, line.z_sections.mean() + 1)
    axes[-1].set_xlabel('Position along taper (mm)')
    return fig, axes


def plot_outline(line):
    """Top view of the conductor edges. The axes have different scales, so angles look exaggerated."""
    fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
    x = np.repeat(line.edges, 2)[1:-1] * 1e3
    inner = np.repeat(line.w / 2, 2) * 1e6
    outer = np.repeat(line.w / 2 + line.s, 2) * 1e6
    ax.fill_between(x, -inner, inner, color=COLORS[0], label='Center conductor')
    ax.fill_between(x, outer, outer.max(), color=GUIDE_COLOR, alpha=0.5, label='Ground')
    ax.fill_between(x, -outer.max(), -outer, color=GUIDE_COLOR, alpha=0.5)
    ax.set_xlabel('Position along taper (mm)')
    ax.set_ylabel('Transverse position (um)')
    ax.set_title('Taper outline (axes not to scale)')
    ax.legend(loc='upper right')
    return fig, ax


def plot_s11(networks, title, ylim=(-60, 0)):
    """|S11| of each network in one plot, with later ones dashed so overlaps stay visible."""
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    for k, network in enumerate(networks):
        network.plot_s_db(m=0, n=0, ax=ax, color=COLORS[k], linewidth=2, linestyle='--' if k else '-',
                          label=network.name)
    ax.set_ylim(*ylim)
    ax.set_ylabel('$|S_{11}|$ (dB)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax
