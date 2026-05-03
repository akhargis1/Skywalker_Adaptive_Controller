"""
plot_smc.py  –  Post-flight visualiser for SMCLogger .npz files
Usage:
    python plot_smc.py sitl_smc_<timestamp>.npz
    python plot_smc.py sitl_smc_<timestamp>.npz --save          # write PNG instead of showing
    python plot_smc.py sitl_smc_<timestamp>.npz --dark          # dark theme
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Column index helpers
# ---------------------------------------------------------------------------
_FIELDS = [
    't',
    'x', 'y', 'z',
    'vx', 'vy', 'vz',
    'phi', 'theta', 'psi',
    'p', 'q', 'r',
    'airspeed',
    'e_n', 'e_t', 'e_z',
    'chi_err',
    'kappa', 'psi_r',
    's1', 's2', 's3',
    'phi_cmd', 'theta_cmd', 'T_cmd',
    'x_r', 'y_r',
    'gamma',
]
_IDX = {f: i for i, f in enumerate(_FIELDS)}


def col(arr, name):
    return arr[:, _IDX[name]]


def rad2deg(x):
    return np.degrees(x)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_npz(path: str) -> np.ndarray:
    npz = np.load(path, allow_pickle=False)
    data = npz['data']
    # Validate field count
    if 'fields' in npz:
        saved = list(npz['fields'])
        if saved != _FIELDS:
            print(f"[WARN] Field mismatch – expected {len(_FIELDS)}, got {len(saved)}. "
                  "Plotting may be incorrect.")
    if data.ndim != 2 or data.shape[1] != len(_FIELDS):
        raise ValueError(
            f"Expected shape (N, {len(_FIELDS)}), got {data.shape}"
        )
    return data


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

DARK = dict(
    fig_bg='#0e1117', axes_bg='#161b22', text='#e6edf3',
    grid='#30363d', accent='#58a6ff',
    colors=['#58a6ff', '#f78166', '#56d364', '#d2a8ff', '#ffa657'],
)
LIGHT = dict(
    fig_bg='#f6f8fa', axes_bg='#ffffff', text='#1f2328',
    grid='#d0d7de', accent='#0969da',
    colors=['#0969da', '#cf222e', '#1a7f37', '#8250df', '#9a6700'],
)


def apply_theme(theme: dict):
    plt.rcParams.update({
        'figure.facecolor':  theme['fig_bg'],
        'axes.facecolor':    theme['axes_bg'],
        'axes.edgecolor':    theme['grid'],
        'axes.labelcolor':   theme['text'],
        'axes.titlecolor':   theme['text'],
        'xtick.color':       theme['text'],
        'ytick.color':       theme['text'],
        'grid.color':        theme['grid'],
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'legend.facecolor':  theme['axes_bg'],
        'legend.edgecolor':  theme['grid'],
        'legend.labelcolor': theme['text'],
        'text.color':        theme['text'],
        'font.family':       'monospace',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


# ---------------------------------------------------------------------------
# Individual panel builders
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(MaxNLocator(8, integer=False))


def panel_track(ax, d, theme):
    """XY ground track with reference path overlay."""
    x, y   = col(d, 'x'),   col(d, 'y')
    xr, yr = col(d, 'x_r'), col(d, 'y_r')

    ax.plot(xr, yr, color=theme['grid'], lw=1.5, ls=':', label='Reference', zorder=1)

    # Colour-code actual track by cross-track error magnitude
    e_n = col(d, 'e_n')
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(0, max(2.0, np.percentile(np.abs(e_n), 95)))
    lc = LineCollection(segs, cmap='RdYlGn_r', norm=norm, lw=1.8, zorder=2)
    lc.set_array(np.abs(e_n[:-1]))
    ax.add_collection(lc)
    ax.autoscale()

    cb = plt.colorbar(lc, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label('|e_n| (m)', fontsize=8)
    cb.ax.yaxis.set_tick_params(color=theme['text'])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=theme['text'])

    ax.scatter(x[0],  y[0],  marker='o', s=60, color='#56d364', zorder=5, label='Start')
    ax.scatter(x[-1], y[-1], marker='s', s=60, color='#f78166', zorder=5, label='End')
    ax.set_xlabel('x (m NED)')
    ax.set_ylabel('y (m NED)')
    ax.set_title('Ground Track  (colour = cross-track error)')
    ax.set_aspect('equal', 'datalim')
    ax.legend(fontsize=8, loc='upper left')
    _style_ax(ax)


def panel_errors(ax_n, ax_t, ax_z, d, theme):
    t = col(d, 't')
    c = theme['colors']
    for ax, key, label, color in [
        (ax_n, 'e_n', 'e_n  cross-track (m)', c[0]),
        (ax_t, 'e_t', 'e_t  along-track (m)', c[1]),
        (ax_z, 'e_z', 'e_z  altitude (m)',     c[2]),
    ]:
        ax.plot(t, col(d, key), color=color, lw=1.2)
        ax.axhline(0, color=theme['grid'], lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        _style_ax(ax)
    ax_n.set_title('Path Errors')
    ax_z.set_xlabel('Time (s)')


def panel_sliding_surfaces(ax, d, theme):
    t = col(d, 't')
    c = theme['colors']
    for key, label, color in [
        ('s1', 's1  (cross-track)', c[0]),
        ('s2', 's2  (along-track)', c[1]),
        ('s3', 's3  (altitude)',    c[2]),
    ]:
        ax.plot(t, col(d, key), label=label, color=color, lw=1.2)
    ax.axhline(0, color=theme['grid'], lw=0.8)
    ax.set_ylabel('Sliding surface')
    ax.set_xlabel('Time (s)')
    ax.set_title('Sliding Surfaces  s1 / s2 / s3')
    ax.legend(fontsize=8)
    _style_ax(ax)


def panel_attitude(ax_phi, ax_theta, d, theme):
    t = col(d, 't')
    c = theme['colors']

    phi_act = rad2deg(col(d, 'phi'))
    phi_cmd = rad2deg(col(d, 'phi_cmd'))
    ax_phi.plot(t, phi_act, color=c[0], lw=1.2, label='φ actual')
    ax_phi.plot(t, phi_cmd, color=c[0], lw=1.0, ls='--', alpha=0.7, label='φ cmd')
    ax_phi.set_ylabel('Roll φ (°)')
    ax_phi.legend(fontsize=8)
    _style_ax(ax_phi)
    ax_phi.set_title('Attitude Commands vs Actual')

    theta_act = rad2deg(col(d, 'theta'))
    theta_cmd = rad2deg(col(d, 'theta_cmd'))
    ax_theta.plot(t, theta_act, color=c[1], lw=1.2, label='θ actual')
    ax_theta.plot(t, theta_cmd, color=c[1], lw=1.0, ls='--', alpha=0.7, label='θ cmd')
    ax_theta.set_ylabel('Pitch θ (°)')
    ax_theta.set_xlabel('Time (s)')
    ax_theta.legend(fontsize=8)
    _style_ax(ax_theta)


def panel_thrust(ax, d, theme):
    t = col(d, 't')
    ax.plot(t, col(d, 'T_cmd'), color=theme['colors'][3], lw=1.2)
    ax.set_ylabel('T_cmd (normalised)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Thrust Command')
    ax.set_ylim(-0.05, 1.05)
    _style_ax(ax)


def panel_airspeed_alt(ax_v, ax_h, d, theme):
    t = col(d, 't')
    c = theme['colors']
    ax_v.plot(t, col(d, 'airspeed'), color=c[4], lw=1.2)
    ax_v.set_ylabel('Airspeed (m/s)')
    _style_ax(ax_v)
    ax_v.set_title('Airspeed & Altitude')

    ax_h.plot(t, -col(d, 'z'), color=c[2], lw=1.2)   # NED z → positive up
    ax_h.set_ylabel('Altitude AGL (m)')
    ax_h.set_xlabel('Time (s)')
    _style_ax(ax_h)


def panel_chi_heading(ax, d, theme):
    t = col(d, 't')
    c = theme['colors']
    ax.plot(t, rad2deg(col(d, 'psi')),   color=c[0], lw=1.2, label='ψ actual')
    ax.plot(t, rad2deg(col(d, 'psi_r')), color=c[0], lw=1.0, ls='--', alpha=0.7, label='ψ_r ref')
    ax.plot(t, rad2deg(col(d, 'chi_err')), color=c[1], lw=1.0, label='χ error')
    ax.set_ylabel('Angle (°)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Heading / Course Error')
    ax.legend(fontsize=8)
    _style_ax(ax)


def panel_rates(ax, d, theme):
    t = col(d, 't')
    c = theme['colors']
    for key, label, color in [
        ('p', 'p  roll rate',  c[0]),
        ('q', 'q  pitch rate', c[1]),
        ('r', 'r  yaw rate',   c[2]),
    ]:
        ax.plot(t, rad2deg(col(d, key)), label=label, color=color, lw=1.0)
    ax.set_ylabel('Body rate (°/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Body Rates  p / q / r')
    ax.legend(fontsize=8)
    _style_ax(ax)


def panel_curvature_gamma(ax_k, ax_g, d, theme):
    t = col(d, 't')
    c = theme['colors']
    ax_k.plot(t, col(d, 'kappa'), color=c[3], lw=1.2)
    ax_k.set_ylabel('κ  curvature (1/m)')
    _style_ax(ax_k)
    ax_k.set_title('Path Curvature & Flight-Path Angle')

    ax_g.plot(t, rad2deg(col(d, 'gamma')), color=c[4], lw=1.2)
    ax_g.set_ylabel('γ  FPA (°)')
    ax_g.set_xlabel('Time (s)')
    _style_ax(ax_g)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(d):
    t        = col(d, 't')
    duration = t[-1] - t[0]
    e_n      = col(d, 'e_n')
    e_t      = col(d, 'e_t')
    e_z      = col(d, 'e_z')
    s1       = col(d, 's1')

    print("\n─── Flight Summary ───────────────────────────────────────────")
    print(f"  Duration      : {duration:.1f} s   ({len(d)} samples)")
    print(f"  Airspeed mean : {col(d,'airspeed').mean():.1f} m/s   "
          f"std {col(d,'airspeed').std():.2f} m/s")
    print(f"  e_n  RMS/max  : {np.sqrt(np.mean(e_n**2)):.3f} m  /  {np.max(np.abs(e_n)):.3f} m")
    print(f"  e_t  RMS/max  : {np.sqrt(np.mean(e_t**2)):.3f} m  /  {np.max(np.abs(e_t)):.3f} m")
    print(f"  e_z  RMS/max  : {np.sqrt(np.mean(e_z**2)):.3f} m  /  {np.max(np.abs(e_z)):.3f} m")
    print(f"  |s1| mean/max : {np.mean(np.abs(s1)):.4f}  /  {np.max(np.abs(s1)):.4f}")
    print(f"  φ_cmd peak    : {math.degrees(np.max(np.abs(col(d,'phi_cmd')))):.1f}°")
    print(f"  θ_cmd peak    : {math.degrees(np.max(np.abs(col(d,'theta_cmd')))):.1f}°")
    print("──────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def build_figure(d, theme, title: str):
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(
        5, 4,
        figure=fig,
        hspace=0.55,
        wspace=0.38,
        left=0.06, right=0.97, top=0.97, bottom=0.04,
    )

    # Row 0: ground track (wide) + curvature/gamma (narrow)
    ax_track = fig.add_subplot(gs[0, :2])
    ax_kappa = fig.add_subplot(gs[0, 2])
    ax_gamma = fig.add_subplot(gs[0, 3])
    panel_track(ax_track, d, theme)
    panel_curvature_gamma(ax_kappa, ax_gamma, d, theme)

    # Row 1: errors (3 stacked sharing x-axis)
    ax_en = fig.add_subplot(gs[1, :2])
    ax_et = fig.add_subplot(gs[2, :2], sharex=ax_en)
    ax_ez = fig.add_subplot(gs[3, :2], sharex=ax_en)
    plt.setp(ax_en.get_xticklabels(), visible=False)
    plt.setp(ax_et.get_xticklabels(), visible=False)
    panel_errors(ax_en, ax_et, ax_ez, d, theme)

    # Row 1-2 right: attitude
    ax_phi   = fig.add_subplot(gs[1, 2:])
    ax_theta = fig.add_subplot(gs[2, 2:], sharex=ax_phi)
    plt.setp(ax_phi.get_xticklabels(), visible=False)
    panel_attitude(ax_phi, ax_theta, d, theme)

    # Row 3 right: sliding surfaces
    ax_s = fig.add_subplot(gs[3, 2:])
    panel_sliding_surfaces(ax_s, d, theme)

    # Row 4: airspeed|altitude | heading | rates | thrust
    ax_v = fig.add_subplot(gs[4, 0])
    ax_h = fig.add_subplot(gs[4, 0])   # twin-style via separate axis below
    # Use twinx for airspeed + altitude on same axes
    ax_v  = fig.add_subplot(gs[4, 0])
    ax_h2 = ax_v.twinx()
    t = col(d, 't')
    ax_v.plot(t, col(d, 'airspeed'), color=theme['colors'][4], lw=1.2, label='airspeed')
    ax_h2.plot(t, -col(d, 'z'), color=theme['colors'][2], lw=1.2, ls='--', label='alt')
    ax_v.set_ylabel('Airspeed (m/s)', fontsize=8)
    ax_h2.set_ylabel('Alt AGL (m)', fontsize=8, color=theme['colors'][2])
    ax_v.set_xlabel('Time (s)')
    ax_v.set_title('Airspeed & Altitude')
    lines1, labels1 = ax_v.get_legend_handles_labels()
    lines2, labels2 = ax_h2.get_legend_handles_labels()
    ax_v.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')
    _style_ax(ax_v)

    ax_chi   = fig.add_subplot(gs[4, 1])
    panel_chi_heading(ax_chi, d, theme)

    ax_pqr   = fig.add_subplot(gs[4, 2])
    panel_rates(ax_pqr, d, theme)

    ax_T     = fig.add_subplot(gs[4, 3])
    panel_thrust(ax_T, d, theme)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Plot SMCLogger .npz flight log')
    ap.add_argument('npz',          help='Path to .npz log file')
    ap.add_argument('--save',       action='store_true',
                    help='Save figure as PNG instead of displaying')
    ap.add_argument('--dark',       action='store_true',
                    help='Use dark theme')
    ap.add_argument('--dpi',        type=int, default=150)
    args = ap.parse_args()

    theme = DARK if args.dark else LIGHT
    apply_theme(theme)

    print(f"[LOAD] {args.npz}")
    d = load_npz(args.npz)
    print(f"[INFO] {len(d)} rows  ×  {d.shape[1]} fields   "
          f"t=[{d[0,0]:.1f} … {d[-1,0]:.1f}] s")

    print_summary(d)

    title = f"SMC Flight Log  –  {Path(args.npz).name}"
    fig   = build_figure(d, theme, title)

    if args.save:
        out = Path(args.npz).with_suffix('.png')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight',
                    facecolor=theme['fig_bg'])
        print(f"[SAVE] {out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
