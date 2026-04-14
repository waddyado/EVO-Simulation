#!/usr/bin/env python3
"""
evo_simulation_3d.py

Interactive 3D EV Field Theory v1 simulator based on EVOPhys-PAPER.pdf.
Runs as a local matplotlib desktop app with no dependencies beyond:
numpy, scipy, sympy, matplotlib.

Developer note — default regime tuning (stable-ish EVO startup):
Earlier defaults produced EEV near/above E_rupture immediately, with 1D guide force
~1/L^2 dominating container transport force, and aggressive radius contraction.
Defaults below bias toward: E_initial well below rupture (~0.5–0.6×E_rupture),
comparable |F_C| and |F_bdy| for typical |x|~µm, gentler radius closure, softer
collapse severity, cooldown + effective rupture margin after events so the
simulation does not chain-retrigger collapses at t=0.

Performance strategy (live matplotlib):
Physics integrates every animation tick (optionally multiple substeps per tick).
Heavy work is throttled: 3D mesh refresh, 2D panel relim/legend, Poisson slice,
and stability (finite-difference of E) use configurable strides via
``_PERFORMANCE_PRESETS``. Figure uses ``subplots_adjust`` instead of
``constrained_layout`` during animation to avoid repeated layout solves.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)
from scipy.optimize import minimize


E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12

# Set True temporarily to print per-callback ms breakdown to stderr (hotspot hunt).
EVO_PROFILE_TIMING = False

# Live view: max points drawn per trace (full history still grows for export unless capped).
PLOT_HISTORY_WINDOW = 160

# Presets: physics remains every step; expensive work is throttled by strides / mesh / grid.
_PERFORMANCE_PRESETS = {
    "quality": {
        "panel_update_stride": 6,
        "scene_update_stride": 5,
        "potential_update_stride": 5,
        "stability_update_stride": 24,
        "jacobi_iters": 14,
        "grid_n": 18,
        "sphere_u_res": 26,
        "sphere_v_res": 15,
        "field_line_res": 44,
        "panel_relim_every": 2,
        "physics_substeps": 1,
        "show_heat_overlay": True,
    },
    "balanced": {
        "panel_update_stride": 8,
        "scene_update_stride": 10,
        "potential_update_stride": 10,
        "stability_update_stride": 32,
        "jacobi_iters": 9,
        "grid_n": 15,
        "sphere_u_res": 20,
        "sphere_v_res": 12,
        "field_line_res": 28,
        "panel_relim_every": 3,
        "physics_substeps": 1,
        "show_heat_overlay": False,
    },
    "performance": {
        "panel_update_stride": 12,
        "scene_update_stride": 14,
        "potential_update_stride": 14,
        "stability_update_stride": 48,
        "jacobi_iters": 6,
        "grid_n": 12,
        "sphere_u_res": 16,
        "sphere_v_res": 10,
        "field_line_res": 22,
        "panel_relim_every": 5,
        "physics_substeps": 1,
        "show_heat_overlay": False,
    },
}


@dataclass
class EVState:
    """State vector S = [Q, R, X, V, Ψ, σ, χ, η] from paper Eq. (2)."""

    Ne: float = 8.0e9
    Q: float = field(init=False)
    R: float = 0.55e-6
    X: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.55e-6]))
    V: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.08, -0.06]))
    A: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.055, 0.035]))
    phi: np.ndarray = field(default_factory=lambda: np.array([0.2, 1.1, 2.2]))
    sigma: float = 0.35
    chi: float = 0.5
    eta: float = 0.42

    def __post_init__(self) -> None:
        self.Q = -self.Ne * E_CHARGE


class EVSimulationApp:
    """Interactive simulator implementing EV Field Theory v1 equations."""

    def __init__(self) -> None:
        print("[EVO] Booting: initializing EV state...", flush=True)
        self.state = EVState()
        self.t = 0.0
        self.dt = 0.03
        self.frame = 0
        self.emission_enabled = True
        self.monte_carlo_result = None
        self.anim_ref = None
        self.last_terms = None
        self.last_stability = (0.0, False)
        self.fps_smooth = 0.0
        self.last_frame_time = None
        self.history: Dict[str, List[float]] = {
            "t": [],
            "sigma": [],
            "chi": [],
            "eta": [],
            "R": [],
            "E": [],
            "Je": [],
            "x": [],
            "v": [],
            "F_C": [],
            "F_bdy": [],
            "radiation_burst_proxy": [],
            "target_interaction_proxy": [],
        }

        # Theory/control parameters (paper Eq. 4, 12, 16, 17, 18).
        self.params = {
            "alphaQ": 1.0,
            "eps_eff": 2.5 * EPS0,
            "a": 2.4e-3,
            "b": 2.1e-15,
            "beta": 1.6,
            "ck": np.array([1.0e-2, 1.4e-2, 2.0e-2]),
            "dk": np.array([0.11, 0.13, 0.16]),
            "R0": 0.9e-6,
            "gamma": 0.17,
            "kC": math.pi / 1.0e-6,
            "Meff": 4.0e-14,
            # Lower interaction drive so startup EEV sits below E_rupture with margin.
            "lambda_int": 4.5e-5,
            "boundary_strength": 2.2e-9,
            "k_container_force": 1.6e-8,
            "gas_drag": 1.2e-3,
            "loss_coeff": 1.4e-18,
            "J0": 2.2e6,
            "Weff": 2.2,
            "ThetaEV": 0.55,
            "m": 1.15,
            "Fcrit": 0.58,
            "Eext": np.array([0.0, 0.0, 4.5e3]),
            "Bext": np.array([0.0, 4.0e-3, 0.0]),
            # Radius dynamics config.
            "enable_radius_dynamics": True,
            "R_initial": 0.55e-6,
            "R_min": 0.18e-6,
            "R_max": 2.8e-6,
            # Softer contraction, modest expansion, stronger damping against slam to R_min.
            "k_expand": 3.5e-8,
            "k_contract": 7.0e-9,
            "k_radius_damp": 14.0,
            "enable_radius_quantization": True,
            "R_quant_base": 1.0e-6,
            "R_quant_strength": 4.5e-8,
            "R_quant_spacing": 0.28e-6,
            # 1D transport config.
            "enable_transport": True,
            "x_initial": 0.0,
            "v_initial": 0.0,
            "M_eff": 4.0e-14,
            "gamma_v": 2.2e-3,
            "boundary_mode": "harmonic",
            # |F_bdy| ~ s|x|/L^2; small s + wider L matches |F_C| ~ k*kappa in default regime.
            "boundary_strength_1d": 4.0e-9,
            "boundary_scale": 2.2e-6,
            "enable_container_transport_force": True,
            "k_container_transport": 2.2e-3,
            # Collapse config.
            "enable_collapse": True,
            # Higher rupture threshold vs typical tuned EEV (~0.5–0.6 of this at startup).
            "E_rupture": 9.5e-8,
            "collapse_mode": "auto",
            "collapse_reset_factor": 0.45,
            "collapse_emission_multiplier": 3.5,
            "collapse_eta_drop": 0.12,
            "collapse_sigma_drop": 0.08,
            "collapse_radius_jump": 0.12,
            "enable_post_collapse_state": True,
            "collapse_cooldown_s": 0.75,
            "post_collapse_rupture_margin": 1.38,
            "post_collapse_margin_decay_s": 1.1,
            # Stochastic perturbation config.
            "enable_noise": False,
            "noise_seed": 44,
            "noise_strength_modes": 0.003,
            "noise_strength_sigma": 0.002,
            "noise_strength_eta": 0.002,
            "noise_strength_radius": 8.0e-10,
            "noise_mode": "additive",
            "mc_vary_seed": True,
        }
        # Snapshot for "Stable defaults" control (tuned regime).
        self._tuned_param_defaults = copy.deepcopy(self.params)
        self.rng = np.random.default_rng(int(self.params["noise_seed"]))
        self.state.R = self.params["R_initial"]
        self.transport_state = {"x": 0.0, "v": 0.0, "F_C": 0.0, "F_bdy": 0.0}
        self.transport_state["x"] = self.params["x_initial"]
        self.transport_state["v"] = self.params["v_initial"]
        self.state.X[0] = self.transport_state["x"]
        self.collapse_events: List[Dict[str, float | str]] = []
        self.collapse_count = 0
        self.lifecycle_state = "active"
        self.collapse_markers_t: List[float] = []
        self.collapse_markers_e: List[float] = []
        self.radiation_burst = 0.0
        self.target_interaction = 0.0
        # Transient margin on rupture threshold after collapse (decays in time).
        self._rupture_scale = 1.0
        self._last_collapse_time = -1e9

        # Performance: see module _PERFORMANCE_PRESETS; default balances FPS vs fidelity.
        self.performance_mode = "balanced"
        self.show_heat_overlay = False
        self._apply_performance_preset(self.performance_mode, initial=True)
        self._panel_relim_counter = 0
        self._static_legends_initialized = False

        # Grid for container law and finite-difference potential (size follows preset grid_n).
        self.grid_extent = 2.2e-6
        self._rebuild_spatial_grid()

        print("[EVO] Booting: building symbolic/model components...", flush=True)
        self.rho_cache = None
        self.uC_cache = None
        self.phi_slice_cache = None
        self.grad2avg_cache = 0.0
        self.grad_phi_mag_cache = 0.0

        self._setup_symbolics()
        print("[EVO] Booting: creating UI and 3D scene...", flush=True)
        self._setup_figure()
        self._draw_static_scene()
        # Radius panel is drawn in _update_panels; avoid forcing expensive immediate redraw here.
        self._boot_completed = False

    def _rebuild_spatial_grid(self) -> None:
        """Rebuild spatial grid after ``grid_n`` or extent changes."""
        self.grid_axis = np.linspace(-self.grid_extent, self.grid_extent, self.grid_n)
        self.dx = self.grid_axis[1] - self.grid_axis[0]
        self.Xg, self.Yg, self.Zg = np.meshgrid(
            self.grid_axis, self.grid_axis, self.grid_axis, indexing="ij"
        )
        self.rg = np.sqrt(self.Xg**2 + self.Yg**2 + self.Zg**2)
        self.rho_cache = None
        self.uC_cache = None
        self.phi_slice_cache = None

    def _apply_performance_preset(self, mode: str, initial: bool = False) -> None:
        """Apply quality / balanced / performance throttles (strides, mesh, Jacobi, grid)."""
        if mode not in _PERFORMANCE_PRESETS:
            mode = "balanced"
        p = _PERFORMANCE_PRESETS[mode]
        self.performance_mode = mode
        self.panel_update_stride = int(p["panel_update_stride"])
        self.scene_update_stride = int(p["scene_update_stride"])
        self.potential_update_stride = int(p["potential_update_stride"])
        self.stability_update_stride = int(p["stability_update_stride"])
        self.jacobi_iters = int(p["jacobi_iters"])
        self.grid_n = int(p["grid_n"])
        self.sphere_u_res = int(p["sphere_u_res"])
        self.sphere_v_res = int(p["sphere_v_res"])
        self.field_line_res = int(p["field_line_res"])
        self.panel_relim_every = int(p["panel_relim_every"])
        self.physics_substeps = int(p["physics_substeps"])
        self.show_heat_overlay = bool(p.get("show_heat_overlay", False))
        if not initial and hasattr(self, "fig"):
            self._rebuild_spatial_grid()
            self._static_legends_initialized = False
            if hasattr(self, "ax_energy"):
                for ax in (self.ax_energy, self.ax_heat, self.ax_quant):
                    leg = ax.get_legend()
                    if leg is not None:
                        leg.remove()

    def _cycle_performance_mode(self) -> None:
        order = ["performance", "balanced", "quality"]
        i = order.index(self.performance_mode) if self.performance_mode in order else 1
        self._apply_performance_preset(order[(i + 1) % len(order)])
        if hasattr(self, "btn_perf"):
            self.btn_perf.label.set_text(f"Perf: {self.performance_mode}")
        ho = "heat on" if self.show_heat_overlay else "heat off"
        self.status_text.set_text(
            f"Status: performance → {self.performance_mode} ({ho}; strides / mesh / grid updated)"
        )

    def _setup_symbolics(self) -> None:
        """Symbolic helper for Eq. (13) consistency check."""
        n, R0, gamma, kC, Rn = sp.symbols("n R0 gamma kC Rn", positive=True)
        self.quant_expr = (Rn - R0 * n**gamma, kC * Rn - n * sp.pi)
        self.quant_lambdas = [sp.lambdify((Rn, n, R0, gamma), self.quant_expr[0], "numpy"),
                              sp.lambdify((Rn, n, kC), self.quant_expr[1], "numpy")]

    def _setup_figure(self) -> None:
        """Grid layout: 3D + state strip |2x3 diagnostics | control row | footer/actions.

        Width/height ratios favor the diagnostic grid so traces stay readable on a typical desktop.
        """
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(18, 11.0), constrained_layout=False)
        # More vertical space for plots (rows 0–1); shorter control strip; narrow footer.
        # Slightly narrower left column pair vs wider right pair so σ/χ/η and traces are larger.
        gs = self.fig.add_gridspec(
            4,
            4,
            height_ratios=[1.22, 1.22, 0.52, 0.20],
            width_ratios=[0.92, 0.92, 1.18, 1.18],
        )

        # 3D still prominent but not dominant; state strip unchanged in role.
        left = gs[0:2, 0:2].subgridspec(2, 1, height_ratios=[3.35, 1.0], hspace=0.10)
        self.ax3d = self.fig.add_subplot(left[0, 0], projection="3d")
        self.ax_state = self.fig.add_subplot(left[1, 0])
        self.ax_state.set_facecolor("#111826")
        self.ax_state.set_xticks([])
        self.ax_state.set_yticks([])
        self.ax_state.set_title("State summary", fontsize=9, color="#d7ecff", pad=5)
        for s in self.ax_state.spines.values():
            s.set_color("#2f4f67")
            s.set_linewidth(0.9)

        # Extra spacing between diagnostic cells so titles/ticks/legends do not crowd.
        diag = gs[0:2, 2:4].subgridspec(2, 3, hspace=0.45, wspace=0.40)
        self.ax_sigma = self.fig.add_subplot(diag[0, 0])
        self.ax_chi = self.fig.add_subplot(diag[0, 1])
        self.ax_eta = self.fig.add_subplot(diag[0, 2])
        self.ax_energy = self.fig.add_subplot(diag[1, 0])
        self.ax_heat = self.fig.add_subplot(diag[1, 1])
        self.ax_quant = self.fig.add_subplot(diag[1, 2])

        ctrl = gs[2, :].subgridspec(1, 4, wspace=0.12)
        self.ax_ctrl_source = self.fig.add_subplot(ctrl[0, 0])
        self.ax_ctrl_transport = self.fig.add_subplot(ctrl[0, 1])
        self.ax_ctrl_collapse = self.fig.add_subplot(ctrl[0, 2])
        self.ax_ctrl_noise = self.fig.add_subplot(ctrl[0, 3])
        self.control_panels = [
            self.ax_ctrl_source,
            self.ax_ctrl_transport,
            self.ax_ctrl_collapse,
            self.ax_ctrl_noise,
        ]

        self.ax_footer = self.fig.add_subplot(gs[3, :])
        self.ax_footer.set_axis_off()
        self.ax_footer.set_facecolor("#0c1018")
        for s in self.ax_footer.spines.values():
            s.set_visible(False)

        for ax in [self.ax_sigma, self.ax_chi, self.ax_eta, self.ax_quant, self.ax_energy, self.ax_heat]:
            ax.set_facecolor("#141722")
            ax.grid(alpha=0.22, linestyle=":", color="#5a6b85")
            ax.tick_params(labelsize=10, colors="#dbe7f5")
            ax.xaxis.label.set_color("#dbe7f5")
            ax.yaxis.label.set_color("#dbe7f5")
            ax.title.set_color("#e8f4ff")

        for ax, title in zip(
            self.control_panels,
            ["Source / Internal", "Transport / Guide", "Collapse / Thresholds", "Stochastic / Monte-Carlo"],
        ):
            ax.set_facecolor("#111826")
            ax.set_title(title, fontsize=8, color="#d7ecff", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2f4f67")
                s.set_linewidth(0.9)

        self.fig.patch.set_facecolor("#10131a")
        self.state_text = self.ax_state.text(
            0.02,
            0.97,
            "",
            transform=self.ax_state.transAxes,
            fontsize=8.0,
            color="#f0f6ff",
            family="monospace",
            va="top",
        )
        self.status_text = self.ax_footer.text(
            0.012,
            0.62,
            "Status: initializing",
            transform=self.ax_footer.transAxes,
            fontsize=8.5,
            color="#9de4ff",
            family="monospace",
            va="bottom",
        )
        self.fps_text = self.ax_footer.text(
            0.72,
            0.62,
            "FPS: --",
            transform=self.ax_footer.transAxes,
            fontsize=8.5,
            color="#ffd166",
            family="monospace",
            va="bottom",
        )

        # Line artists updated in-place in _update_panels (reduces full axes clears).
        self._panel_lines: Dict[str, Optional[Line2D]] = {
            "sigma": None,
            "chi": None,
            "eta": None,
            "energy_e": None,
            "energy_j": None,
            "transport_x": None,
            "transport_v": None,
            "transport_fc": None,
            "transport_fb": None,
            "radius": None,
        }
        self._energy_collapse_scatter = None
        self._collapse_scatter_n = 0
        self._e_rupture_line = None
        self._panels_titles_done = False
        self._radius_pref_lines = None

        # One-time geometry (constrained_layout would re-negotiate every draw and hurts FPS).
        self.fig.subplots_adjust(
            left=0.055, right=0.99, top=0.94, bottom=0.035, hspace=0.45, wspace=0.36
        )

    def _draw_static_scene(self) -> None:
        ax = self.ax3d
        ax.set_title("EV Field Theory v1: State Vector in 3D", fontsize=11, pad=10)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.set_zlabel("z (m)", fontsize=9)
        lim = 3.0e-6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.2e-6, 4.0e-6)
        ax.view_init(elev=26, azim=44)
        ax.set_box_aspect((1.0, 1.0, 1.2))

        # Semi-transparent dielectric/guide boundaries (paper Sec. 10 and postulate 6).
        gx, gy = np.meshgrid(np.linspace(-lim, lim, 20), np.linspace(-lim, lim, 20))
        gz = np.zeros_like(gx)
        self.ax3d.plot_surface(gx, gy, gz, alpha=0.16, color="#4ec6ff", linewidth=0)
        self.ax3d.text(1.8e-6, 1.8e-6, 0.1e-6, "Dielectric plane", color="#7fc6ff", fontsize=8)

        xguide = np.linspace(-lim, lim, 50)
        zguide = np.linspace(0.2e-6, 3.6e-6, 40)
        Xg, Zg = np.meshgrid(xguide, zguide)
        Yg = np.full_like(Xg, 1.7e-6)
        self.ax3d.plot_surface(Xg, Yg, Zg, alpha=0.08, color="#b66dff", linewidth=0)
        self.ax3d.text(-2.4e-6, 1.8e-6, 3.7e-6, "RC/LC guide", color="#d9a2ff", fontsize=8)

        # Placeholders that are replaced every frame.
        self.sphere_artist = None
        self.velocity_artist = None
        self.field_lines = []
        self.heat_scatter = None

    def _build_widgets(self) -> None:
        slider_color = "#20283a"
        self.sliders = {}
        # Short slider track labels; panel titles carry the grouping. boundary_strength_1d scaled as val*1e-10.
        slider_defs = [
            ("a", self.ax_ctrl_source, "a", (1e-4, 1e-2), self.params["a"], "%.2e"),
            ("beta", self.ax_ctrl_source, "beta", (1.05, 2.8), self.params["beta"], "%.2f"),
            ("Ne", self.ax_ctrl_source, "Ne", (0.1, 12.0), self.state.Ne / 1e10, "%.2f"),
            ("k_expand", self.ax_ctrl_source, "k_exp", (0.2, 12.0), self.params["k_expand"] * 1e8, "%.2f"),
            ("k_contract", self.ax_ctrl_source, "k_con", (0.05, 8.0), self.params["k_contract"] * 1e8, "%.2f"),
            ("gamma_v", self.ax_ctrl_transport, "gamma_v", (0.1, 12.0), self.params["gamma_v"] * 1e3, "%.2f"),
            ("boundary", self.ax_ctrl_transport, "U_bdy", (0.5, 120.0), self.params["boundary_strength_1d"] / 1e-10, "%.1f"),
            ("k_transport", self.ax_ctrl_transport, "F_C k", (0.5, 80.0), self.params["k_container_transport"] / 1e-4, "%.1f"),
            ("E_rupture", self.ax_ctrl_collapse, "Erupt", (3.0, 35.0), self.params["E_rupture"] * 1e8, "%.1f"),
            ("collapse_jump", self.ax_ctrl_collapse, "R jmp", (0.0, 0.6), self.params["collapse_radius_jump"], "%.2f"),
            ("noise_modes", self.ax_ctrl_noise, "noise A", (0.0, 0.04), self.params["noise_strength_modes"], "%.3f"),
            ("noise_radius", self.ax_ctrl_noise, "noise R", (0.0, 25.0), self.params["noise_strength_radius"] * 1e12, "%.1f"),
        ]

        local_row: Dict[object, int] = {}
        for key, panel, label, vrange, init, fmt in slider_defs:
            row = local_row.get(panel, 0)
            local_row[panel] = row + 1
            ax_s = panel.inset_axes([0.05, 0.80 - row * 0.165, 0.92, 0.105], facecolor=slider_color)
            s = Slider(ax_s, label, vrange[0], vrange[1], valinit=init, valfmt=fmt)
            s.label.set_fontsize(7.0)
            s.valtext.set_fontsize(7.0)
            s.on_changed(self._on_slider_change)
            self.sliders[key] = s

        # Full-width action bar in footer (no overlap with diagnostics).
        btn_strip = self.ax_footer.inset_axes([0.008, 0.02, 0.984, 0.48])
        btn_strip.set_axis_off()
        bw = 0.112
        gap = 0.006
        x0 = 0.0
        self.btn_min = Button(btn_strip.inset_axes([x0 + 0 * (bw + gap), 0.05, bw, 0.9]), "Minimize EEV", color="#1f3847", hovercolor="#25516a")
        self.btn_launch = Button(btn_strip.inset_axes([x0 + 1 * (bw + gap), 0.05, bw, 0.9]), "Launch EV", color="#294227", hovercolor="#356632")
        self.btn_mc = Button(btn_strip.inset_axes([x0 + 2 * (bw + gap), 0.05, bw, 0.9]), "Monte-Carlo", color="#3f2f22", hovercolor="#704f35")
        self.btn_reset = Button(btn_strip.inset_axes([x0 + 3 * (bw + gap), 0.05, bw, 0.9]), "Reset", color="#3a2732", hovercolor="#6a3b58")
        self.btn_emit = Button(btn_strip.inset_axes([x0 + 4 * (bw + gap), 0.05, bw, 0.9]), "Emission", color="#20342f", hovercolor="#2f5d53")
        self.btn_save = Button(btn_strip.inset_axes([x0 + 5 * (bw + gap), 0.05, bw, 0.9]), "Save Data", color="#26314c", hovercolor="#374d7a")
        self.btn_defaults = Button(btn_strip.inset_axes([x0 + 6 * (bw + gap), 0.05, 0.125, 0.9]), "Stable defaults", color="#1a3d32", hovercolor="#2a6b55")
        self.btn_perf = Button(
            btn_strip.inset_axes([x0 + 6 * (bw + gap) + 0.125 + gap, 0.05, 0.092, 0.9]),
            f"Perf: {self.performance_mode}",
            color="#2a2a40",
            hovercolor="#3d3d5c",
        )
        for b in [
            self.btn_min,
            self.btn_launch,
            self.btn_mc,
            self.btn_reset,
            self.btn_emit,
            self.btn_save,
            self.btn_defaults,
            self.btn_perf,
        ]:
            b.label.set_fontsize(7.0)

        self.btn_min.on_clicked(lambda _: self.minimize_eev())
        self.btn_launch.on_clicked(lambda _: self.launch_ev())
        self.btn_mc.on_clicked(lambda _: self.run_monte_carlo())
        self.btn_reset.on_clicked(lambda _: self.reset_simulation())
        self.btn_emit.on_clicked(lambda _: self.toggle_emission())
        self.btn_save.on_clicked(lambda _: self.save_data())
        self.btn_defaults.on_clicked(lambda _: self.apply_stable_defaults())
        self.btn_perf.on_clicked(lambda _: self._cycle_performance_mode())

    def _on_slider_change(self, _val: float) -> None:
        self.params["a"] = self.sliders["a"].val
        self.params["beta"] = self.sliders["beta"].val
        self.state.Ne = self.sliders["Ne"].val * 1e10
        self.state.Q = -self.state.Ne * E_CHARGE
        self.params["k_expand"] = self.sliders["k_expand"].val * 1e-8
        self.params["k_contract"] = self.sliders["k_contract"].val * 1e-8
        self.params["gamma_v"] = self.sliders["gamma_v"].val * 1e-3
        self.params["boundary_strength_1d"] = self.sliders["boundary"].val * 1e-10
        self.params["k_container_transport"] = self.sliders["k_transport"].val * 1e-4
        self.params["E_rupture"] = self.sliders["E_rupture"].val * 1e-8
        self.params["collapse_radius_jump"] = self.sliders["collapse_jump"].val
        self.params["noise_strength_modes"] = self.sliders["noise_modes"].val
        self.params["noise_strength_radius"] = self.sliders["noise_radius"].val * 1e-12

    def _sync_sliders_from_params(self) -> None:
        """Push current self.params / state into slider widgets after reset/defaults."""
        self.sliders["a"].set_val(self.params["a"])
        self.sliders["beta"].set_val(self.params["beta"])
        self.sliders["Ne"].set_val(self.state.Ne / 1e10)
        self.sliders["k_expand"].set_val(self.params["k_expand"] * 1e8)
        self.sliders["k_contract"].set_val(self.params["k_contract"] * 1e8)
        self.sliders["gamma_v"].set_val(self.params["gamma_v"] * 1e3)
        self.sliders["boundary"].set_val(self.params["boundary_strength_1d"] / 1e-10)
        self.sliders["k_transport"].set_val(self.params["k_container_transport"] / 1e-4)
        self.sliders["E_rupture"].set_val(self.params["E_rupture"] * 1e8)
        self.sliders["collapse_jump"].set_val(self.params["collapse_radius_jump"])
        self.sliders["noise_modes"].set_val(self.params["noise_strength_modes"])
        self.sliders["noise_radius"].set_val(self.params["noise_strength_radius"] * 1e12)

    def apply_stable_defaults(self) -> None:
        """Restore the tuned default parameter set and soft-reset evolution state."""
        self.params = copy.deepcopy(self._tuned_param_defaults)
        self.rng = np.random.default_rng(int(self.params["noise_seed"]))
        self.state = EVState()
        self.state.R = self.params["R_initial"]
        self.state.Q = -self.state.Ne * E_CHARGE
        self.transport_state = {
            "x": self.params["x_initial"],
            "v": self.params["v_initial"],
            "F_C": 0.0,
            "F_bdy": 0.0,
        }
        self.state.X[0] = self.transport_state["x"]
        self.collapse_events.clear()
        self.collapse_markers_t.clear()
        self.collapse_markers_e.clear()
        self.collapse_count = 0
        self.lifecycle_state = "active"
        self._rupture_scale = 1.0
        self._last_collapse_time = -1e9
        self.t = 0.0
        self.frame = 0
        for k in self.history:
            self.history[k].clear()
        self._sync_sliders_from_params()
        self.rho_cache = None
        self.uC_cache = None
        self.phi_slice_cache = None
        self.last_terms = None
        self.last_stability = (0.0, False)
        self.status_text.set_text("Status: restored stable default regime")

    def _rho_charge(self, R: float, A: np.ndarray, phi: np.ndarray, t: float) -> np.ndarray:
        # Core charge density from postulate 4 + modal modulation from Eq. (3).
        sigma = 0.42 * R
        rho0 = self.state.Q / ((2.0 * math.pi) ** 1.5 * sigma**3 + 1e-30)
        theta = np.arctan2(np.sqrt(self.Xg**2 + self.Yg**2), self.Zg + 1e-18)
        ang = np.zeros_like(theta)
        for i, amp in enumerate(A):
            k = i + 1
            ang += amp * np.sin(k * theta + phi[i] + 3.1 * t)
        return rho0 * np.exp(-(self.rg**2) / (2.0 * sigma**2 + 1e-30)) * (1.0 + 0.7 * ang)

    def _container_terms(self, R: float, A: np.ndarray, phi: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray, float]:
        rho = self._rho_charge(R, A, phi, t)
        grad = np.gradient(rho, self.dx, edge_order=1)
        grad2 = grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2
        modal = np.sum(self.params["ck"] * A**2 + self.params["dk"] * A**4)
        uC = -(self.params["a"]) * np.abs(rho) ** self.params["beta"] + self.params["b"] * grad2 + modal
        return rho, uC, float(np.mean(grad2))

    def _energy_terms(
        self,
        R: float,
        A: np.ndarray,
        phi: np.ndarray,
        *,
        rho_uC: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, float]:
        p = self.params
        if rho_uC is None:
            rho, uC, _grad2_avg = self._container_terms(R, A, phi, self.t)
        else:
            rho, uC = rho_uC
        mask = self.rg <= R
        dV = self.dx**3

        EQ = p["alphaQ"] * self.state.Q**2 / (8.0 * math.pi * p["eps_eff"] * max(R, 1e-12))
        EC = float(np.sum(uC[mask]) * dV)
        psi_field = np.zeros_like(rho)
        theta = np.arctan2(np.sqrt(self.Xg**2 + self.Yg**2), self.Zg + 1e-18)
        for i, amp in enumerate(A):
            psi_field += amp * np.cos((i + 1) * theta + phi[i] + 2.7 * self.t)
        Eint = p["lambda_int"] * float(np.sum((np.abs(rho) * psi_field**2)[mask]) * dV)
        Ekin = 0.5 * p["Meff"] * float(np.dot(self.state.V, self.state.V))

        dist_plane = max(self.state.X[2], 1e-9)
        Ebdy = p["boundary_strength"] * (4.0 * math.pi * R**2) * np.exp(-dist_plane / max(R, 1e-12))
        EEV = EQ + EC + Eint + Ekin + Ebdy
        return {"EQ": EQ, "EC": EC, "Eint": Eint, "Ekin": Ekin, "Ebdy": Ebdy, "EEV": EEV}

    def _stability_metric(self, R: float, A: np.ndarray, phi: np.ndarray) -> Tuple[float, bool]:
        # Paper Eq. (10): dE/dR=0 and d2E/dR2>0, with Eq. (11) modal stationarity.
        h = max(0.04 * R, 3e-9)
        e0 = self._energy_terms(R, A, phi)["EEV"]
        em = self._energy_terms(max(1e-9, R - h), A, phi)["EEV"]
        ep = self._energy_terms(R + h, A, phi)["EEV"]
        dE = (ep - em) / (2.0 * h)
        d2E = (ep - 2.0 * e0 + em) / (h * h)
        stable = abs(dE) < 1e-3 * max(abs(e0), 1.0) and d2E > 0.0
        return d2E, stable

    def minimize_eev(self) -> None:
        A0 = self.state.A.copy()
        x0 = np.r_[self.state.R, A0]
        bnds = [(self.params["R_min"], self.params["R_max"])] + [(0.0, 0.35)] * len(A0)

        def objective(x: np.ndarray) -> float:
            R = float(x[0])
            A = np.clip(x[1:], 0.0, 0.5)
            return self._energy_terms(R, A, self.state.phi)["EEV"]

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bnds)
        if res.success:
            self.state.R = float(res.x[0])
            self.state.A = np.array(res.x[1:], dtype=float)
            self.status_text.set_text(
                f"Status: Minimize EEV converged | R*={self.state.R*1e6:.3f} um | EEV={objective(res.x):.3e} J"
            )
        else:
            self.status_text.set_text("Status: Minimize EEV did not converge")

    def launch_ev(self) -> None:
        self.transport_state["x"] = self.params["x_initial"]
        self.transport_state["v"] = self.params["v_initial"]
        self.state.X = np.array([self.transport_state["x"], 0.0, 1.6e-6])
        self.state.V = np.array([0.45, 0.12, -0.08])
        self.t = 0.0
        self.lifecycle_state = "active"
        self.status_text.set_text("Status: EV launched")

    def _formation_score(self, Etip: float, taup: float, Zs: float, G: float, pgas: float, M: float, eps_r: float, kappa: float, xi: float) -> float:
        # Paper Eq. (18): F(...) > Fcrit.
        vector = np.array([Etip / 1e6, taup / 1e-6, Zs, G, 1.0 / (pgas + 1e-6), M, eps_r / 10.0, kappa, xi])
        weights = np.array([0.24, 0.08, 0.06, 0.11, 0.11, 0.1, 0.08, 0.12, 0.1])
        return float(np.dot(vector, weights) / np.sum(weights))

    def run_monte_carlo(self, trials: int = 600) -> None:
        ok = 0
        radii = []
        energies = []
        collapse_hits = 0
        lifetimes = []
        distances = []
        for _ in range(max(trials, 500)):
            if self.params["enable_noise"] and self.params["mc_vary_seed"]:
                self.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
            Etip = self.rng.uniform(1.0e5, 4.0e6)
            taup = self.rng.uniform(0.05e-6, 5.0e-6)
            Zs = self.rng.uniform(0.1, 1.2)
            G = self.rng.uniform(0.2, 1.6)
            pgas = self.rng.uniform(0.02, 2.0)
            M = self.rng.uniform(0.1, 2.0)
            eps_r = self.rng.uniform(1.0, 18.0)
            kappa = self.rng.uniform(0.05, 1.6)
            xi = self.rng.uniform(0.1, 1.4)
            score = self._formation_score(Etip, taup, Zs, G, pgas, M, eps_r, kappa, xi)
            if score > self.params["Fcrit"]:
                ok += 1
                r_final = self.rng.normal(1.0e-6, 0.22e-6)
                e_final = self.rng.lognormal(mean=-16.3, sigma=0.44)
                radii.append(r_final)
                energies.append(e_final)
                lifetime = self.rng.uniform(0.2, 2.4)
                lifetimes.append(lifetime)
                distances.append(self.rng.normal(0.8e-6, 0.2e-6))
                if e_final > self.params["E_rupture"]:
                    collapse_hits += 1
        self.monte_carlo_result = {
            "trials": max(trials, 500),
            "stable": ok,
            "probability": ok / max(trials, 500),
            "radii": np.array(radii),
            "energies": np.array(energies),
            "stable_run_fraction": ok / max(trials, 500),
            "collapse_fraction": collapse_hits / max(max(trials, 500), 1),
            "mean_lifetime": float(np.mean(lifetimes)) if lifetimes else 0.0,
            "mean_final_radius": float(np.mean(radii)) if radii else 0.0,
            "mean_transport_distance": float(np.mean(distances)) if distances else 0.0,
        }
        self.status_text.set_text(
            f"Status: Monte-Carlo {self.monte_carlo_result['stable']}/{self.monte_carlo_result['trials']} stable "
            f"(P={self.monte_carlo_result['probability']:.3f})"
        )

    def reset_simulation(self) -> None:
        self.state = EVState()
        self.state.R = self.params["R_initial"]
        self.t = 0.0
        self.frame = 0
        for k in self.history:
            self.history[k].clear()
        self.transport_state = {"x": self.params["x_initial"], "v": self.params["v_initial"], "F_C": 0.0, "F_bdy": 0.0}
        self.state.X[0] = self.transport_state["x"]
        self.collapse_events.clear()
        self.collapse_markers_t.clear()
        self.collapse_markers_e.clear()
        self.collapse_count = 0
        self.lifecycle_state = "active"
        self._rupture_scale = 1.0
        self._last_collapse_time = -1e9
        self._sync_sliders_from_params()
        self.rho_cache = None
        self.uC_cache = None
        self.phi_slice_cache = None
        self.last_terms = None
        self.last_stability = (0.0, False)
        self.status_text.set_text("Status: simulation reset")

    def toggle_emission(self) -> None:
        self.emission_enabled = not self.emission_enabled
        self.status_text.set_text(f"Status: emission {'ON' if self.emission_enabled else 'OFF'}")

    def save_data(self) -> None:
        fname = "evo_simulation_data.npz"
        np.savez(
            fname,
            history=self.history,
            state=np.array([self.state.Q, self.state.R, *self.state.X, *self.state.V]),
            monte_carlo=self.monte_carlo_result if self.monte_carlo_result is not None else {},
            collapse_events=np.array(self.collapse_events, dtype=object),
            collapse_count=self.collapse_count,
            collapse_times=np.array(self.collapse_markers_t),
            config=np.array(self.params, dtype=object),
            noise_settings=np.array(
                {
                    "enable_noise": self.params["enable_noise"],
                    "noise_seed": self.params["noise_seed"],
                    "noise_strength_modes": self.params["noise_strength_modes"],
                    "noise_strength_sigma": self.params["noise_strength_sigma"],
                    "noise_strength_eta": self.params["noise_strength_eta"],
                    "noise_strength_radius": self.params["noise_strength_radius"],
                    "noise_mode": self.params["noise_mode"],
                },
                dtype=object,
            ),
        )
        self.status_text.set_text(f"Status: saved data to {fname}")

    def _container_potential_slice(self, rho: np.ndarray) -> np.ndarray:
        # Finite-difference / finite-element-like Jacobi relaxation for Phi_C slice.
        mid = self.grid_n // 2
        src = -self.params["a"] * np.abs(rho[:, :, mid]) ** self.params["beta"]
        phi = np.zeros_like(src)
        for _ in range(self.jacobi_iters):
            phi_new = phi.copy()
            phi_new[1:-1, 1:-1] = 0.25 * (
                phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2] - src[1:-1, 1:-1] * self.dx**2
            )
            phi = phi_new
        return phi

    def _boundary_potential_1d(self, x: float) -> float:
        """Guide/channel boundary potential used for 1D transport."""
        mode = self.params["boundary_mode"]
        s = self.params["boundary_strength_1d"]
        L = max(self.params["boundary_scale"], 1e-9)
        if mode == "none":
            return 0.0
        if mode == "harmonic":
            return 0.5 * s * (x / L) ** 2
        if mode == "gaussian":
            return s * (1.0 - math.exp(-0.5 * (x / L) ** 2))
        if mode == "periodic":
            return s * (1.0 - math.cos(2.0 * math.pi * x / L))
        if mode == "barrier":
            return s * math.exp(-0.5 * ((x - 0.7 * L) / (0.2 * L + 1e-12)) ** 2)
        return 0.0

    def _boundary_force_1d(self, x: float) -> float:
        """F_bdy = -dU_bdy/dx from the configured guide potential."""
        h = 0.02 * max(self.params["boundary_scale"], 1e-9)
        up = self._boundary_potential_1d(x + h)
        um = self._boundary_potential_1d(x - h)
        return -(up - um) / (2.0 * h)

    def _container_transport_force_1d(self) -> float:
        """Container-coupled transport force from internal state."""
        if not self.params["enable_container_transport_force"]:
            return 0.0
        coherence = 0.5 * (self.state.sigma + self.state.eta)
        modal = float(np.sum(self.state.A**2))
        return self.params["k_container_transport"] * (0.8 * coherence + 0.2 * modal - 0.4 * self.state.chi)

    def _update_radius_dynamics(self, terms: Dict[str, float]) -> None:
        """Integrate dynamic radius: expand-contract-damp with optional quantized pull."""
        if not self.params["enable_radius_dynamics"]:
            return

        rho_eff = abs(self.state.Q) / (4.0 / 3.0 * math.pi * max(self.state.R, 1e-12) ** 3)
        destabilizing = terms["EQ"] + terms["Ekin"] + max(0.0, 0.3 - self.state.eta) * 1e-7
        stabilizing = abs(terms["EC"]) + abs(self.state.sigma) * rho_eff * 1e-24
        expand = self.params["k_expand"] * destabilizing
        contract = self.params["k_contract"] * stabilizing
        damp = self.params["k_radius_damp"] * (self.state.R - self.params["R_initial"]) * 1e-2
        quant = 0.0

        if self.params["enable_radius_quantization"]:
            base = self.params["R_quant_base"]
            spacing = max(self.params["R_quant_spacing"], 1e-12)
            # Nearest bead center (harmonic well) instead of one-sided indexing toward R_max.
            k = round((self.state.R - base) / spacing)
            r_pref = base + k * spacing
            r_pref = float(np.clip(r_pref, self.params["R_min"], self.params["R_max"]))
            quant = -self.params["R_quant_strength"] * (self.state.R - r_pref)

        noise = 0.0
        if self.params["enable_noise"]:
            xi = self.rng.normal(0.0, self.params["noise_strength_radius"])
            noise = xi if self.params["noise_mode"] == "additive" else xi * self.state.R

        dRdt = expand - contract - damp + quant + noise
        self.state.R = float(np.clip(self.state.R + self.dt * dRdt, self.params["R_min"], self.params["R_max"]))

    def _update_transport_1d(self) -> None:
        """Integrate 1D transport x,v and map x onto rendered 3D position."""
        if not self.params["enable_transport"]:
            return
        x = self.transport_state["x"]
        v = self.transport_state["v"]
        fc = self._container_transport_force_1d()
        fb = self._boundary_force_1d(x)
        gamma_v = self.params["gamma_v"]
        m_eff = max(self.params["M_eff"], 1e-20)
        dvdt = (fc + fb - gamma_v * v) / m_eff
        v = float(np.clip(v + self.dt * dvdt, -30.0, 30.0))
        x = float(np.clip(x + self.dt * v, -2.6e-6, 2.6e-6))
        self.transport_state.update({"x": x, "v": v, "F_C": fc, "F_bdy": fb})
        self.state.X[0] = x
        self.state.V[0] = v

    def _handle_collapse(self, eev: float, je: float) -> float:
        """Apply soft/hard collapse event behavior and return updated Je."""
        if not self.params["enable_collapse"]:
            return je

        # Cooldown prevents collapse cascades from a single overshoot.
        if self.t - self._last_collapse_time < self.params["collapse_cooldown_s"]:
            return je

        eff_rupture = self.params["E_rupture"] * self._rupture_scale
        if eev <= eff_rupture:
            # Hysteresis: leave collapsed/reforming only after energy drops well below nominal threshold.
            if self.lifecycle_state == "collapsed" and eev < 0.62 * self.params["E_rupture"]:
                self.lifecycle_state = "reforming"
            elif self.lifecycle_state == "reforming" and eev < 0.72 * self.params["E_rupture"]:
                self.lifecycle_state = "active"
            return je

        mode = self.params["collapse_mode"]
        if mode == "auto":
            mode = "hard" if eev > 1.45 * eff_rupture else "soft"

        self._last_collapse_time = self.t
        self._rupture_scale = max(self._rupture_scale, self.params["post_collapse_rupture_margin"])

        self.collapse_count += 1
        self.collapse_markers_t.append(self.t)
        self.collapse_markers_e.append(eev)
        pre = {"R": self.state.R, "eta": self.state.eta, "sigma": self.state.sigma, "Q": self.state.Q}

        emul = self.params["collapse_emission_multiplier"]

        if mode == "soft":
            self.state.Q *= (1.0 - 0.04 * self.params["collapse_reset_factor"])
            self.state.eta = max(0.0, self.state.eta - self.params["collapse_eta_drop"] * 0.45)
            self.state.sigma = max(0.0, self.state.sigma - self.params["collapse_sigma_drop"] * 0.35)
            self.state.R = float(
                np.clip(
                    self.state.R * (1.0 + 0.12 * self.params["collapse_radius_jump"]),
                    self.params["R_min"],
                    self.params["R_max"],
                )
            )
            self.state.eta = min(1.0, self.state.eta + 0.06)
            self.state.sigma = min(1.0, self.state.sigma + 0.04)
            self.lifecycle_state = "reforming"
            emul = min(emul, 2.2)
        else:
            self.state.Q *= (1.0 - 0.12 * self.params["collapse_reset_factor"])
            self.state.A *= 0.55
            self.state.eta = max(0.0, self.state.eta - self.params["collapse_eta_drop"])
            self.state.sigma = max(0.0, self.state.sigma - self.params["collapse_sigma_drop"])
            self.state.R = float(
                np.clip(
                    self.state.R * max(0.55, 1.0 - 0.65 * self.params["collapse_radius_jump"]),
                    self.params["R_min"],
                    self.params["R_max"],
                )
            )
            self.state.eta = min(1.0, self.state.eta + 0.05)
            self.state.sigma = min(1.0, self.state.sigma + 0.03)
            self.lifecycle_state = "collapsed"

        self.radiation_burst = emul * je
        self.target_interaction = self.radiation_burst * 0.12
        self.collapse_events.append(
            {
                "time": self.t,
                "energy": eev,
                "mode": mode,
                "pre_R": pre["R"],
                "post_R": self.state.R,
                "pre_eta": pre["eta"],
                "post_eta": self.state.eta,
            }
        )
        return je * emul

    def _forces(self, x: np.ndarray, v: np.ndarray, Je: float, grad_phi_mag: float) -> np.ndarray:
        p = self.params
        FE = self.state.Q * p["Eext"]
        FB = self.state.Q * np.cross(v, p["Bext"])

        # Container gradient points to origin in EV-centered coordinate model.
        FC = -p["k_container_force"] * x / (np.linalg.norm(x) + 2e-9)

        # Boundary force from plane z=0 and guide plane y=1.7um.
        z = x[2]
        Fbdy = np.array([0.0, 0.0, 0.0])
        if z < 1.6e-6:
            Fbdy[2] += p["boundary_strength"] * (1.0 / max(z, 1e-8) ** 2)
        dy = x[1] - 1.7e-6
        Fbdy[1] += -np.sign(dy) * 0.4 * p["boundary_strength"] / (abs(dy) + 1e-8)

        Fgas = -p["gas_drag"] * v
        vhat = v / (np.linalg.norm(v) + 1e-12)
        Floss = p["loss_coeff"] * Je * grad_phi_mag * vhat
        return FE + FB + FC + Fbdy + Fgas - Floss

    def _emission_current(self, grad_phi_mag: float) -> float:
        # Paper Eq. (17): Je = J0 exp(-Weff/ThetaEV) * tau(Psi) * |gradPhi_C|^m
        p = self.params
        tau_psi = 1.0 + 0.8 * float(np.sum(self.state.A**2))
        Je = p["J0"] * math.exp(-p["Weff"] / max(p["ThetaEV"], 1e-6)) * tau_psi * (grad_phi_mag ** p["m"])
        return Je if self.emission_enabled else 0.0

    def _integrate_motion(self, Je: float, grad_phi_mag: float) -> None:
        # Fixed-step semi-implicit update avoids solve_ivp stalls in GUI loop.
        Meff = max(self.params["Meff"], 1e-20)
        x_before = self.transport_state["x"]
        v_before = self.transport_state["v"]
        f = self._forces(self.state.X, self.state.V, Je, grad_phi_mag)
        a = np.clip(f / Meff, -2.0e6, 2.0e6)
        self.state.V = self.state.V + a * self.dt
        self.state.V = np.clip(self.state.V, -120.0, 120.0)
        self.state.X = self.state.X + self.state.V * self.dt

        # Keep EV inside visible domain to avoid runaway trajectories.
        self.state.X[0] = np.clip(self.state.X[0], -2.8e-6, 2.8e-6)
        self.state.X[1] = np.clip(self.state.X[1], -2.8e-6, 2.8e-6)
        self.state.X[2] = np.clip(self.state.X[2], 0.05e-6, 3.8e-6)
        if self.params["enable_transport"]:
            self.state.X[0] = x_before
            self.state.V[0] = v_before

    def _update_3d_scene(self, phi_slice: np.ndarray) -> None:
        ax = self.ax3d
        if self.sphere_artist is not None:
            self.sphere_artist.remove()
        if self.velocity_artist is not None:
            self.velocity_artist.remove()
        for ln in self.field_lines:
            ln.remove()
        self.field_lines.clear()
        if self.heat_scatter is not None:
            self.heat_scatter.remove()
            self.heat_scatter = None

        u = np.linspace(0.0, 2.0 * math.pi, self.sphere_u_res)
        v = np.linspace(0.0, math.pi, self.sphere_v_res)
        uu, vv = np.meshgrid(u, v)

        # Modes on surface (paper Eq. (3) internal state Ψ).
        mode_shape = np.zeros_like(uu)
        for i, amp in enumerate(self.state.A):
            k = i + 1
            mode_shape += amp * np.sin(k * vv + self.state.phi[i] + 2.9 * self.t) * np.cos(k * uu - self.state.phi[i])
        rr = self.state.R * (1.0 + 0.15 * mode_shape)
        xs = self.state.X[0] + rr * np.cos(uu) * np.sin(vv)
        ys = self.state.X[1] + rr * np.sin(uu) * np.sin(vv)
        zs = self.state.X[2] + rr * np.cos(vv)

        # Approximate surface rho from radial interpolation.
        rho_s = -np.abs(self.state.Q) * (1.0 + 0.4 * mode_shape) / (4.0 * math.pi * self.state.R**2 + 1e-30)
        norm = colors.Normalize(vmin=np.min(rho_s), vmax=0.0)
        face_colors = cm.RdBu(norm(rho_s))
        face_colors[..., 3] = 0.68

        self.sphere_artist = ax.plot_surface(
            xs, ys, zs, facecolors=face_colors, linewidth=0, antialiased=False, shade=False
        )

        # Velocity vector V.
        vscale = 3.4e-7
        self.velocity_artist = ax.quiver(
            self.state.X[0], self.state.X[1], self.state.X[2],
            self.state.V[0] * vscale, self.state.V[1] * vscale, self.state.V[2] * vscale,
            color="#ffd166", linewidth=3.0, arrow_length_ratio=0.35
        )

        # Internal field lines (three visible modes).
        for i in range(3):
            theta = np.linspace(0.0, 2.0 * math.pi, self.field_line_res)
            rline = self.state.R * (0.25 + 0.18 * i) * (1.0 + 0.25 * self.state.A[i] * np.sin((i + 1) * theta + self.state.phi[i] + 2.0 * self.t))
            x = self.state.X[0] + rline * np.cos(theta)
            y = self.state.X[1] + rline * np.sin(theta)
            z = self.state.X[2] + 0.14 * self.state.R * np.sin((i + 2) * theta)
            ln, = ax.plot(x, y, z, color=["#55e6c1", "#7db7ff", "#db9dff"][i], lw=1.15, alpha=0.9)
            self.field_lines.append(ln)

        # Heatmap overlay is expensive; disabled by default for responsiveness.
        if self.show_heat_overlay:
            mid = self.grid_n // 2
            stride = 3
            xx = self.state.X[0] + self.Xg[::stride, ::stride, mid]
            yy = self.state.X[1] + self.Yg[::stride, ::stride, mid]
            zz = np.full_like(xx, self.state.X[2] - 1.8e-6)
            pv = phi_slice[::stride, ::stride]
            pvn = (pv - pv.min()) / (pv.ptp() + 1e-18)
            self.heat_scatter = ax.scatter(xx, yy, zz, c=pvn, cmap="plasma", s=8, alpha=0.30, depthshade=False)

    def _update_panels(self, force: bool = False) -> None:
        """Refresh 2D diagnostics; history is appended in ``_integrate_one_step`` each physics step."""
        if self.last_terms is None:
            self.last_terms = self._energy_terms(self.state.R, self.state.A, self.state.phi)
        if force and self.frame == 0:
            self.last_stability = self._stability_metric(self.state.R, self.state.A, self.state.phi)
        terms = self.last_terms
        d2e, stable = self.last_stability

        self._panel_relim_counter += 1
        do_relim = force or (self._panel_relim_counter % max(1, self.panel_relim_every) == 0)

        win = PLOT_HISTORY_WINDOW

        if not self._panels_titles_done:
            for ax, name in [
                (self.ax_sigma, "sigma"),
                (self.ax_chi, "chi"),
                (self.ax_eta, "eta"),
            ]:
                ax.set_xlabel("t (s)", fontsize=10, labelpad=5)
                ax.set_ylabel(name, fontsize=10, labelpad=5)
                ax.grid(alpha=0.22, linestyle=":", color="#5a6b85")
            self.ax_energy.set_xlabel("t (s)", fontsize=10, labelpad=5)
            self.ax_energy.set_ylabel("energy / proxy", fontsize=10, labelpad=5)
            self.ax_energy.grid(alpha=0.22, linestyle=":", color="#5a6b85")
            self.ax_heat.set_xlabel("t (s)", fontsize=10, labelpad=5)
            self.ax_heat.set_ylabel("transport", fontsize=10, labelpad=5)
            self.ax_heat.grid(alpha=0.22, linestyle=":", color="#5a6b85")
            self.ax_quant.set_xlabel("t (s)", fontsize=10, labelpad=5)
            self.ax_quant.set_ylabel("um", fontsize=10, labelpad=5)
            self.ax_quant.grid(alpha=0.22, linestyle=":", color="#5a6b85")
            self._panels_titles_done = True

        def _ensure_line(ax, key: str, color: str, lw: float = 2.1) -> Line2D:
            if self._panel_lines[key] is None:
                (ln,) = ax.plot([], [], color=color, lw=lw)
                self._panel_lines[key] = ln
            return self._panel_lines[key]

        for key, ax, color in [
            ("sigma", self.ax_sigma, "#64d2ff"),
            ("chi", self.ax_chi, "#ffa4de"),
            ("eta", self.ax_eta, "#b3ff96"),
        ]:
            tt = np.asarray(self.history["t"][-win:], dtype=float)
            yy = np.asarray(self.history[key][-win:], dtype=float)
            if len(tt) >= 2:
                ln = _ensure_line(ax, key, color)
                ln.set_data(tt, yy)
                if do_relim:
                    ax.relim()
                    ax.autoscale_view()
            ylab = float(yy[-1]) if len(yy) else float("nan")
            ax.set_title(f"{key} last={ylab:.3f}", fontsize=11, pad=7)
            ax.margins(x=0.05, y=0.16)

        tt = np.asarray(self.history["t"][-win:], dtype=float)
        ee = np.asarray(self.history["E"][-win:], dtype=float)
        je_list = self.history["Je"]
        je = np.asarray(je_list[-win:], dtype=float) if je_list else np.array([])

        if self._panel_lines["energy_e"] is None:
            (self._panel_lines["energy_e"],) = self.ax_energy.plot([], [], color="#ff9d7a", lw=2.0, label="EEV")
        if self._panel_lines["energy_j"] is None:
            (self._panel_lines["energy_j"],) = self.ax_energy.plot([], [], color="#95d5ff", lw=1.65, label="Je x1e-8")
        if len(tt) >= 2:
            self._panel_lines["energy_e"].set_data(tt, ee)
            if len(je) == len(tt):
                self._panel_lines["energy_j"].set_data(tt, je * 1e-8)
            if do_relim:
                self.ax_energy.relim()
                self.ax_energy.autoscale_view()

        eff_rupture = self.params["E_rupture"] * self._rupture_scale
        if self._e_rupture_line is None:
            self._e_rupture_line = self.ax_energy.axhline(
                eff_rupture, color="#c8ccd4", ls="--", lw=1.25, alpha=0.9, label="E_rupture (eff)"
            )
        else:
            self._e_rupture_line.set_ydata([eff_rupture, eff_rupture])

        n_c = len(self.collapse_markers_t)
        if n_c != self._collapse_scatter_n:
            if self._energy_collapse_scatter is not None:
                self._energy_collapse_scatter.remove()
                self._energy_collapse_scatter = None
            if n_c > 0:
                ct = np.array(self.collapse_markers_t)
                ce = np.array(self.collapse_markers_e)
                self._energy_collapse_scatter = self.ax_energy.scatter(
                    ct,
                    ce,
                    s=48,
                    c="#ff3b5c",
                    marker="x",
                    linewidths=2.0,
                    edgecolors="#ff7a93",
                    label="collapse",
                    zorder=6,
                )
            self._collapse_scatter_n = n_c

        self.ax_energy.set_title("Energy + rupture + collapse", fontsize=11, pad=7)
        self.ax_energy.margins(x=0.05, y=0.16)

        xv_t = np.asarray(self.history["t"][-win:], dtype=float)
        if len(xv_t) >= 2:
            _ensure_line(self.ax_heat, "transport_x", "#6ef7b0", 1.9).set_data(
                xv_t, np.asarray(self.history["x"][-win:], dtype=float)
            )
            _ensure_line(self.ax_heat, "transport_v", "#ff9ee6", 1.7).set_data(
                xv_t, np.asarray(self.history["v"][-win:], dtype=float)
            )
            _ensure_line(self.ax_heat, "transport_fc", "#9ed1ff", 1.55).set_data(
                xv_t, np.asarray(self.history["F_C"][-win:], dtype=float) * 1e4
            )
            _ensure_line(self.ax_heat, "transport_fb", "#f7c56e", 1.55).set_data(
                xv_t, np.asarray(self.history["F_bdy"][-win:], dtype=float) * 1e4
            )
            if do_relim:
                self.ax_heat.relim()
                self.ax_heat.autoscale_view()
        self.ax_heat.set_title("Transport x, v, F_C, F_bdy", fontsize=11, pad=7)
        order = ["transport_x", "transport_v", "transport_fc", "transport_fb"]
        labs = ["x (um)", "v (m/s)", "F_C x1e4", "F_bdy x1e4"]
        t_lines = []
        t_labs = []
        for k, lb in zip(order, labs):
            if self._panel_lines[k] is not None:
                t_lines.append(self._panel_lines[k])
                t_labs.append(lb)
        self.ax_heat.margins(x=0.05, y=0.16)

        rr_t = np.asarray(self.history["t"][-win:], dtype=float)
        rr = np.asarray(self.history["R"][-win:], dtype=float)
        if self._panel_lines["radius"] is None:
            (self._panel_lines["radius"],) = self.ax_quant.plot([], [], color="#78dcff", lw=2.15, label="R(t)")
        if len(rr_t) >= 2:
            self._panel_lines["radius"].set_data(rr_t, rr)
            if do_relim:
                self.ax_quant.relim()
                self.ax_quant.autoscale_view()
        if self._radius_pref_lines is None:
            self._radius_pref_lines = [
                self.ax_quant.axhline(
                    self.params["R_quant_base"] * 1e6, color="#f5a35b", ls="--", lw=1.2, label="R_quant_base"
                ),
                self.ax_quant.axhline(self.params["R0"] * 1e6, color="#f25f7f", ls=":", lw=1.2, label="R0"),
            ]
        else:
            y0 = self.params["R_quant_base"] * 1e6
            y1 = self.params["R0"] * 1e6
            self._radius_pref_lines[0].set_ydata([y0, y0])
            self._radius_pref_lines[1].set_ydata([y1, y1])
        self.ax_quant.set_title("Radius R(t) + preferred scales", fontsize=11, pad=7)
        self.ax_quant.margins(x=0.05, y=0.16)

        if not self._static_legends_initialized and len(self.history["t"]) >= 2:
            if self._panel_lines["energy_e"] is not None and self._e_rupture_line is not None:
                self.ax_energy.legend(
                    [self._panel_lines["energy_e"], self._panel_lines["energy_j"], self._e_rupture_line],
                    ["EEV", "Je x1e-8", "E_rupture (eff)"],
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    facecolor="#1a2332",
                    edgecolor="#3d5a73",
                    loc="upper left",
                )
            if t_lines:
                self.ax_heat.legend(
                    t_lines,
                    t_labs,
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    facecolor="#1a2332",
                    edgecolor="#3d5a73",
                    loc="upper left",
                )
            if self._radius_pref_lines is not None and self._panel_lines["radius"] is not None:
                self.ax_quant.legend(
                    [self._panel_lines["radius"], self._radius_pref_lines[0], self._radius_pref_lines[1]],
                    ["R(t)", "R_quant_base", "R0"],
                    fontsize=8,
                    frameon=True,
                    fancybox=False,
                    facecolor="#1a2332",
                    edgecolor="#3d5a73",
                    loc="upper left",
                )
            self._static_legends_initialized = True

        mc_text = ""
        if self.monte_carlo_result is not None:
            mc_text = (
                f"\nMC stable={self.monte_carlo_result['stable_run_fraction']:.2f} "
                f"collapse={self.monte_carlo_result['collapse_fraction']:.2f}"
            )
        eff_er = self.params["E_rupture"] * self._rupture_scale
        self.state_text.set_text(
            f"State: {self.lifecycle_state}   t={self.t:.2f} s\n"
            f"Q={self.state.Q:.3e} C Ne={self.state.Ne:.2e}\n"
            f"R={self.state.R*1e6:.3f} um   x={self.transport_state['x']*1e6:.3f} um   v={self.transport_state['v']:.2f}\n"
            f"E_rupture={self.params['E_rupture']:.2e}   eff={eff_er:.2e}   collapses={self.collapse_count}{mc_text}"
        )

        stab = "stable" if stable else "metastable/unstable"
        armed = "armed" if self.params["enable_collapse"] else "off"
        self.status_text.set_text(
            f"Status: {self.lifecycle_state} | t={self.t:.2f}s | collapse={armed} | "
            f"R*={self.state.R*1e6:.3f} um | EEV={terms['EEV']:.3e} J | d2E/dR2={d2e:.3e} | {stab}"
        )
    def _integrate_one_step(self) -> None:
        # Relax transient rupture margin toward 1 after a collapse (avoids instant retrigger).
        tau = max(self.params["post_collapse_margin_decay_s"], 1e-6)
        if self._rupture_scale > 1.0:
            self._rupture_scale = 1.0 + (self._rupture_scale - 1.0) * math.exp(-self.dt / tau)

        # Dynamic interaction channels σ, χ, η.
        self.state.sigma = 0.35 + 0.1 * math.sin(1.4 * self.t + 0.4)
        self.state.chi = 0.5 + 0.12 * math.sin(1.0 * self.t + 1.1)
        self.state.eta = 0.42 + 0.09 * math.sin(1.9 * self.t + 2.0)
        self.state.phi += self.dt * np.array([0.7, 1.0, 1.25])
        self.radiation_burst *= 0.92
        self.target_interaction *= 0.94

        # Optional stochastic perturbation (reproducible by seed).
        if self.params["enable_noise"]:
            mode_noise = self.rng.normal(0.0, self.params["noise_strength_modes"], size=self.state.A.shape)
            if self.params["noise_mode"] == "multiplicative":
                self.state.A = np.clip(self.state.A * (1.0 + mode_noise), 0.0, 0.6)
            else:
                self.state.A = np.clip(self.state.A + mode_noise, 0.0, 0.6)
            self.state.sigma = float(np.clip(self.state.sigma + self.rng.normal(0.0, self.params["noise_strength_sigma"]), 0.0, 1.0))
            self.state.eta = float(np.clip(self.state.eta + self.rng.normal(0.0, self.params["noise_strength_eta"]), 0.0, 1.0))

        rho_fresh = False
        if self.frame % self.potential_update_stride == 0 or self.rho_cache is None or self.phi_slice_cache is None:
            self.rho_cache, self.uC_cache, self.grad2avg_cache = self._container_terms(
                self.state.R, self.state.A, self.state.phi, self.t
            )
            self.phi_slice_cache = self._container_potential_slice(self.rho_cache)
            self.grad_phi_mag_cache = float(np.sqrt(np.mean(np.gradient(self.phi_slice_cache, self.dx, edge_order=1)[0] ** 2)))
            rho_fresh = True
        phi_slice = self.phi_slice_cache
        grad_phi_mag = self.grad_phi_mag_cache
        Je = self._emission_current(grad_phi_mag)

        if self.last_terms is None:
            self.last_terms = self._energy_terms(
                self.state.R,
                self.state.A,
                self.state.phi,
                rho_uC=(self.rho_cache, self.uC_cache) if rho_fresh and self.rho_cache is not None else None,
            )
        terms_for_rc = self.last_terms
        self._update_radius_dynamics(terms_for_rc)
        self._update_transport_1d()
        Je = self._handle_collapse(terms_for_rc["EEV"], Je)

        self._integrate_motion(Je, grad_phi_mag + 1e-12 * self.grad2avg_cache)

        self.last_terms = self._energy_terms(self.state.R, self.state.A, self.state.phi)
        if self.frame % self.stability_update_stride == 0:
            self.last_stability = self._stability_metric(self.state.R, self.state.A, self.state.phi)

        self.history["t"].append(self.t)
        self.history["sigma"].append(self.state.sigma)
        self.history["chi"].append(self.state.chi)
        self.history["eta"].append(self.state.eta)
        self.history["R"].append(self.state.R * 1e6)
        self.history["E"].append(self.last_terms["EEV"])
        self.history["x"].append(self.transport_state["x"] * 1e6)
        self.history["v"].append(self.transport_state["v"])
        self.history["F_C"].append(self.transport_state["F_C"])
        self.history["F_bdy"].append(self.transport_state["F_bdy"])
        self.history["radiation_burst_proxy"].append(self.radiation_burst)
        self.history["target_interaction_proxy"].append(self.target_interaction)
        self.history["Je"].append(Je)

        self.t += self.dt
        self.frame += 1

        if (self.frame - 1) % self.scene_update_stride == 0:
            self._update_3d_scene(phi_slice)
        if (self.frame - 1) % self.panel_update_stride == 0:
            self._update_panels()

    def animate(self, _frame_id: int) -> None:
        if not self._boot_completed:
            # Defer expensive first panel computation to first render tick.
            self._update_panels(force=True)
            self._boot_completed = True
            print("[EVO] Booted: interactive window ready.", flush=True)
        now = time.perf_counter()
        if self.last_frame_time is None:
            self.last_frame_time = now
        else:
            dt_local = max(now - self.last_frame_time, 1e-9)
            fps_inst = 1.0 / dt_local
            self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * fps_inst if self.fps_smooth > 0 else fps_inst
            self.fps_text.set_text(f"FPS(sim): {self.fps_smooth:5.1f} | dt={self.dt:.3f}s")
            self.last_frame_time = now

        n_sub = max(1, self.physics_substeps)
        if EVO_PROFILE_TIMING:
            t0 = time.perf_counter()
            for _ in range(n_sub):
                self._integrate_one_step()
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            print(f"[EVO profile] frame {_frame_id}: {n_sub} step(s) {elapsed_ms:.2f} ms", flush=True)
        else:
            for _ in range(n_sub):
                self._integrate_one_step()

    def run(self) -> None:
        print("[EVO] Booting: wiring controls and starting renderer...", flush=True)
        self._build_widgets()
        self.anim_ref = FuncAnimation(self.fig, self.animate, interval=20, blit=False, cache_frame_data=False)
        plt.show()


def main() -> None:
    print("[EVO] Booting: launching simulator process...", flush=True)
    app = EVSimulationApp()
    app.run()


if __name__ == "__main__":
    main()
