"""Project SIGNAL Phase 1 dashboard-style desktop simulator."""

from __future__ import annotations

from collections import deque
import csv
import json
import math
from threading import Event, Thread
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import webbrowser

import numpy as np

from signal_lab.experiment.clustering import ClusterObservation, ClusterTracker, lifetime_class
from signal_lab.experiment.networks import NetworkObservation, NetworkTracker
from signal_lab.experiment.protocol import (
    BRANCH_NAMES,
    ExperimentRun,
    ExperimentSpec,
    MeasurementZone,
    ReferenceState,
    capture_reference,
    discover_targets,
    load_experiment,
    run_experiment,
    save_experiment,
)
from signal_lab.physics.engine import SimulationConfig, SimulationEngine
from signal_lab.physics.genome import Genome
from signal_lab.physics.particle import ParticleState
from signal_lab.storage.replay import save_structure_snapshot
from signal_lab.storage.replay import load_structure_snapshot
from signal_lab.storage.export import collect_export_data, export_all
from signal_lab.search.runner import SearchConfig, load_baseline_rows, run_search
from signal_lab.ui.three_viewer import ThreeViewerServer


class SimulatorView:
    """Interactive dashboard closely matching the supplied Project SIGNAL mockup."""

    COLORS = ("#3b9cff", "#ffad3d", "#4fd27d", "#f05d5e", "#b084cc", "#42c6c9", "#e77bdb", "#d9e36a")
    BG = "#07131c"
    PANEL = "#0d1c27"
    PANEL_ALT = "#112532"
    BORDER = "#294354"
    TEXT = "#e9f2f8"
    MUTED = "#9db0bd"

    def __init__(self, seed: int = 0, particle_count: int = 1000) -> None:
        self.root = tk.Tk()
        self.root.title("Project SIGNAL — Experiment #1")
        self.root.geometry("1536x1024")
        self.root.minsize(1180, 760)
        self.root.configure(bg=self.BG)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            pass

        self.engine = SimulationEngine(config=SimulationConfig(particle_count=particle_count, seed=seed))
        self.cluster_tracker = ClusterTracker(self.engine.config.width, self.engine.config.height, self.engine.genome.interaction_radius, min_cluster_size=5)
        self.network_tracker = NetworkTracker(self.engine.config.width, self.engine.config.height, self.engine.genome.interaction_radius)
        self.current_clusters: list[ClusterObservation] = []
        self.current_networks: list[NetworkObservation] = []
        self.saved_network_ids: set[int] = set()
        self.selected_cluster_id: int | None = None
        self.next_cluster_observation_step = 0
        self.running = False
        self.fullscreen_view = False
        self.show_trails = tk.BooleanVar(value=True)
        self.show_clusters = tk.BooleanVar(value=True)
        self.show_radius = tk.BooleanVar(value=False)
        self.show_vectors = tk.BooleanVar(value=False)
        self.show_structure_debug = tk.BooleanVar(value=True)
        self.color_species = tk.BooleanVar(value=True)
        self.dark_theme = tk.BooleanVar(value=True)
        self.seed_var = tk.StringVar(value=str(seed))
        self.status_var = tk.StringVar(value="Ready")
        self.active_metric = tk.StringVar(value="Average Speed")
        self.parameter_vars: dict[str, tk.StringVar] = {}
        self.matrix_vars: list[list[tk.StringVar]] = []
        self.trail_points: deque[np.ndarray] = deque(maxlen=18)
        self.history: dict[str, deque[float]] = {
            "Average Speed": deque(maxlen=1000),
            "Cluster Count": deque(maxlen=1000),
            "Entropy": deque(maxlen=1000),
            "Coherence": deque(maxlen=1000),
        }
        self.events: deque[str] = deque(maxlen=80)
        self.last_metrics_step = -1
        self.search_thread: Thread | None = None
        self.search_stop_event: Event | None = None
        self.three_viewer: ThreeViewerServer | None = None
        self.experiment_window: tk.Toplevel | None = None
        self.experiment_thread: Thread | None = None
        self.experiment_pause_event: Event | None = None
        self.experiment_reference: ReferenceState | None = None
        self.current_experiment: ExperimentRun | None = None
        self._configure_styles()
        self._build_ui()
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen_simulation())
        self.root.bind("<Escape>", lambda _event: self.exit_fullscreen_simulation())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._log("System ready")
        self._refresh_metrics()
        self._draw()
        self.root.after(100, self._draw)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Dashboard.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 11, "bold"))
        style.configure("Body.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 9))
        style.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Segoe UI", 8))
        style.configure("Status.TLabel", background=self.BG, foreground="#94afc1", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 16, "bold"))
        style.configure("Dash.TButton", background="#1a3040", foreground=self.TEXT, bordercolor="#38576a", relief="flat", padding=(9, 7), font=("Segoe UI", 9))
        style.map("Dash.TButton", background=[("active", "#27475b"), ("pressed", "#0d6ea9")])
        style.configure("Primary.TButton", background="#087ec7", foreground="white", bordercolor="#2ba2ed", relief="flat", padding=(9, 7), font=("Segoe UI", 9, "bold"))
        style.map("Primary.TButton", background=[("active", "#1597e6"), ("pressed", "#075c93")])
        style.configure("Nav.TButton", background=self.BG, foreground=self.TEXT, bordercolor=self.BORDER, relief="flat", padding=(16, 8), font=("Segoe UI", 9))
        style.map("Nav.TButton", background=[("active", "#15364b")])
        style.configure("SelectedNav.TButton", background="#087ec7", foreground="white", bordercolor="#2ba2ed", relief="flat", padding=(16, 8), font=("Segoe UI", 9, "bold"))
        style.configure("TEntry", fieldbackground="#07131c", foreground=self.TEXT, insertcolor=self.TEXT, bordercolor=self.BORDER, padding=4)
        style.configure("TCombobox", fieldbackground="#07131c", foreground=self.TEXT, background="#07131c", arrowcolor=self.TEXT)
        style.configure("Horizontal.TScale", background=self.PANEL, troughcolor="#263d4c", sliderthickness=11)
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.TEXT, font=("Segoe UI", 9))

    def _panel(self, parent: tk.Misc, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Panel.TFrame", padding=8)
        title_label = ttk.Label(outer, text=title, style="PanelTitle.TLabel")
        title_label.pack(anchor="w", pady=(0, 8))
        outer.body = ttk.Frame(outer, style="Panel.TFrame")  # type: ignore[attr-defined]
        outer.body.pack(fill=tk.BOTH, expand=True)  # type: ignore[attr-defined]
        return outer

    @staticmethod
    def _body(panel: ttk.Frame) -> ttk.Frame:
        return panel.body  # type: ignore[attr-defined]

    def _build_ui(self) -> None:
        root = ttk.Frame(self.root, style="Dashboard.TFrame", padding=(8, 5, 8, 0))
        root.pack(fill=tk.BOTH, expand=True)
        root.rowconfigure(2, weight=1)
        root.columnconfigure(0, weight=1)

        header = ttk.Frame(root, style="Dashboard.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Project SIGNAL — Experiment #1", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="⚙  Settings     ?", style="Status.TLabel").grid(row=0, column=1, sticky="e")

        nav = ttk.Frame(root, style="Dashboard.TFrame")
        nav.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for index, label in enumerate(("Simulator", "Experiment", "Search", "Results", "Analysis", "Database")):
            style = "SelectedNav.TButton" if index == 0 else "Nav.TButton"
            ttk.Button(nav, text=label, style=style, command=lambda name=label: self._nav_notice(name)).grid(row=0, column=index, padx=(0, 7), sticky="ew")

        workspace = ttk.Frame(root, style="Dashboard.TFrame")
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=0, minsize=225)
        workspace.columnconfigure(1, weight=1, minsize=560)
        workspace.columnconfigure(2, weight=0, minsize=455)
        workspace.rowconfigure(0, weight=1)

        left = ttk.Frame(workspace, style="Dashboard.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        left_canvas = tk.Canvas(left, background=self.BG, highlightthickness=0, borderwidth=0)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_body = ttk.Frame(left_canvas, style="Dashboard.TFrame")
        left_body.columnconfigure(0, weight=1)
        left_window = left_canvas.create_window((0, 0), window=left_body, anchor="nw")
        left_body.bind("<Configure>", lambda _event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))

        def scroll_left(event: tk.Event) -> None:
            if event.delta:
                left_canvas.yview_scroll(int(-event.delta / 120), "units")

        left_canvas.bind("<Enter>", lambda _event: left_canvas.bind_all("<MouseWheel>", scroll_left))
        left_canvas.bind("<Leave>", lambda _event: left_canvas.unbind_all("<MouseWheel>"))
        self.left_canvas = left_canvas
        self.left_body = left_body
        control_panel = self._panel(left_body, "Simulation Control")
        control_panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for text, command, style in (("▶   Start", self.start, "Primary.TButton"), ("Ⅱ   Pause", self.pause, "Dash.TButton"), ("▶|  Step (1 frame)", lambda: self._advance(1), "Dash.TButton"), ("↻   Reset", self.reset, "Dash.TButton"), ("⚄   Random Genome", self.randomize_genome, "Dash.TButton"), ("▣   Load Genome", self.load_genome, "Dash.TButton"), ("▤   Save Genome", self.save_genome, "Dash.TButton"), ("⚙   Change Seed", self.apply_seed, "Dash.TButton")):
            ttk.Button(control_panel, text=text, command=command, style=style).pack(fill=tk.X, pady=2)
        ttk.Button(control_panel, text="⛶   Fullscreen Simulation", command=self.toggle_fullscreen_simulation, style="Dash.TButton").pack(fill=tk.X, pady=(7, 2))
        ttk.Button(control_panel, text="✦   Three.js Particle View", command=self.open_three_view, style="Dash.TButton").pack(fill=tk.X, pady=(2, 2))
        ttk.Button(control_panel, text="⇩   Export Results...", command=self._export_results_dialog, style="Dash.TButton").pack(fill=tk.X, pady=(2, 2))

        lower_left = ttk.Frame(left_body, style="Dashboard.TFrame")
        lower_left.grid(row=1, column=0, sticky="nsew")
        lower_left.rowconfigure(1, weight=1)
        parameters = self._panel(lower_left, "Simulation Parameters")
        parameters.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_sliders(parameters)
        options = self._panel(lower_left, "Display Options")
        options.grid(row=1, column=0, sticky="new")
        for label, variable in (("Show Trails", self.show_trails), ("Show Clusters", self.show_clusters), ("Show Interaction Radius", self.show_radius), ("Show Velocity Vectors", self.show_vectors), ("Structure Debug", self.show_structure_debug), ("Color by Species", self.color_species), ("Dark Theme", self.dark_theme)):
            ttk.Checkbutton(options, text=label, variable=variable, command=self._draw).pack(anchor="w", pady=2)

        center = ttk.Frame(workspace, style="Dashboard.TFrame")
        center.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        view_panel = self._panel(center, "Simulation View")
        self.view_panel = view_panel
        view_panel.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        view_panel.rowconfigure(1, weight=1)
        view_panel.columnconfigure(0, weight=1)
        self.view_title = view_panel.winfo_children()[0]
        # The panel helper creates an expandable body frame. Keep the canvas
        # inside that body so the body does not consume height above it.
        view_body = self._body(view_panel)
        self.canvas = tk.Canvas(view_body, background="#000306", highlightthickness=1, highlightbackground=self.BORDER)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw())
        self.canvas.bind("<Button-1>", self._select_cluster_from_canvas)
        self.legend = tk.Frame(self.canvas, bg="#07131c", highlightbackground=self.BORDER, highlightthickness=1)
        self._build_legend()
        self._build_time_series(center)

        right = ttk.Frame(workspace, style="Dashboard.TFrame")
        right.grid(row=0, column=2, sticky="nsew")
        right.rowconfigure(2, weight=1)
        self._build_matrix_panel(right)
        self._build_genome_panel(right)
        self._build_stats_panel(right)

        bottom = ttk.Frame(root, style="Dashboard.TFrame")
        bottom.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=1)
        self._build_cluster_panel(bottom)
        self._build_event_panel(bottom)
        self._build_footer(root)
        self.dashboard_root = root
        self.header = header
        self.nav = nav
        self.workspace = workspace
        self.left = left
        self.center = center
        self.right = right
        self.bottom = bottom

    def _build_sliders(self, panel: ttk.Frame) -> None:
        body = self._body(panel)
        definitions = (("Particle Count", "particle_count", 100, 3000, 1000), ("Interaction Radius", "interaction_radius", 20, 150, 60), ("Core Radius Ratio", "core_radius_ratio", 0.05, 0.5, 0.18), ("Core Repulsion", "core_repulsion", 0, 3, 1.0), ("Force Gain", "force_gain", 0, 3, 1.0), ("Damping", "damping", 0.5, 1.0, 0.94), ("Time Step (dt)", "dt", 0.01, 0.5, 0.1))
        for row, (label, key, minimum, maximum, value) in enumerate(definitions):
            variable = tk.DoubleVar(value=value)
            self.parameter_vars[key] = tk.StringVar(value=self._format_parameter(key, value))
            ttk.Label(body, text=label, style="Body.TLabel").grid(row=row * 2, column=0, sticky="w", pady=(2, 0))
            ttk.Label(body, textvariable=self.parameter_vars[key], style="Body.TLabel").grid(row=row * 2, column=1, sticky="e", pady=(2, 0))
            ttk.Scale(body, from_=minimum, to=maximum, variable=variable, command=lambda raw, k=key: self._slider_changed(k, raw)).grid(row=row * 2 + 1, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        body.columnconfigure(0, weight=1)

    def _build_matrix_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Interaction Matrix (Asymmetric)")
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        body = self._body(panel)
        table = tk.Frame(body, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1)
        table.pack(fill=tk.X)
        tk.Label(table, text="From \\ To", bg=self.PANEL_ALT, fg=self.TEXT, width=10, pady=6).grid(row=0, column=0, sticky="nsew")
        for column in range(3):
            tk.Label(table, text=f"Species {column}", bg=self.PANEL_ALT, fg=self.TEXT, width=12, pady=6).grid(row=0, column=column + 1, sticky="nsew")
        self.matrix_vars = []
        for row in range(3):
            tk.Label(table, text=f"Species {row}", bg=self.PANEL_ALT, fg=self.TEXT, width=10, pady=5).grid(row=row + 1, column=0, sticky="nsew")
            row_vars: list[tk.StringVar] = []
            for column in range(3):
                variable = tk.StringVar(value=f"{self.engine.genome.interaction_matrix[row, column]:.2f}")
                row_vars.append(variable)
                tk.Entry(table, textvariable=variable, justify="center", bg="#0a1821", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", width=12).grid(row=row + 1, column=column + 1, padx=1, pady=1, sticky="nsew")
            self.matrix_vars.append(row_vars)
        for column in range(4):
            table.columnconfigure(column, weight=1)
        buttons = ttk.Frame(body, style="Panel.TFrame")
        buttons.pack(fill=tk.X, pady=(7, 0))
        ttk.Button(buttons, text="Randomize Matrix", style="Dash.TButton", command=self.randomize_matrix).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(buttons, text="Apply", style="Dash.TButton", command=self.apply_matrix).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

    def _build_genome_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Genome Parameters")
        panel.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        body = self._body(panel)
        fields = (("Interaction Radius", "interaction_radius"), ("Core Radius Ratio", "core_radius_ratio"), ("Core Repulsion", "core_repulsion"), ("Force Gain", "force_gain"), ("Damping", "damping"), ("Time Step (dt)", "dt"))
        form = ttk.Frame(body, style="Panel.TFrame")
        form.pack(fill=tk.X)
        for row, (label, key) in enumerate(fields):
            ttk.Label(form, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(form, textvariable=self.parameter_vars[key], width=9).grid(row=row, column=1, sticky="e", pady=3, padx=(8, 0))
        ttk.Label(form, text="Species Count", style="Body.TLabel").grid(row=0, column=2, sticky="w", padx=(16, 0), pady=3)
        ttk.Combobox(form, values=("3",), state="readonly", width=8).grid(row=0, column=3, sticky="e", pady=3)
        ttk.Label(form, text="Random Seed", style="Body.TLabel").grid(row=1, column=2, sticky="w", padx=(16, 0), pady=3)
        seed_row = ttk.Frame(form, style="Panel.TFrame")
        seed_row.grid(row=1, column=3, sticky="e", pady=3)
        ttk.Entry(seed_row, textvariable=self.seed_var, width=8).pack(side=tk.LEFT)
        ttk.Button(seed_row, text="↻", width=2, style="Dash.TButton", command=self.apply_seed).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(body, text="▤  Save Genome...", style="Dash.TButton", command=self.save_genome).pack(fill=tk.X, pady=(8, 3))
        ttk.Button(body, text="↻  Load Genome...", style="Dash.TButton", command=self.load_genome).pack(fill=tk.X)

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Statistics (Live)")
        panel.grid(row=2, column=0, sticky="nsew")
        body = self._body(panel)
        stats = (("Average Speed", "avg_speed"), ("Speed Variance", "speed_variance"), ("Cluster Count", "cluster_count"), ("Largest Cluster", "largest_cluster"), ("Spatial Entropy", "entropy"), ("Directional Coherence", "coherence"))
        self.stat_labels: dict[str, ttk.Label] = {}
        for row, (label, key) in enumerate(stats):
            ttk.Label(body, text=label, style="Body.TLabel").grid(row=row, column=0 if row < 3 else 2, sticky="w", pady=5, padx=(0, 10) if row < 3 else (12, 10))
            value = ttk.Label(body, text="0.00", style="Body.TLabel")
            value.grid(row=row, column=1 if row < 3 else 3, sticky="e", pady=5)
            self.stat_labels[key] = value
        distribution = ttk.Frame(body, style="Panel.TFrame")
        distribution.grid(row=0, column=4, rowspan=6, sticky="nsew", padx=(16, 0))
        ttk.Label(distribution, text="Species Distribution", style="Body.TLabel").pack(anchor="w")
        self.pie_canvas = tk.Canvas(distribution, width=100, height=105, bg=self.PANEL, highlightthickness=0)
        self.pie_canvas.pack(side=tk.LEFT, pady=5)
        self.species_labels = ttk.Frame(distribution, style="Panel.TFrame")
        self.species_labels.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))
        self.distribution_labels: list[ttk.Label] = []
        for index in range(3):
            label = ttk.Label(self.species_labels, text=f"■  Species {index}   33.3%", style="Body.TLabel")
            label.pack(anchor="w", pady=3)
            self.distribution_labels.append(label)

    def _build_time_series(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Time Series (Last 1,000 steps)")
        self.time_series_panel = panel
        panel.grid(row=1, column=0, sticky="ew")
        body = self._body(panel)
        tabs = ttk.Frame(body, style="Panel.TFrame")
        tabs.pack(fill=tk.X, pady=(0, 5))
        for metric in self.history:
            ttk.Button(tabs, text=metric, style="SelectedNav.TButton" if metric == "Average Speed" else "Nav.TButton", command=lambda m=metric: self._select_metric(m)).pack(side=tk.LEFT, padx=(0, 2))
        self.chart_canvas = tk.Canvas(body, height=118, bg="#07131c", highlightbackground=self.BORDER, highlightthickness=1)
        self.chart_canvas.pack(fill=tk.X, expand=True)
        self.chart_canvas.bind("<Configure>", lambda _event: self._draw_chart())

    def _build_cluster_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Cluster Analysis")
        self.cluster_panel = panel
        self.cluster_panel_collapsed = False
        self.cluster_panel_toggle = ttk.Button(panel, text="−", width=2, style="Dash.TButton", command=self._toggle_cluster_panel)
        self.cluster_panel_toggle.place(relx=1.0, y=0, anchor="ne")
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        body = self._body(panel)
        self.cluster_canvas = tk.Canvas(body, width=255, height=150, bg="#000306", highlightbackground=self.BORDER, highlightthickness=1)
        self.cluster_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        self.cluster_canvas.bind("<Button-1>", self._select_cluster_from_preview)
        info = ttk.Frame(body, style="Panel.TFrame")
        info.grid(row=1, column=1, sticky="nsew")
        self.cluster_info = ttk.Label(info, text="Clusters: 0\nLargest: 0\nMean Size: 0.0\nTracker Persistence: 0.00", style="Body.TLabel", justify=tk.LEFT)
        self.cluster_info.pack(anchor="nw", pady=(3, 10))
        self.cluster_detail = ttk.Label(info, text="Select a cluster to inspect", style="Body.TLabel", justify=tk.LEFT)
        self.cluster_detail.pack(anchor="nw", pady=(0, 6))
        ttk.Label(info, text="Particle count history", style="Muted.TLabel").pack(anchor="w")
        self.cluster_history_canvas = tk.Canvas(info, width=190, height=55, bg="#07131c", highlightbackground=self.BORDER, highlightthickness=1)
        self.cluster_history_canvas.pack(fill=tk.X, pady=(3, 0))
        self.cluster_legend: list[ttk.Label] = []
        for index in range(5):
            label = ttk.Label(info, text=f"■  Cluster {index + 1} (0)", style="Body.TLabel")
            label.pack(anchor="w", pady=2)
            self.cluster_legend.append(label)

    def _build_event_panel(self, parent: ttk.Frame) -> None:
        panel = self._panel(parent, "Event Log")
        self.event_panel = panel
        self.event_panel_collapsed = False
        self.event_panel_toggle = ttk.Button(panel, text="−", width=2, style="Dash.TButton", command=self._toggle_event_panel)
        self.event_panel_toggle.place(relx=1.0, y=0, anchor="ne")
        panel.grid(row=0, column=2, sticky="nsew")
        body = self._body(panel)
        ttk.Button(body, text="Clear", style="Dash.TButton", command=self._clear_events).place(relx=1.0, y=-5, anchor="ne")
        self.event_text = tk.Text(body, height=12, bg="#07131c", fg=self.TEXT, insertbackground=self.TEXT, relief="flat", font=("Consolas", 9), wrap=tk.NONE)
        self.event_text.pack(fill=tk.BOTH, expand=True, pady=(2, 0))
        self.event_text.configure(state=tk.DISABLED)

    def _toggle_cluster_panel(self) -> None:
        self.cluster_panel_collapsed = not self.cluster_panel_collapsed
        if self.cluster_panel_collapsed:
            self._body(self.cluster_panel).pack_forget()
            self.cluster_panel_toggle.configure(text="+")
        else:
            self._body(self.cluster_panel).pack(fill=tk.BOTH, expand=True)
            self.cluster_panel_toggle.configure(text="−")
        self.bottom.update_idletasks()

    def _toggle_event_panel(self) -> None:
        self.event_panel_collapsed = not self.event_panel_collapsed
        if self.event_panel_collapsed:
            self._body(self.event_panel).pack_forget()
            self.event_panel_toggle.configure(text="+")
        else:
            self._body(self.event_panel).pack(fill=tk.BOTH, expand=True)
            self.event_panel_toggle.configure(text="−")
        self.bottom.update_idletasks()

    def _build_footer(self, root: ttk.Frame) -> None:
        footer = ttk.Frame(root, style="Dashboard.TFrame")
        self.footer = footer
        footer.grid(row=4, column=0, sticky="ew", pady=(4, 5))
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(footer, text="●  Project SIGNAL  |  v0.1.0  |  Phase 1 — Core Simulation Engine", style="Status.TLabel").pack(side=tk.RIGHT)

    def toggle_fullscreen_simulation(self) -> None:
        """Show only the particle viewport, preserving Esc/F11 to return."""
        if self.fullscreen_view:
            self.exit_fullscreen_simulation()
            return
        self.fullscreen_view = True
        for widget in (self.header, self.nav, self.left, self.right, self.bottom, self.footer, self.time_series_panel):
            widget.grid_remove()
        self.workspace.grid_configure(row=0, column=0, sticky="nsew", padx=0, pady=0)
        # The dashboard columns retain their minsize values after widgets are
        # hidden. Clear them or the old left/right panels continue consuming
        # fullscreen width even though they are no longer visible.
        for column in range(3):
            self.workspace.columnconfigure(column, weight=0, minsize=0)
        self.workspace.columnconfigure(0, weight=1, minsize=0)
        self.center.grid_configure(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.view_panel.grid_configure(row=0, column=0, sticky="nsew", pady=0)
        self.center.rowconfigure(0, weight=1)
        self.center.columnconfigure(0, weight=1)
        self.view_panel.rowconfigure(1, weight=1)
        self.view_panel.columnconfigure(0, weight=1)
        self.canvas.pack_configure(fill=tk.BOTH, expand=True)
        self.dashboard_root.rowconfigure(0, weight=1)
        self.dashboard_root.rowconfigure(2, weight=0)
        self.root.attributes("-fullscreen", True)
        self.status_var.set("Fullscreen simulation — press Esc or F11 to return")
        self.root.after(50, self._resize_fullscreen_view)

    def exit_fullscreen_simulation(self) -> None:
        """Restore the dashboard layout after fullscreen simulation mode."""
        if not self.fullscreen_view:
            return
        self.fullscreen_view = False
        self.root.attributes("-fullscreen", False)
        self.header.grid()
        self.nav.grid()
        self.left.grid()
        self.right.grid()
        self.bottom.grid()
        self.footer.grid()
        self.time_series_panel.grid()
        self.workspace.grid_configure(row=2, column=0, sticky="nsew")
        self.workspace.columnconfigure(0, weight=0, minsize=225)
        self.workspace.columnconfigure(1, weight=1, minsize=560)
        self.workspace.columnconfigure(2, weight=0, minsize=455)
        self.center.grid_configure(row=0, column=1, sticky="nsew", padx=(0, 8))
        self.view_panel.grid_configure(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.dashboard_root.rowconfigure(0, weight=0)
        self.dashboard_root.rowconfigure(2, weight=1)
        self.status_var.set("Ready")
        self.root.after(50, self._draw)

    def _resize_fullscreen_view(self) -> None:
        """Reflow the Tk canvas after the window manager enters fullscreen."""
        if not self.fullscreen_view:
            return
        self.root.update_idletasks()
        self.canvas.pack_configure(fill=tk.BOTH, expand=True)
        self._draw()

    def _build_legend(self) -> None:
        for index, label in enumerate(("Species 0", "Species 1", "Species 2")):
            tk.Label(self.legend, text=f"●  {label}", bg="#07131c", fg=self.COLORS[index], font=("Segoe UI", 9)).pack(anchor="w", padx=8, pady=3)

    def _format_parameter(self, key: str, value: float) -> str:
        if key == "particle_count":
            return f"{int(value):,}"
        return f"{value:.2f}" if key in {"core_radius_ratio", "damping"} else f"{value:.1f}"

    def _slider_changed(self, key: str, raw: str) -> None:
        value = float(raw)
        self.parameter_vars[key].set(self._format_parameter(key, value))

    def _select_metric(self, metric: str) -> None:
        self.active_metric.set(metric)
        self._draw_chart()

    def _nav_notice(self, name: str) -> None:
        if name == "Experiment":
            self._open_experiment_window()
            return
        if name == "Search":
            self._open_search_window()
            return
        if name == "Results":
            self._open_results_window()
            return
        self.status_var.set(f"{name} view is reserved for a later experiment phase")
        self._log(f"{name} tab selected")

    def _open_search_window(self) -> None:
        if getattr(self, "search_window", None) is not None and self.search_window.winfo_exists():
            self.search_window.lift()
            return
        window = self.search_window = tk.Toplevel(self.root)
        window.title("Project SIGNAL — Structure Search")
        window.geometry("560x450")
        window.configure(bg=self.BG)
        body = ttk.Frame(window, style="Dashboard.TFrame", padding=14)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Persistent Structure Search", style="Title.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(body, text="Edit the experiment size before starting the next run.", style="Status.TLabel").pack(anchor="w", pady=(0, 12))
        form = ttk.Frame(body, style="Dashboard.TFrame")
        form.pack(fill=tk.X)
        fields = (("Genomes / runs", "runs", "1,000"), ("Steps per genome", "steps", "5,000"), ("Particles / universe", "particles", "1,000"), ("Minimum detector cluster size", "minimum_cluster_size", "20"), ("Tracker persistence threshold", "tracker_persistence_threshold", "0.55"))
        self.search_vars: dict[str, tk.StringVar] = {}
        for row, (label, key, default) in enumerate(fields):
            self.search_vars[key] = tk.StringVar(value=default)
            ttk.Label(form, text=label, style="Status.TLabel").grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(form, textvariable=self.search_vars[key], width=12).grid(row=row, column=1, sticky="e", pady=5)
        actions = ttk.Frame(body, style="Dashboard.TFrame")
        actions.pack(fill=tk.X, pady=(12, 8))
        ttk.Button(actions, text="START SEARCH", style="Primary.TButton", command=self._start_search).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(actions, text="STOP SEARCH", style="Dash.TButton", command=self._stop_search).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))
        self.search_status = ttk.Label(body, text="Ready", style="Status.TLabel")
        self.search_status.pack(anchor="w", pady=(0, 6))
        self.search_progress_vars = {key: tk.StringVar(value=value) for key, value in {
            "genomes": "Genome 0 / 1,000   0.0%",
            "steps": "Simulation steps completed: 0",
            "elapsed": "Elapsed time: 00:00:00",
            "remaining": "Estimated remaining time: --",
            "candidates": "Candidates discovered: 0",
            "best": "Best structure score: 0.0000",
        }.items()}
        progress_panel = ttk.Frame(body, style="Dashboard.TFrame")
        progress_panel.pack(fill=tk.X, pady=(0, 8))
        for row, key in enumerate(("genomes", "steps", "elapsed", "remaining", "candidates", "best")):
            ttk.Label(progress_panel, textvariable=self.search_progress_vars[key], style="Status.TLabel").grid(row=row // 2, column=row % 2, sticky="w", padx=(0, 20), pady=2)
        self.search_log = tk.Text(body, height=13, bg="#07131c", fg=self.TEXT, relief="flat", font=("Consolas", 9))
        self.search_log.pack(fill=tk.BOTH, expand=True)
        window.protocol("WM_DELETE_WINDOW", window.destroy)

    def _start_search(self) -> None:
        if self.search_thread is not None and self.search_thread.is_alive():
            return
        try:
            values = {key: float(variable.get().replace(",", "").strip()) for key, variable in self.search_vars.items()}
            config = SearchConfig(runs=int(values["runs"]), steps=int(values["steps"]), particle_count=int(values["particles"]), minimum_cluster_size=int(values["minimum_cluster_size"]), tracker_persistence_threshold=values["tracker_persistence_threshold"], stop_event=Event())
            if config.runs < 1 or config.steps < 1 or config.particle_count < 1 or config.minimum_cluster_size < 1:
                raise ValueError("Runs, steps, particles, and minimum cluster size must be positive")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Invalid search settings", str(error))
            return
        self.search_stop_event = config.stop_event
        self.search_log.delete("1.0", tk.END)
        self.search_status.configure(text="Search running...")
        self._search_started_at = time.monotonic()
        self._log("Persistent structure search started")
        self.search_thread = Thread(target=self._run_search_worker, args=(config,), daemon=True)
        self.search_thread.start()

    def _run_search_worker(self, config: SearchConfig) -> None:
        def progress(line: str) -> None:
            self.root.after(0, lambda text=line: self._append_search_log(text))

        def progress_state(state: dict[str, object]) -> None:
            self.root.after(0, lambda snapshot=dict(state): self._update_search_progress(snapshot))
        try:
            results = run_search(config, progress=progress, progress_state=progress_state)
            existing = load_baseline_rows(config.result_root, config.experiment_id)
            complete = sum(1 for row in existing if row.get("status") == "COMPLETED")
            status = "Search complete" if complete >= config.runs else "Search stopped - completed results committed"
            self.root.after(0, lambda: self.search_status.configure(text=f"{status} ({complete} genomes recorded)"))
        except Exception as error:  # surface worker errors in the UI without crashing Tk
            message = str(error)
            self.root.after(0, lambda: self.search_status.configure(text=f"Search failed: {message}"))

    def _append_search_log(self, line: str) -> None:
        if not hasattr(self, "search_log") or not self.search_log.winfo_exists():
            return
        self.search_log.insert(tk.END, line + "\n")
        self.search_log.see(tk.END)
        self.search_status.configure(text=line.splitlines()[0][:80])

    def _stop_search(self) -> None:
        if self.search_stop_event is not None:
            self.search_stop_event.set()
            self.search_status.configure(text="Stop requested; committing completed genomes...")

    def _update_search_progress(self, state: dict[str, object]) -> None:
        if not hasattr(self, "search_progress_vars"):
            return
        completed = int(state.get("completed_runs", 0))
        total = int(state.get("total_runs", 1))
        percent = float(state.get("percent", 0.0))
        elapsed = float(state.get("elapsed_seconds", 0.0))
        remaining = float(state.get("estimated_remaining_seconds", 0.0))
        steps = int(state.get("simulation_steps_completed", 0))
        candidates = int(state.get("candidates_discovered", 0))
        best = float(state.get("best_structure_score", 0.0))
        self.search_progress_vars["genomes"].set(f"Genome {completed} / {total:,}   {percent:.1f}%")
        self.search_progress_vars["steps"].set(f"Simulation steps completed: {steps:,}")
        self.search_progress_vars["elapsed"].set(f"Elapsed time: {self._format_duration(elapsed)}")
        self.search_progress_vars["remaining"].set(f"Estimated remaining time: {self._format_duration(remaining) if completed else '--'}")
        self.search_progress_vars["candidates"].set(f"Candidates discovered: {candidates}")
        self.search_progress_vars["best"].set(f"Best structure score: {best:.4f}")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _open_results_window(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Project SIGNAL — Search Results")
        window.geometry("1280x560")
        window.configure(bg=self.BG)
        body = ttk.Frame(window, style="Dashboard.TFrame", padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Structure Search Results", style="Title.TLabel").pack(anchor="w", pady=(0, 10))
        actions = ttk.Frame(body, style="Dashboard.TFrame")
        actions.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(actions, text="Export PDF / Excel / CSV / JSON...", style="Dash.TButton", command=self._export_results_dialog).pack(side=tk.LEFT)
        columns = ("run", "status", "genome_hash", "seed", "clusters", "persistent", "macro", "structure_score", "lifetime", "particles", "cohesion", "identity", "shape", "dynamics", "bridge_fraction", "candidate", "candidate_path")
        tree = ttk.Treeview(body, columns=columns, show="headings")
        headings = {"run": "Genome #", "status": "Status", "genome_hash": "Genome Hash", "seed": "Seed", "clusters": "Clusters", "persistent": "Persistent", "macro": "Macro", "structure_score": "Structure Score", "lifetime": "Lifetime", "particles": "Particle Count", "cohesion": "Cohesion", "identity": "Identity", "shape": "Shape", "dynamics": "Dynamics", "bridge_fraction": "Bridge Fraction", "candidate": "Candidate", "candidate_path": "Candidate Path"}
        numeric_columns = {"run", "seed", "clusters", "persistent", "macro", "structure_score", "lifetime", "particles", "cohesion", "identity", "shape", "dynamics", "bridge_fraction"}
        for column in columns:
            tree.heading(column, text=headings[column], command=lambda key=column: self._sort_results_tree(tree, key, False))
            tree.column(column, width=100 if column not in {"genome_hash", "candidate_path"} else 240, anchor="e" if column in numeric_columns else "w")
        tree.pack(fill=tk.BOTH, expand=True)
        rows = load_baseline_rows()
        if rows:
            for row in rows:
                values = (row.get("run", ""), row.get("status", ""), row.get("genome_hash", ""), row.get("seed", ""), row.get("clusters_detected", ""), row.get("persistent_clusters", ""), row.get("macro_clusters", ""), row.get("best_structure_score", ""), row.get("best_candidate_lifetime", ""), row.get("best_candidate_particle_count", ""), row.get("best_cohesion", ""), row.get("best_identity", ""), row.get("best_shape", ""), row.get("best_dynamics", ""), row.get("best_bridge_fraction", ""), "YES" if row.get("candidate") else "NO", row.get("candidate_path", ""))
                tree.insert("", tk.END, values=values)
        else:
            tree.insert("", tk.END, values=("No baseline results yet",) + ("",) * (len(columns) - 1))
        tree.bind("<Double-1>", lambda _event: self._load_result_row(tree))

    def _sort_results_tree(self, tree: ttk.Treeview, column: str, descending: bool) -> None:
        """Sort baseline rows by a requested numeric or text metric."""
        items = [(tree.set(item, column), item) for item in tree.get_children("")]
        if column in {"status", "genome_hash", "candidate_path"}:
            items.sort(key=lambda item: item[0], reverse=descending)
        else:
            def numeric(item: tuple[str, str]) -> float:
                try:
                    return float(item[0])
                except (TypeError, ValueError):
                    return float("-inf")
            items.sort(key=numeric, reverse=descending)
        for position, (_value, item) in enumerate(items):
            tree.move(item, "", position)
        tree.heading(column, command=lambda: self._sort_results_tree(tree, column, not descending))

    def _export_results_dialog(self) -> None:
        """Export one consistent snapshot of the live simulation and search results."""
        output_dir = filedialog.askdirectory(title="Choose an export folder")
        if not output_dir:
            return
        try:
            data = collect_export_data(self.engine, self.current_clusters, self.history, self.events)
            exported = export_all(data, output_dir)
        except (OSError, RuntimeError, ValueError) as error:
            messagebox.showerror("Export failed", str(error))
            self._log(f"Export failed: {error}")
            return
        self._log(f"Results exported to {Path(output_dir).name}")
        excel_line = (
            Path(exported["xlsx"]).name
            if "xlsx" in exported
            else "Excel: not created (install openpyxl from requirements.txt)"
        )
        messagebox.showinfo(
            "Export complete",
            "Created:\n"
            f"{excel_line}\n"
            f"{Path(exported['pdf']).name}\n"
            f"{Path(exported['json']).name}\n"
            f"{len(exported['csv'])} CSV files",
        )

    def _three_view_state(self) -> dict[str, object]:
        """Return a JSON-safe copy of the live state for the WebGL viewer."""
        positions = np.asarray(self.engine.state.positions, dtype=np.float32)
        velocities = np.asarray(self.engine.state.velocities, dtype=np.float32)
        species = np.asarray(self.engine.state.species, dtype=np.int8)
        clusters = [
            {
                "cluster_id": int(cluster.cluster_id),
                "classification": cluster.classification,
                "size_class": cluster.size_class,
                "particle_count": int(cluster.particle_count),
                "age_steps": int(cluster.age_steps),
                "centroid_x": float(cluster.centroid[0]),
                "centroid_y": float(cluster.centroid[1]),
                "radius": float(cluster.radius),
                "tracker_persistence_score": float(cluster.tracker_persistence_score),
                "structure_score": float(cluster.structure_score),
                "cohesion_score": float(cluster.cohesion_score),
                "identity_score": float(cluster.identity_score),
                "shape_score": float(cluster.shape_score),
                "dynamic_score": float(cluster.dynamic_score),
                "bridge_fraction": float(cluster.bridge_fraction),
                "particle_ids": sorted(int(value) for value in cluster.particle_ids),
            }
            for cluster in self.current_clusters
        ]
        networks = [network.to_dict() for network in self.current_networks]
        return {
            "width": float(self.engine.config.width),
            "height": float(self.engine.config.height),
            "step": int(self.engine.step_count),
            "particle_count": int(self.engine.state.count),
            "interaction_radius": float(self.engine.genome.interaction_radius),
            "selected_cluster_id": self.selected_cluster_id,
            "ids": np.asarray(self.engine.state.ids, dtype=np.int64).tolist(),
            "positions": positions.tolist(),
            "velocities": velocities.tolist(),
            "species": species.tolist(),
            "clusters": clusters,
            "networks": networks,
            "network_debug": bool(self.show_structure_debug.get()),
        }

    def open_three_view(self) -> None:
        """Open or focus the GPU-accelerated Three.js particle view."""
        if self.three_viewer is None:
            self.three_viewer = ThreeViewerServer(self._three_view_state)
        url = self.three_viewer.start()
        webbrowser.open(url)
        self._log("Three.js WebGL particle view opened")
        self.status_var.set("Three.js particle view opened in your browser")

    def close(self) -> None:
        """Stop local services before closing the desktop dashboard."""
        if self.three_viewer is not None:
            self.three_viewer.stop()
        self.root.destroy()

    def _load_result_row(self, tree: ttk.Treeview) -> None:
        selection = tree.selection()
        if not selection:
            return
        values = tree.item(selection[0], "values")
        path = values[-1] if values else ""
        if not path:
            return
        try:
            self.engine, _metadata = load_structure_snapshot(path)
            self._reset_tracker()
            self.seed_var.set(str(self.engine.config.seed))
            self._sync_parameters_from_genome()
            self._sync_matrix_from_genome()
            self._refresh_metrics()
            self._draw()
            self._log(f"Loaded structure replay from {Path(path).name}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            messagebox.showerror("Replay failed", str(error))

    def _log(self, message: str) -> None:
        timestamp = f"[{self.engine.step_count:06d}]"
        self.events.append(f"{timestamp}  {message}")
        if hasattr(self, "event_text"):
            self.event_text.configure(state=tk.NORMAL)
            self.event_text.delete("1.0", tk.END)
            self.event_text.insert(tk.END, "\n".join(self.events) + "\n")
            self.event_text.see(tk.END)
            self.event_text.configure(state=tk.DISABLED)

    def _clear_events(self) -> None:
        self.events.clear()
        self._log("Event log cleared")

    def _advance(self, steps: int) -> None:
        self.engine.step(steps)
        self._refresh_metrics()
        self._draw()

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._log("Simulation started")
            self.status_var.set("Simulation running...")
            self._tick()

    def _tick(self) -> None:
        if not self.running:
            return
        self._advance(1)
        self.root.after(16, self._tick)

    def pause(self) -> None:
        self.running = False
        self._log("Simulation paused")
        self.status_var.set("Ready")

    def reset(self) -> None:
        self.pause()
        self.apply_seed(log_event=False)
        self._log("Universe reset")

    def apply_seed(self, log_event: bool = True) -> None:
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Invalid seed", "Seed must be an integer.")
            return
        self.engine.reset(seed)
        self.trail_points.clear()
        self._reset_tracker()
        self._refresh_metrics()
        self._draw()
        if log_event:
            self._log(f"Seed changed to {seed}")

    def randomize_matrix(self) -> None:
        self.engine.genome = Genome.random(3, seed=self.engine.config.seed + self.engine.step_count + 1)
        self._sync_matrix_from_genome()
        self._sync_spatial_hash()
        self._reset_tracker()
        self._log("Interaction matrix randomized")
        self._draw()

    def randomize_genome(self) -> None:
        self.engine.genome = Genome.random(3, seed=self.engine.config.seed + self.engine.step_count + 1)
        self._sync_matrix_from_genome()
        self._sync_parameters_from_genome()
        self._sync_spatial_hash()
        self.engine.reset()
        self.trail_points.clear()
        self._reset_tracker()
        self._log("Genome randomized")
        self._draw()

    def _sync_spatial_hash(self) -> None:
        self.engine.spatial_hash.cell_size = self.engine.genome.interaction_radius
        self.engine.spatial_hash.columns = max(1, int(np.ceil(self.engine.config.width / self.engine.genome.interaction_radius)))
        self.engine.spatial_hash.rows = max(1, int(np.ceil(self.engine.config.height / self.engine.genome.interaction_radius)))

    def _reset_tracker(self) -> None:
        self.cluster_tracker = ClusterTracker(self.engine.config.width, self.engine.config.height, self.engine.genome.interaction_radius, min_cluster_size=5)
        self.network_tracker = NetworkTracker(self.engine.config.width, self.engine.config.height, self.engine.genome.interaction_radius)
        self.current_clusters = []
        self.current_networks = []
        self.saved_network_ids = set()
        self.selected_cluster_id = None
        self.next_cluster_observation_step = 0
        self.last_metrics_step = -1

    def _sync_matrix_from_genome(self) -> None:
        for row in range(3):
            for column in range(3):
                self.matrix_vars[row][column].set(f"{self.engine.genome.interaction_matrix[row, column]:.2f}")

    def _sync_parameters_from_genome(self) -> None:
        for key in ("interaction_radius", "core_radius_ratio", "core_repulsion", "force_gain", "damping", "dt"):
            self.parameter_vars[key].set(self._format_parameter(key, getattr(self.engine.genome, key)))

    def apply_matrix(self) -> None:
        try:
            matrix = np.array([[float(variable.get()) for variable in row] for row in self.matrix_vars], dtype=np.float64)
            self.engine.genome = Genome(3, matrix, self.engine.genome.interaction_radius, self.engine.genome.core_radius_ratio, self.engine.genome.core_repulsion, self.engine.genome.force_gain, self.engine.genome.damping, self.engine.genome.dt)
            self._sync_spatial_hash()
            self._log("Interaction matrix applied")
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Invalid matrix", str(error))

    def _parameterized_genome(self) -> Genome:
        def number(key: str) -> float:
            return float(self.parameter_vars[key].get().replace(",", ""))
        return Genome(self.engine.genome.species_count, self.engine.genome.interaction_matrix.copy(), number("interaction_radius"), number("core_radius_ratio"), number("core_repulsion"), number("force_gain"), number("damping"), number("dt"))

    def save_genome(self) -> None:
        try:
            self.engine.genome = self._parameterized_genome()
        except ValueError as error:
            messagebox.showerror("Invalid parameters", str(error))
            return
        path = filedialog.asksaveasfilename(title="Save Genome", defaultextension=".json", filetypes=[("Genome JSON", "*.json")])
        if path:
            self.engine.genome.save(path)
            self._log(f"Genome saved: {Path(path).name}")

    def load_genome(self) -> None:
        path = filedialog.askopenfilename(title="Load Genome", filetypes=[("Genome JSON", "*.json")])
        if not path:
            return
        try:
            genome = Genome.load(path)
            if genome.species_count != 3:
                raise ValueError("Phase 1 viewer requires exactly 3 species")
            self.engine.genome = genome
            self._sync_matrix_from_genome()
            self._sync_parameters_from_genome()
            self._sync_spatial_hash()
            self._log(f"Genome loaded: {Path(path).name}")
            self._draw()
        except (OSError, ValueError, json.JSONDecodeError) as error:
            messagebox.showerror("Load failed", str(error))

    def _refresh_metrics(self) -> None:
        if self.last_metrics_step == self.engine.step_count:
            return
        if self.engine.step_count >= self.next_cluster_observation_step:
            tracking = self.cluster_tracker.update(self.engine.state, self.engine.step_count)
            self.current_clusters = tracking.clusters
            self.current_networks = self.network_tracker.update(self.engine.state, tracking.clusters, self.engine.step_count)
            self.next_cluster_observation_step = self.engine.step_count + 10
            self._handle_cluster_events(tracking.events)
            self._handle_network_events(self.network_tracker.last_events)
            for network in self.current_networks:
                if network.is_candidate and network.network_id not in self.saved_network_ids:
                    try:
                        destination = save_network_snapshot(self.engine, network)
                        self.saved_network_ids.add(network.network_id)
                        self._log(f"Network candidate saved: {destination.name}")
                    except OSError as error:
                        self._log(f"Network snapshot failed: {error}")
            if self.selected_cluster_id not in {cluster.cluster_id for cluster in self.current_clusters}:
                self.selected_cluster_id = None
        speeds = np.linalg.norm(self.engine.state.velocities, axis=1)
        mean_speed = float(np.mean(speeds))
        variance = float(np.var(speeds))
        vectors = self.engine.state.velocities
        norms = np.linalg.norm(vectors, axis=1)
        coherence = float(np.linalg.norm(np.sum(vectors, axis=0)) / max(np.sum(norms), 1e-12))
        grid = np.zeros((20, 20), dtype=np.int32)
        coords = (self.engine.state.positions / np.array([self.engine.config.width, self.engine.config.height]) * 20).astype(int) % 20
        np.add.at(grid, (coords[:, 1], coords[:, 0]), 1)
        probabilities = grid.ravel() / max(len(coords), 1)
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        cluster_count = len(self.current_clusters)
        largest_cluster = max((cluster.particle_count for cluster in self.current_clusters), default=0)
        metrics = {"avg_speed": mean_speed, "speed_variance": variance, "cluster_count": cluster_count, "largest_cluster": largest_cluster, "entropy": entropy, "coherence": coherence}
        formats = {"avg_speed": "{:.2f}", "speed_variance": "{:.2f}", "cluster_count": "{:.0f}", "largest_cluster": "{:.0f}", "entropy": "{:.2f}", "coherence": "{:.2f}"}
        for key, value in metrics.items():
            self.stat_labels[key].configure(text=formats[key].format(value))
        self.history["Average Speed"].append(mean_speed)
        self.history["Cluster Count"].append(float(cluster_count))
        self.history["Entropy"].append(entropy)
        self.history["Coherence"].append(coherence)
        proportions = np.bincount(self.engine.state.species, minlength=3) / max(len(self.engine.state.species), 1)
        self._draw_pie(proportions)
        self.cluster_info.configure(text=f"Clusters: {cluster_count}\nLargest: {largest_cluster}\nMean Size: {len(self.engine.state.ids) / max(cluster_count, 1):.1f}\nTracker Persistence: {min(0.99, 0.45 + self.engine.step_count / 3000):.2f}")
        for index, label in enumerate(self.cluster_legend):
            size = max(0, int(largest_cluster * (0.66 ** index)))
            label.configure(text=f"■  Cluster {index + 1} ({size})")
        self.last_metrics_step = self.engine.step_count
        self._draw_chart()
        self._draw_cluster_preview()
        self._update_cluster_inspector()

    def _handle_cluster_events(self, events: list[object]) -> None:
        """Log meaningful structure events and snapshot persistent transitions."""
        for event in events:
            if not hasattr(event, "kind"):
                continue
            if event.kind == "formed" and event.particle_count < 20:
                continue
            if event.kind not in {"formed", "reached", "merged", "split", "dissolved"}:
                continue
            self._log(event.message)
            if event.kind == "reached" and lifetime_class(event.particle_count) in {"PERSISTENT", "LONG_LIVED"}:
                cluster = next((item for item in self.current_clusters if item.cluster_id == event.cluster_id), None)
                if cluster is not None:
                    try:
                        destination = save_structure_snapshot(self.engine, cluster)
                        self._log(f"Structure snapshot saved: {destination.name}")
                    except OSError as error:
                        self._log(f"Structure snapshot failed: {error}")

    def _handle_network_events(self, events: list[object]) -> None:
        """Log only persistent network topology changes."""
        for event in events:
            if getattr(event, "kind", "") in {"node_persistent", "edge_persistent", "edge_dissolved", "topology_changed", "became_persistent", "dissolved"}:
                self._log(event.message)

    def _coarse_clusters(self) -> tuple[int, int]:
        coords = (self.engine.state.positions / np.array([self.engine.config.width, self.engine.config.height]) * 12).astype(int) % 12
        counts = np.zeros((12, 12), dtype=int)
        np.add.at(counts, (coords[:, 1], coords[:, 0]), 1)
        occupied = counts[counts > 0]
        if not len(occupied):
            return 0, 0
        clusters = max(1, int(round(np.count_nonzero(counts > max(2, len(self.engine.state.ids) // 110)))))
        return clusters, int(np.max(occupied))

    def _draw_pie(self, proportions: np.ndarray) -> None:
        if not hasattr(self, "pie_canvas"):
            return
        self.pie_canvas.delete("all")
        start = 0.0
        for index, proportion in enumerate(proportions[:3]):
            extent = float(proportion) * 360.0
            self.pie_canvas.create_arc(6, 6, 96, 96, start=start, extent=extent, fill=self.COLORS[index], outline=self.PANEL)
            self.distribution_labels[index].configure(text=f"■  Species {index}   {proportion * 100:.1f}%", foreground=self.COLORS[index])
            start += extent

    def _draw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / self.engine.config.width, height / self.engine.config.height)
        offset_x = (width - self.engine.config.width * scale) / 2
        offset_y = (height - self.engine.config.height * scale) / 2
        if self.show_radius.get():
            radius = self.engine.genome.interaction_radius * scale
            self.canvas.create_oval(width / 2 - radius, height / 2 - radius, width / 2 + radius, height / 2 + radius, outline="#345368")
        if len(self.trail_points) > 1:
            for previous, current in zip(self.trail_points, list(self.trail_points)[1:]):
                x1, y1 = offset_x + previous[0] * scale, offset_y + previous[1] * scale
                x2, y2 = offset_x + current[0] * scale, offset_y + current[1] * scale
                self.canvas.create_line(x1, y1, x2, y2, fill="#0b2936", width=1)
        self.trail_points.append(np.mean(self.engine.state.positions, axis=0))
        radius = 2.2 if self.engine.state.count >= 700 else 3.5
        for position, velocity, species in zip(self.engine.state.positions, self.engine.state.velocities, self.engine.state.species):
            x = offset_x + float(position[0]) * scale
            y = offset_y + float(position[1]) * scale
            color = self.COLORS[int(species)] if self.color_species.get() else "#67b9e8"
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline="")
            if self.show_vectors.get():
                self.canvas.create_line(x, y, x + velocity[0] * 15, y + velocity[1] * 15, fill=color, width=1)
        if self.show_clusters.get():
            self._draw_cluster_outlines(self.canvas, width, height, scale, offset_x, offset_y)
        self.canvas.create_text(10, height - 34, anchor="sw", text=f"World: {int(self.engine.config.width)} × {int(self.engine.config.height)} (toroidal)", fill=self.TEXT, font=("Segoe UI", 9))
        self.canvas.create_text(10, height - 16, anchor="sw", text=f"Interaction Radius: {self.engine.genome.interaction_radius:.0f}", fill=self.TEXT, font=("Segoe UI", 9))
        self.canvas.create_window(width - 115, height - 76, window=self.legend, anchor="nw")
        self.view_title.configure(text=f"Simulation View   |   {self.engine.state.count:,} particles   |   Step: {self.engine.step_count:,}   |   FPS: 60")

    def _draw_cluster_outlines(self, canvas: tk.Canvas, width: float, height: float, scale: float, offset_x: float, offset_y: float) -> None:
        for cluster in sorted(self.current_clusters, key=lambda item: item.particle_count, reverse=True):
            if cluster.particle_count < 20 and cluster.cluster_id != self.selected_cluster_id:
                continue
            center_x = offset_x + float(cluster.centroid[0]) * scale
            center_y = offset_y + float(cluster.centroid[1]) * scale
            radius = max(cluster.radius, self.engine.genome.interaction_radius * 0.25) * scale + 5
            if cluster.cluster_id == self.selected_cluster_id:
                color, line_width = "#ffffff", 2
            elif cluster.classification in {"PERSISTENT", "LONG_LIVED"}:
                color, line_width = "#4fd27d", 2
            elif cluster.classification == "SHORT_LIVED":
                color, line_width = "#ffad3d", 1
            else:
                color, line_width = "#3b9cff", 1
            canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline=color, width=line_width)
            canvas.create_text(center_x, center_y - radius - 6, text=f"C{cluster.cluster_id}\n n={cluster.particle_count}  age={cluster.age_steps}", fill=color, font=("Segoe UI", 8, "bold"), justify=tk.CENTER)
            if self.show_structure_debug.get() and cluster.cluster_id == self.selected_cluster_id:
                self._draw_debug_cluster(canvas, cluster, scale, offset_x, offset_y)

    def _draw_debug_cluster(self, canvas: tk.Canvas, cluster: ClusterObservation, scale: float, offset_x: float, offset_y: float) -> None:
        """Overlay centroid, local neighbor graph, and tracked centroid trajectory."""
        id_to_index = {int(value): index for index, value in enumerate(self.engine.state.ids)}
        member_indices = [id_to_index[value] for value in cluster.particle_ids if value in id_to_index]
        positions = self.engine.state.positions
        radius = self.engine.genome.interaction_radius
        size = np.array([self.engine.config.width, self.engine.config.height], dtype=np.float64)
        for local_index, index in enumerate(member_indices):
            for other in member_indices[local_index + 1:]:
                delta = (positions[other] - positions[index] + size / 2.0) % size - size / 2.0
                if float(np.linalg.norm(delta)) <= radius:
                    x1 = offset_x + positions[index, 0] * scale
                    y1 = offset_y + positions[index, 1] * scale
                    x2 = offset_x + positions[other, 0] * scale
                    y2 = offset_y + positions[other, 1] * scale
                    canvas.create_line(x1, y1, x2, y2, fill="#24576a", width=1)
        cx = offset_x + cluster.centroid[0] * scale
        cy = offset_y + cluster.centroid[1] * scale
        canvas.create_line(cx - 7, cy, cx + 7, cy, fill="#ffffff", width=1)
        canvas.create_line(cx, cy - 7, cx, cy + 7, fill="#ffffff", width=1)
        trajectory = [(item.get("centroid_x"), item.get("centroid_y")) for item in cluster.history if item.get("centroid_x") is not None]
        if len(trajectory) > 1:
            points: list[float] = []
            for x, y in trajectory:
                points.extend((offset_x + float(x) * scale, offset_y + float(y) * scale))
            canvas.create_line(*points, fill="#ffffff", width=1, dash=(3, 3))

    def _select_cluster_from_canvas(self, event: tk.Event) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / self.engine.config.width, height / self.engine.config.height)
        offset_x = (width - self.engine.config.width * scale) / 2
        offset_y = (height - self.engine.config.height * scale) / 2
        world = np.array([(event.x - offset_x) / scale, (event.y - offset_y) / scale])
        self._select_cluster_at(world)

    def _select_cluster_from_preview(self, event: tk.Event) -> None:
        width = max(1, self.cluster_canvas.winfo_width())
        height = max(1, self.cluster_canvas.winfo_height())
        world = np.array([event.x / width * self.engine.config.width, event.y / height * self.engine.config.height])
        self._select_cluster_at(world)

    def _select_cluster_at(self, world_position: np.ndarray) -> None:
        best: tuple[float, int] | None = None
        for cluster in self.current_clusters:
            distance = self._toroidal_point_distance(world_position, cluster.centroid)
            threshold = max(cluster.radius * 1.6, self.engine.genome.interaction_radius * 0.35)
            if distance <= threshold and (best is None or distance < best[0]):
                best = (distance, cluster.cluster_id)
        self.selected_cluster_id = best[1] if best else None
        if self.selected_cluster_id is not None:
            self._log(f"Cluster C{self.selected_cluster_id} selected")
        self._update_cluster_inspector()
        self._draw()
        self._draw_cluster_preview()

    def _toroidal_point_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        size = np.array([self.engine.config.width, self.engine.config.height], dtype=np.float64)
        delta = (np.asarray(b) - np.asarray(a) + size / 2.0) % size - size / 2.0
        return float(np.linalg.norm(delta))

    def _draw_cluster_preview(self) -> None:
        if not hasattr(self, "cluster_canvas"):
            return
        canvas = self.cluster_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        for position, species in zip(self.engine.state.positions, self.engine.state.species):
            canvas.create_oval(position[0] / self.engine.config.width * width - 1.5, position[1] / self.engine.config.height * height - 1.5, position[0] / self.engine.config.width * width + 1.5, position[1] / self.engine.config.height * height + 1.5, fill=self.COLORS[int(species)], outline="")
        if self.show_clusters.get():
            self._draw_cluster_outlines(canvas, width, height, min(width / self.engine.config.width, height / self.engine.config.height), 0, 0)
        selected = next((item for item in self.current_clusters if item.cluster_id == self.selected_cluster_id), None)
        if selected is not None and selected.particle_count <= 80 and self.show_structure_debug.get():
            id_to_index = {int(value): index for index, value in enumerate(self.engine.state.ids)}
            for particle_id in sorted(selected.particle_ids):
                index = id_to_index.get(particle_id)
                if index is not None:
                    x = self.engine.state.positions[index, 0] / self.engine.config.width * width
                    y = self.engine.state.positions[index, 1] / self.engine.config.height * height
                    canvas.create_text(x + 5, y, text=str(particle_id), fill="#ffffff", anchor="w", font=("Consolas", 7))
        self._draw_cluster_history()

    def _update_cluster_inspector(self) -> None:
        if not hasattr(self, "cluster_detail"):
            return
        cluster = next((item for item in self.current_clusters if item.cluster_id == self.selected_cluster_id), None)
        if cluster is None:
            self.cluster_detail.configure(text="Select a cluster to inspect")
            self._draw_cluster_history()
            return
        composition = ", ".join(f"S{i}: {value * 100:.0f}%" for i, value in enumerate(cluster.species_distribution[:3]))
        self.cluster_detail.configure(text=f"C{cluster.cluster_id}  {cluster.classification}  ({cluster.size_class})\nParticles: {cluster.particle_count}   Age: {cluster.age_steps}\nTracker Persistence: {cluster.tracker_persistence_score:.2f}\nStructure Score: {cluster.structure_score:.2f}   Rg: {cluster.radius_of_gyration:.1f}   Diameter: {cluster.diameter:.1f}\nCohesion: {cluster.cohesion_score:.2f}   Identity Score: {cluster.identity_score:.2f}\nShape Score: {cluster.shape_score:.2f}   Dynamics: {cluster.dynamic_score:.2f}   Lifetime Score: {cluster.lifetime_score:.2f}\nVelocity: ({cluster.mean_velocity[0]:.2f}, {cluster.mean_velocity[1]:.2f})\nSpecies: {composition}\nBridges: {cluster.bridge_fraction:.2f}   Degree: {cluster.average_degree:.1f}")
        self._draw_cluster_history()

    def _draw_cluster_history(self) -> None:
        if not hasattr(self, "cluster_history_canvas"):
            return
        canvas = self.cluster_history_canvas
        canvas.delete("all")
        cluster = next((item for item in self.current_clusters if item.cluster_id == self.selected_cluster_id), None)
        if cluster is None or len(cluster.history) < 2:
            return
        values = [item["particle_count"] for item in cluster.history]
        high = max(values) or 1.0
        points: list[float] = []
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        for index, value in enumerate(values):
            points.extend((10 + index / max(len(values) - 1, 1) * (width - 20), height - 8 - value / high * (height - 16)))
        canvas.create_line(*points, fill="#3c98ff", width=2, smooth=True)

    def _draw_chart(self) -> None:
        if not hasattr(self, "chart_canvas"):
            return
        canvas = self.chart_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        for x in np.linspace(0, width, 5):
            canvas.create_line(x, 0, x, height, fill="#1e3441")
        for y in np.linspace(0, height, 4):
            canvas.create_line(0, y, width, y, fill="#1e3441")
        values = list(self.history[self.active_metric.get()])
        if len(values) < 2:
            return
        low, high = min(values), max(values)
        if math.isclose(low, high):
            high = low + 1.0
        points = []
        for index, value in enumerate(values):
            x = index / max(len(values) - 1, 1) * (width - 20) + 10
            y = height - 10 - (value - low) / (high - low) * (height - 20)
            points.extend((x, y))
        canvas.create_line(*points, fill="#3c98ff", width=2, smooth=True)
        canvas.create_text(8, 8, anchor="nw", text=f"{high:.2f}", fill=self.MUTED, font=("Segoe UI", 8))
        canvas.create_text(8, height - 8, anchor="sw", text=f"{low:.2f}", fill=self.MUTED, font=("Segoe UI", 8))

    def run(self) -> None:
        self.root.mainloop()
