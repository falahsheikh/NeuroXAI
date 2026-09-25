"""Widgets of the Slice Viewer: styles, menus, toolbar, control panel, views and status bar."""

import tkinter as tk
from functools import partial
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..annotations import DRAWING, MEASUREMENT
from ..volume import AXIAL, CORONAL, SAGITTAL

VIEW_COLORS = {"axial": "#00ffff", "coronal": "#00ff6e", "sagittal": "#ff0000"}
COLUMNS = {"left": ("axial", "sagittal"), "right": ("coronal",)}  # views in each column, from top to bottom
CONTROL_WIDTH = 350
DRAW_COLORS = ("yellow", "red", "lime", "cyan", "magenta", "orange", "white")
MODE_LABELS = {"measure": "Measure", "area": "Area", "draw": "Draw", "zoom_select": "Zoom Select", "pan": "Pan"}

BASE02, BASE2, BASE3, BLUE = "#073642", "#eee8d5", "#fdf6e3", "#000080"


class LayoutMixin:
    """Builds the window. The other parts of the viewer use the widgets that it stores on self."""

    def _build_ui(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self._configure_styles()
        self._build_menu()
        self._build_toolbar()
        self._build_status_bar()  # packed before the views, so that it keeps its space
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.main_pane = ttk.PanedWindow(container, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        self.main_pane.add(self._build_control_panel(), weight=0)
        self.main_pane.add(self._build_view_panel(), weight=1)

    def _configure_styles(self):
        style = self.style
        self.root.configure(bg=BASE3)
        style.configure(".", background=BASE3, foreground=BASE02, font=("Segoe UI", 9), borderwidth=0)
        style.configure("TFrame", background=BASE3)
        style.configure("TLabel", background=BASE3, foreground=BASE02)
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground=BASE02)
        style.configure("Info.TLabel", foreground="#586e75", font=("Segoe UI", 8))
        style.configure("TNotebook", background=BASE3, borderwidth=1)
        style.configure("TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 9), background=BASE2, foreground=BASE02)
        style.map(
            "TNotebook.Tab", background=[("selected", BLUE), ("active", BASE3)], foreground=[("selected", "white")]
        )
        style.configure(
            "TButton",
            padding=5,
            font=("Segoe UI", 9),
            background=BASE2,
            foreground=BASE02,
            borderwidth=1,
            relief="raised",
        )
        style.map("TButton", background=[("active", BASE3), ("pressed", BASE3)], relief=[("pressed", "sunken")])
        style.configure("Toolbutton.TButton", padding=0, relief="flat", background="black", foreground="white")
        style.map("Toolbutton.TButton", background=[("active", "#333333")])
        style.configure(
            "TLabelframe", relief="solid", borderwidth=1, bordercolor="#93a1a1", padding=10, background=BASE2
        )
        style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"), background=BASE2, foreground=BASE02)
        style.configure("TCheckbutton", indicatorrelief="flat", background=BASE3, foreground=BASE02)
        style.configure("TScale", troughcolor=BASE2, background=BLUE)
        style.configure("Treeview", background=BASE2, foreground=BASE02, fieldbackground=BASE2, rowheight=25)
        style.map("Treeview", background=[("selected", BLUE)], foreground=[("selected", "white")])
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background=BASE3)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Volume...", command=self.open_volume, accelerator="Ctrl+O")
        file_menu.add_command(label="Edit Patient Info...", command=self.edit_patient_info)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session...", command=self.save_session, accelerator="Ctrl+Shift+S")
        file_menu.add_command(label="Load Session...", command=self.load_session, accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save Report...", command=self.save_report, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Overlays", variable=self.show_overlays)
        view_menu.add_checkbutton(label="Axis Scales in a Maximized View", variable=self.show_axis_scales)
        view_menu.add_separator()
        view_menu.add_command(label="Reset Layout", command=self.reset_layout, accelerator="Ctrl+R")
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom, accelerator="Ctrl+0")

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_checkbutton(label="Crosshairs", variable=self.crosshair_enabled)
        tools_menu.add_separator()
        for name, label in MODE_LABELS.items():
            tools_menu.add_checkbutton(label=label, variable=self.modes[name])
        tools_menu.add_separator()
        tools_menu.add_command(label="Clear All Measurements", command=lambda: self.clear_annotations(MEASUREMENT))
        tools_menu.add_command(label="Clear All Drawings", command=lambda: self.clear_annotations(DRAWING))
        tools_menu.add_separator()
        tools_menu.add_command(label="Launch Analysis Tool", command=self.launch_analysis_tool)

        for sequence, command in {
            "<Control-o>": self.open_volume,
            "<Control-s>": self.save_report,
            "<Control-S>": self.save_session,  # Ctrl+Shift+S
            "<Control-O>": self.load_session,  # Ctrl+Shift+O
            "<Control-r>": self.reset_layout,
            "<Control-0>": self.reset_zoom,
            "<Control-z>": self.undo,
            "<Control-y>": self.redo,
            "<Escape>": self.cancel_pending,
        }.items():
            self.root.bind(sequence, lambda _event, command=command: command())

    def _build_toolbar(self):
        bar = ttk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(2, 5))

        def separator():
            ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        ttk.Button(bar, text="Load Volume", command=self.open_volume).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Patient Info", command=self.edit_patient_info).pack(side=tk.LEFT, padx=2)
        separator()
        ttk.Button(bar, text="Save Session", command=self.save_session).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Load Session", command=self.load_session).pack(side=tk.LEFT, padx=2)
        separator()
        self.undo_button = ttk.Button(bar, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT, padx=2)
        self.redo_button = ttk.Button(bar, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT, padx=2)
        separator()
        ttk.Button(bar, text="Reset Layout", command=self.reset_layout).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        separator()
        ttk.Checkbutton(bar, text="Crosshairs", variable=self.crosshair_enabled).pack(side=tk.LEFT, padx=2)
        for name, label in MODE_LABELS.items():
            ttk.Checkbutton(bar, text=label, variable=self.modes[name]).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(bar, text="Overlays", variable=self.show_overlays).pack(side=tk.LEFT, padx=2)
        separator()
        ttk.Button(bar, text="Launch Analysis Tool", command=self.launch_analysis_tool).pack(side=tk.LEFT, padx=2)

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(5, 2))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = ttk.Label(bar, text="Ready", style="Info.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.view_label = ttk.Label(bar, text="View: ---", style="Info.TLabel", width=15)
        self.view_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.position_label = ttk.Label(bar, text="Voxel: ---", style="Info.TLabel", width=28)
        self.position_label.pack(side=tk.LEFT, padx=5)
        self.intensity_label = ttk.Label(bar, text="Intensity: ---", style="Info.TLabel", width=20)
        self.intensity_label.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=150)  # packed only while busy

    def _build_control_panel(self):
        panel = ttk.Frame(self.main_pane, width=CONTROL_WIDTH)
        panel.pack_propagate(False)
        header = ttk.Frame(panel)
        header.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(header, text="Controls & Analysis", style="Title.TLabel").pack()
        patient = ttk.Frame(header)
        patient.pack(fill=tk.X, pady=(5, 0))
        self.patient_label = ttk.Label(patient, text="Patient: N/A (ID: N/A)", style="Info.TLabel", wraplength=280)
        self.patient_label.pack(side=tk.LEFT)
        ttk.Button(patient, text="Edit", command=self.edit_patient_info, width=6).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._build_navigation_tab(notebook)
        self._build_annotation_tab(notebook)
        return panel

    def _build_navigation_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Navigation")

        levels = ttk.LabelFrame(tab, text="Window & Level")
        levels.pack(fill=tk.X, pady=10, padx=5)
        for row, (name, command) in enumerate((("Window", self._on_window_slider), ("Level", self._on_level_slider))):
            ttk.Label(levels, text=f"{name}:", style="Info.TLabel").grid(row=2 * row, column=0, sticky=tk.W, padx=5)
            value = ttk.Label(levels, text="---", style="Info.TLabel")
            value.grid(row=2 * row, column=1, sticky=tk.E, padx=5)
            slider = ttk.Scale(levels, from_=0, to=1, command=command)
            slider.grid(row=2 * row + 1, column=0, columnspan=2, sticky=tk.EW, padx=5)
            if name == "Window":
                self.window_slider, self.window_value = slider, value
            else:
                self.level_slider, self.level_value = slider, value
        levels.columnconfigure(0, weight=1)

        positions = ttk.LabelFrame(tab, text="Slice Position")
        positions.pack(fill=tk.X, pady=10, padx=5)
        self.slice_sliders, self.slice_values = {}, {}
        rows = (
            ("Axial (Z)", "axial", AXIAL),
            ("Coronal (Y)", "coronal", CORONAL),
            ("Sagittal (X)", "sagittal", SAGITTAL),
        )
        for row, (label, view, plane) in enumerate(rows):
            ttk.Label(positions, text=label, foreground=VIEW_COLORS[view], font=("Segoe UI", 9, "bold")).grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=(5, 0)
            )
            slider = ttk.Scale(positions, from_=0, to=1, command=partial(self._on_slice_slider, plane))
            slider.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            value = ttk.Label(positions, text="0/0", style="Info.TLabel", width=8)
            value.grid(row=row, column=2, padx=5, pady=2)
            self.slice_sliders[plane], self.slice_values[plane] = slider, value
        positions.columnconfigure(1, weight=1)

    def _build_annotation_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Annotations")

        tool = ttk.LabelFrame(tab, text="Drawing Tool")
        tool.pack(fill=tk.X, pady=10, padx=5)
        color_row = ttk.Frame(tool)
        color_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(color_row, text="Color:").pack(side=tk.LEFT)
        ttk.Combobox(color_row, textvariable=self.draw_color, values=DRAW_COLORS, state="readonly", width=10).pack(
            side=tk.RIGHT
        )
        size_row = ttk.Frame(tool)
        size_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_row, text="Brush Size (px):").pack(side=tk.LEFT)
        ttk.Spinbox(size_row, from_=1, to=20, increment=1, textvariable=self.draw_size, width=10).pack(side=tk.RIGHT)
        opacity_row = ttk.Frame(tool)
        opacity_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(opacity_row, text="Opacity:").pack(side=tk.LEFT)
        ttk.Scale(opacity_row, from_=0.1, to=1.0, variable=self.draw_opacity, length=100).pack(side=tk.RIGHT)
        opacity_value = ttk.Label(opacity_row, text=f"{self.draw_opacity.get():.1f}")
        opacity_value.pack(side=tk.RIGHT, padx=(5, 10))
        self.draw_opacity.trace_add("write", lambda *_: opacity_value.config(text=f"{self.draw_opacity.get():.1f}"))

        lists = ttk.Notebook(tab)
        lists.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.trees = {}
        for kind, title in ((DRAWING, "Drawings"), (MEASUREMENT, "Measurements")):
            page = ttk.Frame(lists)
            lists.add(page, text=title)
            buttons = ttk.Frame(page)
            buttons.pack(fill=tk.X, pady=2)
            for text, command in (
                ("Comment", self.comment_annotation),
                ("Delete", self.delete_annotation),
                ("Clear All", self.clear_annotations),
            ):
                ttk.Button(buttons, text=text, style="Toolbutton.TButton", command=partial(command, kind)).pack(
                    side=tk.LEFT, padx=2, expand=True
                )
            tree = ttk.Treeview(page, columns=("info",), show="headings", selectmode="browse", height=8)
            tree.heading("info", text=f"{title} (select one to show it)")
            tree.column("info", width=250)
            scrollbar = ttk.Scrollbar(page, orient=tk.VERTICAL, command=tree.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.configure(yscrollcommand=scrollbar.set)
            tree.pack(fill=tk.BOTH, expand=True)
            tree.bind("<<TreeviewSelect>>", partial(self._on_tree_select, kind))
            self.trees[kind] = tree
        ttk.Label(
            tab,
            text="Comment adds or changes the comment of the selected or the newest item.",
            style="Info.TLabel",
            wraplength=320,
        ).pack(fill=tk.X, padx=5, pady=5)

    def _build_view_panel(self):
        panel = ttk.Frame(self.main_pane)
        self.columns = ttk.PanedWindow(panel, orient=tk.HORIZONTAL)
        self.columns.pack(fill=tk.BOTH, expand=True)
        self.column_panes = {}
        for side in COLUMNS:
            self.column_panes[side] = ttk.PanedWindow(self.columns, orient=tk.VERTICAL)
            self.columns.add(self.column_panes[side], weight=1)
        self.views = {}
        for side, names in COLUMNS.items():
            for name in names:
                self._build_view(name, self.column_panes[side])
        return panel

    def _build_view(self, name, column):
        frame = ttk.Frame(column)
        column.add(frame, weight=1)
        figure = Figure(figsize=(6, 6), facecolor="black")
        ax = figure.add_subplot(facecolor="black")
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        overlay = tk.Frame(frame, bg="black", bd=0, highlightthickness=0)
        overlay.place(relx=1.0, rely=0, anchor=tk.NE, x=-5, y=5)
        button = ttk.Button(
            overlay, text="❐", width=2, style="Toolbutton.TButton", command=partial(self.toggle_maximize, name)
        )
        button.pack()
        for event, handler in (
            ("scroll_event", self._on_scroll),
            ("button_press_event", self._on_press),
            ("button_release_event", self._on_release),
            ("motion_notify_event", self._on_motion),
            ("figure_enter_event", self._on_view_enter),
        ):
            canvas.mpl_connect(event, partial(handler, view=name))
        canvas.mpl_connect("figure_leave_event", self._on_view_leave)
        self.views[name] = {
            "frame": frame,
            "figure": figure,
            "ax": ax,
            "canvas": canvas,
            "button": button,
            "column": column,
        }

    # Pane sizes and maximized views

    def reset_layout(self):
        """Restore a maximized view and set the default pane sizes."""
        if self.maximized_view:
            self.toggle_maximize(self.maximized_view)
        self.root.update_idletasks()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        if width > CONTROL_WIDTH and height > 1:
            self.main_pane.sashpos(0, CONTROL_WIDTH)
            self.columns.sashpos(0, (width - CONTROL_WIDTH) // 2)
            self.column_panes["left"].sashpos(0, (height - 50) // 2)

    def toggle_maximize(self, name):
        if self.maximized_view == name:
            self._restore_views()
        else:
            if self.maximized_view:
                self._restore_views()
            self._maximize_view(name)

    def _maximize_view(self, name):
        for other, view in self.views.items():
            if other != name:
                view["column"].forget(view["frame"])
        self.root.update_idletasks()
        width = self.columns.winfo_width()
        self.columns.sashpos(0, width - 5 if name in COLUMNS["left"] else 5)
        self.maximized_view = name
        for other, view in self.views.items():
            view["button"].config(text="▣" if other == name else "❐")
        self.update_views()

    def _restore_views(self):
        for side, names in COLUMNS.items():
            column = self.column_panes[side]
            for position, name in enumerate(names):
                # insert() moves a frame to its position; it cannot insert past the end of a column.
                if position < len(column.panes()):
                    column.insert(position, self.views[name]["frame"], weight=1)
                else:
                    column.add(self.views[name]["frame"], weight=1)
        self.maximized_view = None
        for view in self.views.values():
            view["button"].config(text="❐")
        self.root.after(50, self.reset_layout)
        self.update_views()
