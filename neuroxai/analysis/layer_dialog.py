"""Dialog to select the layer that each explanation method uses."""

import tkinter as tk
from tkinter import messagebox, ttk

from ..xai import METHODS


class LayerDialog:
    """Modal dialog for an Explainer.

    show() returns {"layers": {method: layer}, "reanalyze": bool} after the user applies a selection,
    or None if the user cancels. Apply sets the layers of the explainer.
    """

    def __init__(self, parent, explainer):
        self.parent = parent
        self.explainer = explainer
        self.result = None
        self.window = None
        self.choices = {}

    def show(self):
        self.window = tk.Toplevel(self.parent)
        self.window.title("Explanation Layers")
        self.window.transient(self.parent)
        self.window.geometry(f"+{self.parent.winfo_rootx() + 100}+{self.parent.winfo_rooty() + 50}")
        self._build()
        self.window.grab_set()
        self.window.wait_window()
        return self.result

    def _build(self):
        frame = ttk.Frame(self.window, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        layers = self.explainer.layers

        ttk.Label(frame, text="Explanation Layers", font=("TkDefaultFont", 13, "bold")).pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="Grad-CAM++ uses the output of the selected layer. "
            f"The default layer ({self.explainer.default_layer}) makes the final feature map, "
            "which the classifier pools. Earlier layers give maps with more detail and less class information.",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 12))

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            list_frame, height=12, width=60, font="TkFixedFont", yscrollcommand=scrollbar.set, activestyle="none"
        )
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for i, name in enumerate(layers, start=1):
            self.listbox.insert(tk.END, f"{i:3d}. {name:<32} {self._output_shape(name)}")
        self.listbox.bind("<Double-1>", self._on_double_click)
        ttk.Label(frame, text="Double-click a layer to use it for a method.", foreground="#8B4513").pack(
            anchor=tk.W, pady=(4, 12)
        )

        choice_frame = ttk.Frame(frame)
        choice_frame.pack(fill=tk.X)
        for row, (method, label) in enumerate(METHODS.items()):
            ttk.Label(choice_frame, text=f"{label}:", width=20).grid(row=row, column=0, sticky=tk.W, pady=4)
            self.choices[method] = tk.StringVar(value=self.explainer.selected[method])
            ttk.Combobox(
                choice_frame, textvariable=self.choices[method], values=layers, state="readonly", width=34
            ).grid(row=row, column=1, sticky=tk.EW, pady=4)
        choice_frame.columnconfigure(1, weight=1)

        quick = ttk.Frame(frame)
        quick.pack(fill=tk.X, pady=(8, 16))
        ttk.Label(quick, text="All methods:").pack(side=tk.LEFT)
        for text, layer in (
            ("First", layers[0]),
            ("Middle", layers[len(layers) // 2]),
            ("Last", layers[-1]),
            ("Default", self.explainer.default_layer),
        ):
            ttk.Button(quick, text=text, command=lambda layer=layer: self._set_all(layer)).pack(
                side=tk.LEFT, padx=(6, 0)
            )

        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Apply and Analyze Again", command=lambda: self._apply(reanalyze=True)).pack(
            side=tk.RIGHT
        )
        ttk.Button(buttons, text="Apply", command=lambda: self._apply(reanalyze=False)).pack(side=tk.RIGHT, padx=6)
        ttk.Button(buttons, text="Cancel", command=self.window.destroy).pack(side=tk.RIGHT)
        self.window.bind("<Escape>", lambda _: self.window.destroy())

    def _output_shape(self, name):
        try:
            shape = self.explainer.model.get_layer(name).output.shape
        except (AttributeError, ValueError):
            return ""
        return " x ".join(str(size) for size in shape[1:])

    def _set_all(self, layer):
        for choice in self.choices.values():
            choice.set(layer)

    def _on_double_click(self, event):
        layer = self.explainer.layers[self.listbox.nearest(event.y)]
        menu = tk.Menu(self.window, tearoff=0)
        for method, label in METHODS.items():
            menu.add_command(label=f"Use for {label}", command=lambda m=method: self.choices[m].set(layer))
        menu.add_command(label="Use for all methods", command=lambda: self._set_all(layer))
        menu.tk_popup(event.x_root, event.y_root)

    def _apply(self, reanalyze):
        try:
            for method, choice in self.choices.items():
                self.explainer.set_layer(method, choice.get())
        except ValueError as e:
            messagebox.showerror("Explanation Layers", str(e), parent=self.window)
            return
        self.result = {"layers": dict(self.explainer.selected), "reanalyze": reanalyze}
        self.window.destroy()
