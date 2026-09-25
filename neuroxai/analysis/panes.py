"""Panes of the Analysis Tool grid. Each pane shows one matplotlib figure."""

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .charts import placeholder

TITLE_BAR = "#000080"
BUTTON, BUTTON_ACTIVE = "#C0C0C0", "#A0A0A0"


class ChartToolbar(NavigationToolbar2Tk):
    """Matplotlib toolbar with only the navigation buttons. Save Report saves the figures."""

    toolitems = (
        *(item for item in NavigationToolbar2Tk.toolitems if item[0] in ("Home", "Back", "Forward")),
        (None, None, None, None),
        *(item for item in NavigationToolbar2Tk.toolitems if item[0] in ("Pan", "Zoom")),
    )


class ChartPane(tk.Frame):
    """A pane with a title bar and a figure. on_maximize(pane) is called when the user clicks Max or Min.

    If placeholder_when_small is true, the pane shows a placeholder until it is maximized.
    """

    def __init__(self, parent, title, on_maximize, placeholder_when_small=False):
        super().__init__(parent, relief=tk.SUNKEN, bd=1, bg="white")
        self.title = title
        self.placeholder_when_small = placeholder_when_small
        self.figure = None
        self.is_maximized = False

        bar = tk.Frame(self, bg=TITLE_BAR, height=28, relief=tk.RAISED, bd=1)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text=title, fg="white", bg=TITLE_BAR, font=("MS Sans Serif", 8, "bold"), anchor="w").pack(
            side=tk.LEFT, padx=8, pady=4, fill=tk.X, expand=True
        )
        self.button = tk.Button(
            bar,
            text="Max",
            command=lambda: on_maximize(self),
            font=("MS Sans Serif", 7),
            fg="black",
            bg=BUTTON,
            activebackground="#E0E0E0",
            relief=tk.RAISED,
            bd=2,
            width=4,
            cursor="hand2",
        )
        self.button.pack(side=tk.RIGHT, padx=4, pady=2)
        self.content = tk.Frame(self, bg="#eee8d5")
        self.content.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    @property
    def shows_placeholder(self):
        return self.figure is not None and self.placeholder_when_small and not self.is_maximized

    def show(self, figure):
        self.figure = figure
        self._render()

    def clear(self):
        self.figure = None
        self._render()

    def set_maximized(self, maximized):
        self.is_maximized = maximized
        self.button.config(text="Min" if maximized else "Max", bg=BUTTON_ACTIVE if maximized else BUTTON)
        if self.placeholder_when_small:
            self._render()

    def _render(self):
        for child in self.content.winfo_children():
            child.destroy()
        if self.figure is None:
            return
        figure = placeholder() if self.shows_placeholder else self.figure
        canvas = FigureCanvasTkAgg(figure, self.content)
        if figure is self.figure:
            ChartToolbar(canvas, self.content).update()  # packs itself at the bottom
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
