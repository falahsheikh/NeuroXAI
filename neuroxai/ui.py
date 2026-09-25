"""Tkinter helpers that the two applications share."""

import tkinter as tk

BACKGROUND = "#fdf6e3"
ACCENT = "#000080"
MUTED = "#93a1a1"


class Splash:
    """Start screen that shows while an application loads its libraries.

    Tkinter cannot update the window while a library loads, so the screen shows a message and no animation.
    """

    def __init__(self, root, title, message, width=400, height=200):
        self.window = tk.Toplevel(root)
        self.window.title(title)
        self.window.resizable(False, False)
        self.window.configure(bg=BACKGROUND)
        x = (self.window.winfo_screenwidth() - width) // 2
        y = (self.window.winfo_screenheight() - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        frame = tk.Frame(self.window, bg=BACKGROUND, padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(frame, text=title, font=("Segoe UI", 14, "bold"), fg=ACCENT, bg=BACKGROUND).pack(pady=(10, 16))
        tk.Label(frame, text=message, font=("Segoe UI", 10), wraplength=350, bg=BACKGROUND).pack()
        tk.Label(
            frame, text="Developed by Falah Sheikh", font=("Segoe UI", 10, "italic"), fg=MUTED, bg=BACKGROUND
        ).pack(side=tk.BOTTOM)
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)
        self.window.update()

    def close(self):
        self.window.destroy()


def maximize_window(root):
    """Maximize a window. Windows and macOS use the "zoomed" state; X11 uses the -zoomed attribute."""
    try:
        root.state("zoomed")
    except tk.TclError:
        try:
            root.attributes("-zoomed", True)
        except tk.TclError:
            pass  # the window keeps its geometry
