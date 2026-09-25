"""Start the Slice Viewer: python -m neuroxai.viewer [VOLUME]"""

import argparse
import tkinter as tk

from ..ui import Splash


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neuroxai.viewer", description="NeuroXAI Slice Viewer")
    parser.add_argument("volume", nargs="?", help="NIfTI (.nii, .nii.gz) or MetaImage (.mhd) file to open")
    args = parser.parse_args(argv)

    root = tk.Tk()
    root.withdraw()
    splash = Splash(root, "NeuroXAI Slice Viewer", "Loading...")
    from .app import SliceViewer  # imports SimpleITK and matplotlib

    viewer = SliceViewer(root)
    splash.close()
    root.deiconify()
    if args.volume:
        root.after(200, lambda: viewer.open_volume(args.volume))
    root.mainloop()


if __name__ == "__main__":
    main()
