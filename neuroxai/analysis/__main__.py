"""Start the Analysis Tool: python -m neuroxai.analysis [--model MODEL] [--image IMAGE]"""

import argparse
import tkinter as tk

from ..config import DEFAULT_MODEL
from ..ui import Splash


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m neuroxai.analysis", description="NeuroXAI Analysis Tool")
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="Keras model to load (default: the model of the paper)"
    )
    parser.add_argument("--image", help="coronal slice to analyze when the tool starts")
    args = parser.parse_args(argv)

    root = tk.Tk()
    root.withdraw()
    splash = Splash(root, "NeuroXAI Analysis Tool", "Loading TensorFlow...")
    from .app import AnalysisApp  # imports TensorFlow, which takes some seconds

    AnalysisApp(root, model_path=args.model, image_path=args.image)
    splash.close()
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
