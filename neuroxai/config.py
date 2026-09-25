"""Constants shared by the NeuroXAI applications."""

from pathlib import Path

CLASS_NAMES = ("CN", "EMCI", "LMCI")
CLASS_DESCRIPTIONS = {
    "CN": "Cognitively normal",
    "EMCI": "Early mild cognitive impairment",
    "LMCI": "Late mild cognitive impairment",
}

# Input size (height, width) of the classifier.
IMG_SIZE = (224, 224)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "efficientnetv2b0.keras"

RESEARCH_NOTICE = "Research software. Not a medical device. Do not use the results for diagnosis."
