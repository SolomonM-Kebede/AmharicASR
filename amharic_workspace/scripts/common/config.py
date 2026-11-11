"""Project paths and Hugging Face settings.

Machine-specific locations come from environment variables (see env.amh.sh).
Import this module before transformers or datasets: it sets the Hugging Face
cache location and offline flags when it loads.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]          # amharic_workspace/
DATA_DIR = Path(os.environ.get("AMHARIC_DATA_DIR", REPO_ROOT)).expanduser()


def dataset_path(name, default):
    """Resolve an input path. Existence is checked by require(), when a script needs it."""
    return Path(os.environ.get(name, default)).expanduser()


def require(path):
    """Fail with a clear message if an input a script needs is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Set the matching AMHARIC_* variable in env.amh.sh.")
    return path


def output_path(name, default):
    """Output folders: created by the scripts when they are needed."""
    return Path(os.environ.get(name, default)).expanduser()


# Repository layout (results and splits are committed; see README)
RESULTS = REPO_ROOT / "results"
PERSAMPLE = RESULTS / "persample"
GOOGLE = RESULTS / "google"
SPLITS = REPO_ROOT / "splits"
MANIFESTS = SPLITS / "manifests"
FIGURES = REPO_ROOT / "figures"
SUBSET_688_IDX = SPLITS / "subset_688_idx.txt"

# Inputs (checked when imported)
CV_DIR = dataset_path("AMHARIC_CV_DIR", DATA_DIR / "cv-corpus-25.0-2026-03-09" / "am")
LEYU_JOINT_CACHE = dataset_path(
    "AMHARIC_LEYU_JOINT_CACHE", DATA_DIR / "phase2_leyu" / "dataset_cache_joint_independent")
LEYU_SI_CACHE = dataset_path(
    "AMHARIC_LEYU_SI_CACHE", DATA_DIR / "phase2_leyu" / "dataset_cache_2026-08-expanded-spkindep")
JOINT_STANDARD_CACHE = dataset_path(
    "AMHARIC_JOINT_STANDARD_CACHE", DATA_DIR / "phase2b_leyu_standard" / "standard_speech_cache_ratio1.0")
PHASE1_MERGED = dataset_path(
    "AMHARIC_PHASE1_MERGED", DATA_DIR / "phase1_curriculum" / "models" / "phase1_best_model_merged")
D1_CACHE = dataset_path(
    "AMHARIC_D1_CACHE", DATA_DIR / "phase2_leyu" / "dataset_cache_si_size_matched")

TI_CACHE = dataset_path(
    "AMHARIC_TI_CACHE", DATA_DIR / "phase2_leyu" / "dataset_cache_text_independent")
PHASE1_STD_TEST = dataset_path(
    "AMHARIC_PHASE1_STD_TEST", DATA_DIR / "phase1_curriculum" / "dataset_cache_rebuilt")

# Outputs (created by the scripts)
CACHE_DIR = output_path("AMHARIC_CACHE_DIR", Path.home() / "AmharicASR" / "clean_cache")
CHECKPOINT_DIR = output_path("AMHARIC_CHECKPOINT_DIR", DATA_DIR / "checkpoints")
LOG_DIR = output_path("AMHARIC_LOG_DIR", DATA_DIR / "slurm_logs")

# Shared constants
WHISPER_MODEL = "openai/whisper-large-v3"
SEED = 42
MAX_LABEL_LENGTH = 448
GOOGLE_MAX_SYNC_SECONDS = 55   # sending rule for Google's synchronous API (API limit: 60 s)

# Hugging Face: set before transformers or datasets load.
# HF_HOME is the cache root; hub/ inside it holds the Whisper snapshot.
HF_HOME = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = str(HF_HOME)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"