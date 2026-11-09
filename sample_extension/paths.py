"""Centralized path configuration for the Miller sample-extension pipeline.

All machine-specific data and code locations are defined once here so that the
individual runner scripts import them instead of hard-coding paths. To run on
another machine, set the BCI_DATA_DIR environment variable to the location of
the downloaded datasets (defaults to ~/BCI_data).
"""
import os
import pathlib

# Machine-specific roots.
DATA_ROOT = pathlib.Path(os.environ.get("BCI_DATA_DIR", str(pathlib.Path.home() / "BCI_data")))
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Miller finger-flex dataset.
MILLER_DIR = DATA_ROOT / "Miller_fingerflex"

# Regression-benchmark code roots.
WIENER_DIR = PROJECT_ROOT / "Regression" / "wiener" / "code"
SW_DIR = PROJECT_ROOT / "Regression" / "switching_lm" / "code"
DTC_DIR = PROJECT_ROOT / "Regression" / "dtcnet"
DTC_CODE_DIR = DTC_DIR / "code"
FINGERFLEX_DIR = PROJECT_ROOT / "Regression" / "fingerflex"

# Sample-extension workspace.
SAMPLE_DIR = PROJECT_ROOT / "sample_extension"
PREP_DIR = SAMPLE_DIR / "results"
OUT_DIR = SAMPLE_DIR / "results"

# Method-specific processed-data output roots.
FINGERFLEX_PROCESSED = FINGERFLEX_DIR / "data" / "processed"
DTC_DATA_MILLER = DTC_DIR / "data_dtcnet_miller"
