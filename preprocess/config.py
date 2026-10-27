import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_CSV = os.path.join(DATA_DIR, "Airline_Delay_Cause.csv")
MANIFEST = os.path.join(DATA_DIR, "MANIFEST.txt")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CLEAN_PARQUET = os.path.join(PROCESSED_DIR, "clean.parquet")
BENCH_DIR = os.path.join(DATA_DIR, "bench")
SPARK_DIR = os.path.join(DATA_DIR, "spark_scaled")

RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")


def make_folders():
    """Create the output folders if they are not there yet."""
    for folder in (PROCESSED_DIR, BENCH_DIR, SPARK_DIR, RESULTS_DIR, FIG_DIR):
        if not os.path.isdir(folder):
            os.makedirs(folder)
