"""Task 2 - storage format investigation.

Writes the raw data (all 21 columns) as CSV, JSON lines, pickle and Parquet, with and
without compression, and measures:
   file size | time to write | time to read everything | time to read ONE column
Each timing is repeated 6 times, the first run is thrown away (warm-up) and the median and
the range (min, max) of the other 5 are saved.

YAML is far too slow for the whole file, so it is only tested on a small sample
(compared with CSV on the same sample).

Run from the project folder:  python scripts/step2_storage_benchmark.py
(for a quick test:            python scripts/step2_storage_benchmark.py --rows 50000)
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from flightproc import config
from flightproc import CsvLoader
from flightproc.utils import system_info

REPEATS = 6              # the first one is the warm-up, so 5 timings are kept
ONE_COLUMN = "arr_delay"


def time_it(function):
    """Run function() REPEATS times. Returns (median, min, max) of runs 2..6 in seconds."""
    times = []
    for i in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    kept = times[1:]   # throw the first (warm-up) away
    return statistics.median(kept), min(kept), max(kept)


def get_formats():
    """Returns a list of (name, file name, write function, read-all function, read-one-column function)."""
    c = ONE_COLUMN
    return [
        ("CSV", "data.csv",
         lambda df, p: df.to_csv(p, index=False),
         lambda p: pd.read_csv(p, low_memory=False),
         lambda p: pd.read_csv(p, usecols=[c])),
        ("CSV gzip", "data.csv.gz",
         lambda df, p: df.to_csv(p, index=False, compression="gzip"),
         lambda p: pd.read_csv(p, low_memory=False),
         lambda p: pd.read_csv(p, usecols=[c])),
        ("JSON lines", "data.jsonl",
         lambda df, p: df.to_json(p, orient="records", lines=True),
         lambda p: pd.read_json(p, lines=True),
         lambda p: pd.read_json(p, lines=True)[[c]]),        # JSON has to read everything
        ("JSON lines gzip", "data.jsonl.gz",
         lambda df, p: df.to_json(p, orient="records", lines=True, compression="gzip"),
         lambda p: pd.read_json(p, lines=True),
         lambda p: pd.read_json(p, lines=True)[[c]]),
        ("Pickle", "data.pkl",
         lambda df, p: df.to_pickle(p),
         lambda p: pd.read_pickle(p),
         lambda p: pd.read_pickle(p)[[c]]),                  # pickle also has to read everything
        ("Pickle gzip", "data.pkl.gz",
         lambda df, p: df.to_pickle(p, compression="gzip"),
         lambda p: pd.read_pickle(p),
         lambda p: pd.read_pickle(p)[[c]]),
        ("Parquet (no compression)", "data_none.parquet",
         lambda df, p: df.to_parquet(p, compression=None, index=False),
         lambda p: pd.read_parquet(p),
         lambda p: pd.read_parquet(p, columns=[c])),
        ("Parquet snappy", "data_snappy.parquet",
         lambda df, p: df.to_parquet(p, compression="snappy", index=False),
         lambda p: pd.read_parquet(p),
         lambda p: pd.read_parquet(p, columns=[c])),
        ("Parquet zstd", "data_zstd.parquet",
         lambda df, p: df.to_parquet(p, compression="zstd", index=False),
         lambda p: pd.read_parquet(p),
         lambda p: pd.read_parquet(p, columns=[c])),
    ]


def dtype_changes(original, read_back):
    """How many columns came back with a different data type than they went in with."""
    count = 0
    for col in original.columns:
        if col in read_back.columns and original[col].dtype != read_back[col].dtype:
            count = count + 1
    return count


def yaml_test(df, n_rows):
    """YAML against CSV on a small sample."""
    import yaml
    if hasattr(yaml, "CSafeDumper"):
        dumper, loader = yaml.CSafeDumper, yaml.CSafeLoader   # the fast C versions
    else:
        dumper, loader = yaml.SafeDumper, yaml.SafeLoader
    small = df.head(n_rows)
    records = small.to_dict(orient="records")
    yaml_path = os.path.join(config.BENCH_DIR, "small.yaml")
    csv_path = os.path.join(config.BENCH_DIR, "small.csv")

    def write_yaml():
        with open(yaml_path, "w") as f:
            yaml.dump(records, f, Dumper=dumper)

    def read_yaml():
        with open(yaml_path) as f:
            return pd.DataFrame(yaml.load(f, Loader=loader))

    rows = []
    jobs = [("YAML", write_yaml, read_yaml, yaml_path),
            ("CSV", lambda: small.to_csv(csv_path, index=False), lambda: pd.read_csv(csv_path), csv_path)]
    for name, write, read, path in jobs:
        w = time_it(write)
        r = time_it(read)
        rows.append({"format": name, "rows": n_rows, "size_mb": os.path.getsize(path) / 1e6,
                     "write_median_s": w[0], "read_median_s": r[0]})
    return pd.DataFrame(rows)


def make_chart(res, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    y = list(range(len(res)))

    ax1.barh(y, res["size_mb"], color="0.6", edgecolor="black")
    ax1.set_yticks(y)
    ax1.set_yticklabels(res["format"])
    ax1.invert_yaxis()
    ax1.set_xlabel("File size on disk (MB)")
    ax1.set_title("(a) File size")

    h = 0.4
    err_all = [res["read_median_s"] - res["read_min_s"], res["read_max_s"] - res["read_median_s"]]
    err_col = [res["col_median_s"] - res["col_min_s"], res["col_max_s"] - res["col_median_s"]]
    ax2.barh([i - h / 2 for i in y], res["read_median_s"], h, xerr=err_all, color="0.35",
             edgecolor="black", label="read all columns")
    ax2.barh([i + h / 2 for i in y], res["col_median_s"], h, xerr=err_col, color="white",
             edgecolor="black", hatch="//", label="read one column (" + ONE_COLUMN + ")")
    ax2.set_yticks(y)
    ax2.set_yticklabels(res["format"])
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("Read time in seconds (log scale)")
    ax2.set_title("(b) Read time (bar = median of 5, line = min to max)")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=0, help="use only the first N rows (0 = all rows)")
    parser.add_argument("--yaml-rows", type=int, default=10000)
    args = parser.parse_args()

    config.make_folders()
    info = system_info()
    print(json.dumps(info, indent=2))

    df = CsvLoader(config.RAW_CSV).load()
    if args.rows > 0:
        df = df.head(args.rows)
    print("benchmark data: {:,} rows x {} columns".format(len(df), df.shape[1]))

    results = []
    for name, file_name, write, read_all, read_col in get_formats():
        path = os.path.join(config.BENCH_DIR, file_name)
        print(name, "...")
        w = time_it(lambda: write(df, path))
        r = time_it(lambda: read_all(path))
        c = time_it(lambda: read_col(path))
        back = read_all(path)
        results.append({
            "format": name,
            "size_mb": os.path.getsize(path) / 1e6,
            "write_median_s": w[0], "write_min_s": w[1], "write_max_s": w[2],
            "read_median_s": r[0], "read_min_s": r[1], "read_max_s": r[2],
            "col_median_s": c[0], "col_min_s": c[1], "col_max_s": c[2],
            "rows_back": len(back),
            "dtype_changes": dtype_changes(df, back),
        })
        del back
        print("   size {:.1f} MB | write {:.2f}s | read {:.2f}s | one column {:.3f}s".format(
            results[-1]["size_mb"], w[0], r[0], c[0]))

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(config.RESULTS_DIR, "storage_benchmark.csv"), index=False)
    details = dict(info)
    details["rows"] = len(df)
    details["columns"] = int(df.shape[1])
    details["timings_kept_per_measurement"] = REPEATS - 1
    with open(os.path.join(config.RESULTS_DIR, "storage_benchmark_system.json"), "w") as f:
        json.dump(details, f, indent=2)
    make_chart(res, os.path.join(config.FIG_DIR, "fig_storage_formats.png"))
    print(res.round(3).to_string(index=False))

    print("YAML side test ...")
    yaml_table = yaml_test(df, min(args.yaml_rows, len(df)))
    yaml_table.to_csv(os.path.join(config.RESULTS_DIR, "yaml_vs_csv_small.csv"), index=False)
    print(yaml_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
