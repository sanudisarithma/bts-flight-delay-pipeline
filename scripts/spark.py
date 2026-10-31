import argparse
import contextlib
import gc
import io
import json
import os
import statistics
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Spark must use the same python as this script (important on Windows)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from preprocess import config
from preprocess.utils import system_info

COLUMNS = ["year", "carrier", "arr_arrived", "arr_del15", "Usable"]
REPEATS = 6              # first run is the warm-up, the other 5 are kept
DRIVER_MEMORY = "4g"     # lower this if your laptop has little RAM


def time_it(function):
    times = []
    for i in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    kept = times[1:]
    return statistics.median(kept), min(kept), max(kept)


def spark_path(path):
    return path.replace("\\", "/")


def make_scaled_files(factors):
    base = pd.read_parquet(config.CLEAN_PARQUET, columns=COLUMNS)
    paths = {}
    for k in factors:
        path = os.path.join(config.SPARK_DIR, "rows_x{}.parquet".format(k))
        if not os.path.exists(path):
            big = pd.concat([base] * k, ignore_index=True)
            big.to_parquet(path, index=False)
        paths[k] = (path, len(base) * k)
    return paths


def pandas_stage(path):
    df = pd.read_parquet(path, columns=COLUMNS)
    df = df[df["Usable"]]
    out = df.groupby(["year", "carrier"]).agg(arrived=("arr_arrived", "sum"),
                                              late=("arr_del15", "sum")).reset_index()
    out["delay_rate"] = out["late"] / out["arrived"]
    return out.sort_values(["year", "carrier"]).reset_index(drop=True)


def spark_query(spark, path):
    df = spark.read.parquet(spark_path(path))
    return (df.filter(F.col("Usable"))                               # narrow
              .groupBy("year", "carrier")                            # wide: shuffle
              .agg(F.sum("arr_arrived").alias("arrived"),
                   F.sum("arr_del15").alias("late"))
              .withColumn("delay_rate", F.col("late") / F.col("arrived"))
              .orderBy("year", "carrier"))                           # wide: shuffle


def spark_stage(spark, path):
    rows = spark_query(spark, path).collect()     # collect() is the action that runs everything
    return pd.DataFrame([r.asDict() for r in rows])


def check_same_answer(p, s):
    p = p.copy()
    s = s.copy()
    p["year"] = p["year"].astype(int)
    s["year"] = s["year"].astype(int)
    p["carrier"] = p["carrier"].astype(str)
    s["carrier"] = s["carrier"].astype(str)
    both = p.merge(s, on=["year", "carrier"], suffixes=("_pd", "_sp"))
    if not (len(both) == len(p) == len(s)):
        raise ValueError("pandas and Spark returned different groups")
    if (both["arrived_pd"] - both["arrived_sp"]).abs().max() > 1e-6:
        raise ValueError("arrived flights differ")
    if (both["delay_rate_pd"] - both["delay_rate_sp"]).abs().max() > 1e-9:
        raise ValueError("delay rates differ")
    print("  pandas and Spark give the same {} groups".format(len(both)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 5, 25],
                        help="how many times the cleaned data is repeated (default 1 5 25)")
    args = parser.parse_args()

    config.make_folders()
    info = system_info()
    print(json.dumps(info, indent=2))
    files = make_scaled_files(args.factors)

    start = time.perf_counter()
    spark = (SparkSession.builder.master("local[*]").appName("task4")
             .config("spark.driver.memory", DRIVER_MEMORY)
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    startup = time.perf_counter() - start
    spark.sparkContext.setLogLevel("ERROR")
    cores = spark.sparkContext.defaultParallelism
    print("Spark {} started in {:.1f}s; local[*] uses {} cores".format(spark.version, startup, cores))

    # the physical plan: look for the word Exchange, that is where the shuffles are
    biggest_path = files[max(files)][0]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        spark_query(spark, biggest_path).explain(mode="formatted")
    with open(os.path.join(config.RESULTS_DIR, "spark_physical_plan.txt"), "w") as f:
        f.write(buffer.getvalue())
    print("  physical plan saved to results/spark_physical_plan.txt")
    check_same_answer(pandas_stage(biggest_path), spark_stage(spark, biggest_path))

    spark_settings = [
        ("Spark default (AQE on, 200 shuffle partitions)", "true", 200),
        ("Spark AQE off, 200 shuffle partitions", "false", 200),
        ("Spark AQE off, {} shuffle partitions".format(cores), "false", cores),
    ]
    rows = []
    for k in sorted(files):
        path, n_rows = files[k]
        input_partitions = spark.read.parquet(spark_path(path)).rdd.getNumPartitions()
        med, lo, hi = time_it(lambda: pandas_stage(path))
        rows.append({"rows": n_rows, "engine": "pandas", "input_partitions": None,
                     "median_s": med, "min_s": lo, "max_s": hi})
        print("{:>12,} rows | pandas {:.3f}s".format(n_rows, med))
        for label, aqe, shuffle_partitions in spark_settings:
            spark.conf.set("spark.sql.adaptive.enabled", aqe)
            spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
            med, lo, hi = time_it(lambda: spark_stage(spark, path))
            rows.append({"rows": n_rows, "engine": label, "input_partitions": input_partitions,
                         "median_s": med, "min_s": lo, "max_s": hi})
            print("{:>12,} rows | {}: {:.3f}s (input partitions {})".format(n_rows, label, med, input_partitions))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(config.RESULTS_DIR, "spark_vs_pandas.csv"), index=False)
    details = dict(info)
    details["spark_version"] = spark.version
    details["spark_startup_seconds"] = round(startup, 2)
    details["default_parallelism"] = cores
    with open(os.path.join(config.RESULTS_DIR, "spark_system.json"), "w") as f:
        json.dump(details, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["-o", "--s", "-.^", ":D"]
    i = 0
    for engine in res["engine"].unique():
        g = res[res["engine"] == engine]
        ax.errorbar(g["rows"], g["median_s"], yerr=[g["median_s"] - g["min_s"], g["max_s"] - g["median_s"]],
                    fmt=styles[i % 4], color="black", capsize=3, label=engine)
        i = i + 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_ylabel("Time in seconds, median of 5 (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG_DIR, "fig5_spark_vs_pandas.png"), dpi=200)
    plt.close(fig)
    spark.stop()


if __name__ == "__main__":
    main()
