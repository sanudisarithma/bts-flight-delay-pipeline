import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

from preprocess import config
from preprocess import CsvLoader, Cleaner, DelayAnalyzer, default_steps, CAUSE_LABELS
from preprocess.utils import sha256_of

RECENT_YEARS = 3          # carriers and airports are ranked over the latest 3 full years
TOP_CARRIERS = 15         # only the busiest carriers / airports are ranked, small ones are too noisy
TOP_AIRPORTS = 50
MIN_FLIGHTS_MODEL = 20    # rows with fewer arrived flights give very noisy rates, so the model skips them


def busiest_then_rank(table, how_many):
    return table.nlargest(how_many, "arr_arrived").sort_values("delay_rate", ascending=False)


def name_history(raw):
    tables = []
    for code_col, name_col in (("carrier", "carrier_name"), ("airport", "airport_name")):
        d = raw.dropna(subset=[code_col, name_col]).copy()
        d["month_number"] = d["year"] * 100 + d["month"]          # 200306 = June 2003
        t = d.groupby([code_col, name_col]).agg(rows=("month_number", "size"),
                                                 first_month=("month_number", "min"),
                                                 last_month=("month_number", "max")).reset_index()
        t.columns = ["code", "name", "rows", "first_month", "last_month"]
        t.insert(0, "kind", code_col)
        names_per_code = t.groupby("code")["name"].nunique()
        t = t[t["code"].isin(names_per_code[names_per_code > 1].index)]
        tables.append(t)
    return pd.concat(tables).sort_values(["kind", "code", "first_month"])


def save_fig(fig, name):
    path = os.path.join(config.FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  saved", path)


def obtain():
    print("== Obtain ==")
    digest = sha256_of(config.RAW_CSV)
    if os.path.isfile(config.MANIFEST):
        with open(config.MANIFEST, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if digest not in text:
            raise SystemExit("the raw csv does not match the checksum in MANIFEST.txt - it has been changed")
        print("  checksum matches MANIFEST.txt")
    else:
        print("  (no MANIFEST.txt found, skipping the checksum check)")
    loader = CsvLoader(config.RAW_CSV)
    raw = loader.load()
    print("  rows: {:,}  columns: {}".format(loader.rows_loaded, raw.shape[1]))
    return raw


def scrub(raw):
    print("== Scrub ==")
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    table = pd.DataFrame({"missing": missing, "percent": (missing / len(raw) * 100).round(3)})
    table.index.name = "column"
    table.to_csv(os.path.join(config.RESULTS_DIR, "missing_values_raw.csv"))
    print("  columns with missing values in the raw file:", len(table))

    history = name_history(raw)
    history.to_csv(os.path.join(config.RESULTS_DIR, "code_name_history.csv"), index=False)
    print("  codes with more than one name: carriers {}, airports {}  (results/code_name_history.csv)".format(
        history[history["kind"] == "carrier"]["code"].nunique(), history[history["kind"] == "airport"]["code"].nunique()))

    cleaner = Cleaner(default_steps())
    clean = cleaner.run(raw)
    report = cleaner.report
    report.to_csv(os.path.join(config.RESULTS_DIR, "scrub_report.csv"), index=False)
    print(report[["step", "records_affected", "rows_before", "rows_after"]].to_string(index=False))
    for i in range(len(report)):
        print("  - " + report.loc[i, "step"] + ": " + report.loc[i, "note"])
    print("  cleaning took", cleaner.seconds, "seconds")

    clean.to_parquet(config.CLEAN_PARQUET, index=False)
    print("  cleaned data saved to", config.CLEAN_PARQUET)
    return clean



def explore(analyzer):
    print("== Explore ==")
    found = {
        "total_rows_after_dedup": analyzer.total_rows,
        "usable_rows": analyzer.usable_rows,
        "excluded_rows": analyzer.excluded_rows,
        "overall_delay_rate": analyzer.overall_delay_rate(),
        "overall_cancel_rate": analyzer.overall_cancel_rate(),
    }
    last_full = analyzer.last_full_year()
    first_recent = last_full - RECENT_YEARS + 1
    found["last_full_year"] = last_full
    found["recent_window"] = [first_recent, last_full]

    by_year = analyzer.rate_by("year", min_flights=0).sort_index()
    by_year["months_in_data"] = analyzer.months_per_year()
    by_year.to_csv(os.path.join(config.RESULTS_DIR, "delay_by_year.csv"))
    found["best_year"] = int(by_year["delay_rate"].idxmin())
    found["best_year_rate"] = float(by_year["delay_rate"].min())
    found["worst_year"] = int(by_year["delay_rate"].idxmax())
    found["worst_year_rate"] = float(by_year["delay_rate"].max())
    full = by_year[by_year["months_in_data"] == 12]
    part = by_year[by_year["months_in_data"] < 12]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(full.index, full["delay_rate"] * 100, marker="o", color="black", label="full year")
    if len(part) > 0:
        ax.plot(part.index, part["delay_rate"] * 100, marker="o", linestyle="none",
                markerfacecolor="white", color="black", label="partial year (fewer than 12 months)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Arrived flights 15+ min late (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_fig(fig, "fig1_delay_rate_by_year.png")

    by_month = analyzer.rate_by("month", min_flights=0).sort_index()
    by_month.to_csv(os.path.join(config.RESULTS_DIR, "delay_by_month.csv"))
    found["best_month"] = int(by_month["delay_rate"].idxmin())
    found["best_month_rate"] = float(by_month["delay_rate"].min())
    found["worst_month"] = int(by_month["delay_rate"].idxmax())
    found["worst_month_rate"] = float(by_month["delay_rate"].max())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(by_month.index, by_month["delay_rate"] * 100, color="0.6", edgecolor="black")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month of the year (all years together)")
    ax.set_ylabel("Arrived flights 15+ min late (%)")
    save_fig(fig, "fig2_delay_rate_by_month.png")

    by_carrier = analyzer.rate_by("carrier", min_flights=0, first_year=first_recent)
    by_carrier = busiest_then_rank(by_carrier, TOP_CARRIERS)
    names = analyzer.names_for("carrier", "carrier_label")
    by_carrier.insert(0, "carrier_name", [names[c] for c in by_carrier.index])
    by_carrier.to_csv(os.path.join(config.RESULTS_DIR, "delay_by_carrier_recent.csv"))
    recent_rate = analyzer.overall_delay_rate(first_year=first_recent)
    found["recent_delay_rate"] = recent_rate
    found["carriers_ranked"] = len(by_carrier)
    found["worst_carrier"] = str(by_carrier.index[0])
    found["worst_carrier_rate"] = float(by_carrier["delay_rate"].iloc[0])
    found["best_carrier"] = str(by_carrier.index[-1])
    found["best_carrier_rate"] = float(by_carrier["delay_rate"].iloc[-1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(list(by_carrier.index), by_carrier["delay_rate"] * 100, color="0.6", edgecolor="black")
    ax.axvline(recent_rate * 100, color="black", linestyle="--", label="all carriers together")
    ax.invert_yaxis()
    ax.set_xlabel("Arrived flights 15+ min late (%), {}-{}".format(first_recent, last_full))
    ax.set_ylabel("Carrier code")
    ax.legend()
    save_fig(fig, "fig3_delay_rate_by_carrier.png")

    shares = analyzer.cause_share_by_year()
    shares.to_csv(os.path.join(config.RESULTS_DIR, "cause_share_by_year.csv"))
    overall = analyzer.cause_share()
    found["cause_share_all_years"] = {CAUSE_LABELS[k]: float(v) for k, v in overall.items()}
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(shares))
    greys = ["0.15", "0.35", "0.55", "0.75", "0.9"]
    hatches = ["", "//", "..", "xx", "\\\\"]
    for i, col in enumerate(shares.columns):
        ax.bar(shares.index, shares[col] * 100, bottom=bottom, color=greys[i], edgecolor="black",
               hatch=hatches[i], label=CAUSE_LABELS[col])
        bottom = bottom + shares[col].to_numpy() * 100
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of delay minutes (%)")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32))
    save_fig(fig, "fig4_delay_causes_by_year.png")

    all_airports = analyzer.rate_by("airport", min_flights=1000, first_year=first_recent)
    by_airport = busiest_then_rank(all_airports, TOP_AIRPORTS)
    anames = analyzer.names_for("airport", "airport_label")
    by_airport.insert(0, "airport_name", [anames[a] for a in by_airport.index])
    by_airport.to_csv(os.path.join(config.RESULTS_DIR, "delay_by_airport_recent.csv"))
    found["airports_ranked"] = len(by_airport)
    found["worst_airport"] = str(by_airport.index[0])
    found["worst_airport_rate"] = float(by_airport["delay_rate"].iloc[0])
    found["best_airport"] = str(by_airport.index[-1])
    found["best_airport_rate"] = float(by_airport["delay_rate"].iloc[-1])
    print("  overall delay rate {:.3f}; recent {:.3f}".format(found["overall_delay_rate"], recent_rate))
    if len(all_airports) >= 3:
        # uses every airport with at least 1000 flights in the window, not only the busiest ones
        found["airports_in_correlation"] = len(all_airports)
        found["spearman_airport_size_vs_rate"] = float(all_airports["arr_flights"].corr(all_airports["delay_rate"], method="spearman"))
        print("  spearman(airport size, delay rate) = {:.3f}".format(found["spearman_airport_size_vs_rate"]))
    return found


def make_features(frame, codes):
    """Feature table for the model. `codes` are the numbers for carrier and airport."""
    return pd.DataFrame({
        "carrier": codes[:, 0],
        "airport": codes[:, 1],
        "year": frame["year"].to_numpy(),
        "month": frame["month"].to_numpy(),
        "arr_flights": frame["arr_flights"].to_numpy(),
    })


def score(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def model(analyzer):
    print("== Model ==")
    data = analyzer.get_rows(["year", "month", "carrier", "airport", "arr_flights", "arr_arrived", "arr_del15"])
    data = data[data["arr_arrived"] >= MIN_FLIGHTS_MODEL].copy()
    data["delay_rate"] = data["arr_del15"] / data["arr_arrived"]

    test_year = analyzer.last_full_year()
    train = data[data["year"] < test_year].copy()
    test = data[data["year"] == test_year].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError("not enough years to make a train/test split")
    print("  train: years before {} ({:,} rows)   test: {} ({:,} rows)".format(test_year, len(train), test_year, len(test)))

    # baseline 1: the same number for everybody
    overall_mean = train["delay_rate"].mean()
    # baseline 2: the average of this carrier+airport in the training years
    pair_mean = train.groupby(["carrier", "airport"])["delay_rate"].mean().rename("pair_mean").reset_index()
    test = test.merge(pair_mean, on=["carrier", "airport"], how="left")
    pairs_not_seen = int(test["pair_mean"].isna().sum())
    y_test = test["delay_rate"].to_numpy()
    pred_base1 = np.full(len(test), overall_mean)
    pred_base2 = test["pair_mean"].fillna(overall_mean).to_numpy()

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    codes_train = encoder.fit_transform(train[["carrier", "airport"]])
    codes_test = encoder.transform(test[["carrier", "airport"]])
    x_train = make_features(train, codes_train)
    x_test = make_features(test, codes_test)
    reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=42)
    reg.fit(x_train, train["delay_rate"].to_numpy())
    pred_model = reg.predict(x_test)

    rows = []
    for name, pred in [("baseline: same value for all", pred_base1),
                       ("baseline: carrier+airport average", pred_base2),
                       ("HistGradientBoostingRegressor", pred_model)]:
        row = score(y_test, pred)
        row["model"] = name
        rows.append(row)
    results = pd.DataFrame(rows)[["model", "mae", "rmse", "r2"]]
    results.to_csv(os.path.join(config.RESULTS_DIR, "model_results.csv"), index=False)
    print(results.round(4).to_string(index=False))

    sample_size = min(20000, len(x_test))
    pick = np.random.RandomState(0).choice(len(x_test), size=sample_size, replace=False)
    imp = permutation_importance(reg, x_test.iloc[pick], y_test[pick], n_repeats=5, random_state=0,
                                 scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"feature": x_test.columns, "mae_increase": imp.importances_mean}).sort_values(
        "mae_increase", ascending=False)
    importance.to_csv(os.path.join(config.RESULTS_DIR, "permutation_importance.csv"), index=False)
    print(importance.round(5).to_string(index=False))

    return {
        "test_year": test_year,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_pairs_not_seen_in_training": pairs_not_seen,
        "model_results": results.to_dict(orient="records"),
        "feature_importance": importance.to_dict(orient="records"),
    }



def interpret(found):
    print("== iNterpret ==")
    with open(os.path.join(config.RESULTS_DIR, "key_findings.json"), "w") as f:
        json.dump(found, f, indent=2, default=str)
    best_model = min(found["model"]["model_results"], key=lambda r: r["mae"])
    print("  {:.1%} of arrived flights were 15+ min late over the whole data ({:.1%} of scheduled flights were cancelled).".format(
        found["overall_delay_rate"], found["overall_cancel_rate"]))
    print("  Lowest year {} ({:.1%}), highest year {} ({:.1%}).".format(
        found["best_year"], found["best_year_rate"], found["worst_year"], found["worst_year_rate"]))
    print("  Best month {} ({:.1%}), worst month {} ({:.1%}).".format(
        found["best_month"], found["best_month_rate"], found["worst_month"], found["worst_month_rate"]))
    print("  Carriers {}-{}: worst {} ({:.1%}), best {} ({:.1%}).".format(
        found["recent_window"][0], found["recent_window"][1], found["worst_carrier"], found["worst_carrier_rate"],
        found["best_carrier"], found["best_carrier_rate"]))
    print("  Lowest test MAE: {} ({:.4f}).".format(best_model["model"], best_model["mae"]))
    print("  saved results/key_findings.json")


def main():
    config.make_folders()
    raw = obtain()
    clean = scrub(raw)
    del raw
    analyzer = DelayAnalyzer(clean)
    found = explore(analyzer)
    found["model"] = model(analyzer)
    interpret(found)


if __name__ == "__main__":
    main()
