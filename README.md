# Airline Delay Project (NIB7143CEM)

This project cleans and studies US airline delay data from the Bureau of Transportation Statistics (BTS).
I wrote a Python package called `preprocess` to load and clean the data, and one script for each task.

## Data

- File: `data/Airline_Delay_Cause.csv` (BTS "Airline On-Time Statistics and Delay Causes")
- Source: https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp (the download date is in `data/MANIFEST.txt`)
- Licence: BTS data is a US government work. The government data catalog lists it with the USA.gov public domain label.
- Size: 398,233 rows and 21 columns, 38 carriers, 425 airports, June 2003 to January 2025
- One row is one carrier, one airport and one month. There is no personal data.
- SHA-256: `74bb02986758aa8fa7d74b723c9a9cfeaeb6ee9c97589fcba5c58607cc16922c`
- I never change this file. Cleaned data goes to `data/processed/` (it is not in git).

## My computer

| Item | Value |
|---|---|
| Operating system | Windows 11 (10.0.26300) |
| Processor | AMD (AMD64 Family 25 Model 68 Stepping 1), 16 logical cores |
| RAM | 17.0 GB |
| Python | 3.14.7 |
| Java (for Spark) | OpenJDK 17.0.20.1 (Temurin) |
| Main libraries | pandas 2.3.3, numpy 2.5.2, pyarrow 24.0.0 |

All other library versions are in `requirements.txt`.

## How to set up

```
git clone <link of this repository>
cd <repository folder>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Git Bash use `source .venv/Scripts/activate` instead.

## How to run

Run these from the main folder, in this order:

| Step | Command | What it does |
|---|---|---|
| 1 | `pytest -v` | runs the tests (Task 1) |
| 2 | `python scripts/data_summary.py` | checks the csv and its checksum (Task 3, Obtain) |
| 3 | `python scripts/ds_demo.py` | times a list against a set (Task 1) |
| 4 | `python scripts/storage_formats.py` | compares file formats (Task 2). It takes about 10 minutes. |
| 5 | `python scripts/osemn.py` | cleans the data, makes tables, figures and the model (Task 3) |
| 6 | `python scripts/spark.py` | pandas against Spark (Task 4). It needs Java and the file made in step 5. |

## Where the results are

All results are saved in `results/` (csv and json files) and `figures/`.

| Task | Main files |
|---|---|
| 1 | output of `pytest -v`, `results/ds_choice_demo.txt` |
| 2 | `results/storage_benchmark.csv`, `results/yaml_vs_csv_small.csv`, `figures/fig_storage_formats.png` |
| 3 | `results/scrub_report.csv`, `results/key_findings.json`, `results/model_results.csv`, `figures/fig1` to `fig4` |
| 4 | `results/spark_vs_pandas.csv`, `figures/fig5_spark_vs_pandas.png` |
| 5 | `results/git_history.txt` |

## Numbers to check

These are the same on every computer. If you get them, the set up is right.

- Rows in the csv: 398,233
- Blank `arr_del15` set to 0 (every flight was cancelled or diverted): 293 rows
- Rows with all numbers missing: 657
- Rows left out of the analysis: 659
- Arrived flights that were 15 minutes or more late: 19.6%

The times (Task 2 and Task 4) are different on every computer. The model errors change a little with the library versions.

## Folders

```
preprocess/   the package (loaders, cleaning steps, Cleaner, DelayAnalyzer)
tests/        unit tests
scripts/      one script for each task
data/         the csv and MANIFEST.txt
results/      csv and json results
figures/      figures
```

## Problems I had

- "the raw csv does not match the checksum": the csv was changed. Copy the original file back and do not save it in Excel.
- "Java not found": install Java 17 and open a new terminal.
- A script cannot find `preprocess`: run it from the main folder, not from inside `scripts/`.
