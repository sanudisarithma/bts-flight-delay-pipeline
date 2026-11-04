# Local integration validation

Branch: local/integration-validation-20261104. No remote push was performed.

## Run on Windows

Python 3.14.2 was used for the local validation. Install the pinned direct dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-21.0.9'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:SPARK_LOCAL_IP = '127.0.0.1'
$env:PYTHONIOENCODING = 'utf-8'
.\venv\Scripts\python.exe -m pytest -v
.\venv\Scripts\python.exe scripts/ds_demo.py
.\venv\Scripts\python.exe scripts/storage_formats.py
.\venv\Scripts\python.exe scripts/data_summary.py
.\venv\Scripts\python.exe scripts/osemn.py
.\venv\Scripts\python.exe scripts/spark.py
```

Adjust JAVA_HOME to your installed compatible JDK. Raw input is data/Airline_Delay_Cause.csv. The OSEMN script verifies its checksum against data/MANIFEST.txt and creates data/processed/clean.parquet for Spark.

Dataset download page supplied by the project owner: https://www.kaggle.com/datasets/daryaheyko/airline-on-time-statistics-and-delay-causes-bts . The real download date and licence still need owner verification; do not invent them.

## Evidence and limitations

See results/local_run_summary.json and the corresponding *_run.txt logs for actual execution status. The initial merged branches contained no tests. A subsequent local test suite in tests/test_processing.py now passes 15 cases; see results/pytest_output.txt. The initial run summary remains historical. Existing report values must be compared with regenerated outputs before reuse. Timing results are specific to the machine and installed versions recorded in results/*system.json.

The original results and figures are preserved locally under .local-validation/before-run (excluded from Git). This README documents local integration and execution, not completion of every coursework criterion.
