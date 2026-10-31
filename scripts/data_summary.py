
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import config
from preprocess import CsvLoader, SchemaError
from preprocess.utils import sha256_of


def main():
    config.make_folders()
    print("raw file:", config.RAW_CSV)

    digest = sha256_of(config.RAW_CSV)
    size_mb = os.path.getsize(config.RAW_CSV) / 1e6
    print("size: %.1f MB" % size_mb)
    print("sha256:", digest)

    if os.path.isfile(config.MANIFEST):
        with open(config.MANIFEST, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if digest in text:
            print("checksum is in MANIFEST.txt -> raw file is unchanged")
        else:
            print("WARNING: checksum is NOT in MANIFEST.txt -> the raw file has changed!")
    else:
        print("no data/MANIFEST.txt yet - create it and paste the checksum, source URL and download date")

    try:
        raw = CsvLoader(config.RAW_CSV).load()
    except SchemaError as e:
        print("The file does not have the columns the package expects.")
        print("missing:", e.missing)
        print("Send me the header line of your csv and I will change REQUIRED_COLUMNS.")
        return

    print("rows: {:,}   columns: {}".format(len(raw), raw.shape[1]))
    if len(raw) < 100000:
        print("NOTE: fewer than 100,000 rows. The brief wants a written justification for this.")
    print("years: {} to {}".format(int(raw["year"].min()), int(raw["year"].max())))
    print("carriers: {}   airports: {}".format(raw["carrier"].nunique(), raw["airport"].nunique()))
    print("column types:")
    print(raw.dtypes.to_string())

    summary = {
        "file": os.path.basename(config.RAW_CSV),
        "sha256": digest,
        "size_mb": round(size_mb, 1),
        "rows": len(raw),
        "columns": int(raw.shape[1]),
        "first_year": int(raw["year"].min()),
        "last_year": int(raw["year"].max()),
        "carriers": int(raw["carrier"].nunique()),
        "airports": int(raw["airport"].nunique()),
    }
    with open(os.path.join(config.RESULTS_DIR, "obtain_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved results/obtain_summary.json")


if __name__ == "__main__":
    main()
