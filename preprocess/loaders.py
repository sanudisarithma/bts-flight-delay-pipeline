import os

import pandas as pd

from .exceptions import SchemaError

REQUIRED_COLUMNS = (
    "year", "month", "carrier", "carrier_name", "airport", "airport_name",
    "arr_flights", "arr_del15",
    "carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct",
    "arr_cancelled", "arr_diverted", "arr_delay",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
)


class BaseLoader:

    extension = ""   # the child classes set this

    def __init__(self, path):
        if not isinstance(path, str) or path.strip() == "":
            raise ValueError("path must be a non-empty string")
        if self.extension != "" and not path.lower().endswith(self.extension):
            raise ValueError(type(self).__name__ + " needs a " + self.extension + " file, got: " + path)
        self._path = path
        self._rows_loaded = 0

    @property
    def path(self):
        """The file this loader reads."""
        return self._path

    @property
    def rows_loaded(self):
        """Rows in the last successful load() (0 if load() has not worked yet)."""
        return self._rows_loaded

    def load(self):
        """Read the file, tidy the column names and check the columns. Returns a DataFrame."""
        if not os.path.isfile(self._path):
            raise FileNotFoundError("file not found: " + self._path)
        df = self._read()
        df.columns = [str(c).strip().lower() for c in df.columns]
        keep = [c for c in df.columns if not c.startswith("unnamed")]
        df = df[keep]
        missing = set(REQUIRED_COLUMNS) - set(df.columns)   # set difference is quick and simple
        if len(missing) > 0:
            raise SchemaError(missing)
        self._rows_loaded = len(df)   # only set when everything worked
        return df

    def _read(self):
        raise NotImplementedError("child classes must write their own _read()")

    def __repr__(self):
        return type(self).__name__ + "(" + repr(self._path) + ")"


class CsvLoader(BaseLoader):

    extension = ".csv"

    def _read(self):
        try:
            return pd.read_csv(self._path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            # not utf-8, so try the older latin-1 encoding instead
            return pd.read_csv(self._path, encoding="latin-1", low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            raise ValueError("cannot read csv file " + self._path + ": " + str(e)) from e


class ParquetLoader(BaseLoader):
    extension = ".parquet"

    def _read(self):
        return pd.read_parquet(self._path)
