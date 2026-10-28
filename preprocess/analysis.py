"""Summary numbers on the cleaned data (the output of Cleaner.run)."""
import numpy as np
import pandas as pd

from .cleaning import CAUSE_MINUTE_COLUMNS
from .exceptions import SchemaError

CAUSE_LABELS = {
    "carrier_delay": "Carrier",
    "weather_delay": "Weather",
    "nas_delay": "NAS (air traffic system)",
    "security_delay": "Security",
    "late_aircraft_delay": "Late aircraft",
}


class DelayAnalyzer:


    NEEDED = ("Usable", "year", "month", "carrier", "carrier_name", "airport", "airport_name",
              "arr_flights", "arr_del15", "arr_cancelled", "arr_delay") + CAUSE_MINUTE_COLUMNS

    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DelayAnalyzer needs a pandas DataFrame")
        missing = set(self.NEEDED) - set(df.columns)
        if len(missing) > 0:
            raise SchemaError(missing)
        if df["Usable"].dtype != bool:
            raise ValueError("column Usable must be True/False - run the Cleaner first")
        rows = df[df["Usable"]].copy()
        rows["year"] = rows["year"].astype(int)
        rows["month"] = rows["month"].astype(int)
        self._rows = rows
        self._total_rows = len(df)

    @property
    def total_rows(self):
        return self._total_rows

    @property
    def usable_rows(self):
        return len(self._rows)

    @property
    def excluded_rows(self):
        return self._total_rows - len(self._rows)

    def _select(self, first_year):
        if first_year is None:
            return self._rows
        if not isinstance(first_year, int):
            raise ValueError("first_year must be an int or None")
        return self._rows[self._rows["year"] >= first_year]

    def get_rows(self, columns, first_year=None):
        columns = list(columns)
        missing = set(columns) - set(self._rows.columns)
        if len(missing) > 0:
            raise SchemaError(missing)
        return self._select(first_year)[columns].copy()

    def months_per_year(self):
        return self._rows.groupby("year")["month"].nunique()

    def last_full_year(self):
        
        counts = self.months_per_year()
        full = counts[counts == 12]
        if len(full) == 0:
            raise ValueError("no year in the data has all 12 months")
        return int(full.index.max())

    def overall_delay_rate(self, first_year=None):

        rows = self._select(first_year)
        return float(rows["arr_del15"].sum() / rows["arr_flights"].sum())

    def rate_by(self, column, min_flights=1000, first_year=None):

        if column not in self._rows.columns:
            raise ValueError("unknown column: " + str(column))
        if not isinstance(min_flights, int) or min_flights < 0:
            raise ValueError("min_flights must be a whole number >= 0")
        rows = self._select(first_year)
        g = rows.groupby(column)[["arr_flights", "arr_del15", "arr_delay", "arr_cancelled"]].sum()
        g["delay_rate"] = g["arr_del15"] / g["arr_flights"]
        g["minutes_per_late_flight"] = (g["arr_delay"] / g["arr_del15"]).replace([np.inf, -np.inf], np.nan)
        g["cancelled_per_100_flights"] = 100 * g["arr_cancelled"] / g["arr_flights"]
        g = g[g["arr_flights"] >= min_flights]
        return g.sort_values("delay_rate", ascending=False)

    def cause_share(self, first_year=None):
        rows = self._select(first_year)
        totals = rows[list(CAUSE_MINUTE_COLUMNS)].sum()
        return totals / totals.sum()

    def cause_share_by_year(self):
        per_year = self._rows.groupby("year")[list(CAUSE_MINUTE_COLUMNS)].sum()
        return per_year.div(per_year.sum(axis=1), axis=0)

    def names_for(self, code_column, name_column):
        pairs = self._rows[[code_column, name_column]].drop_duplicates(subset=code_column)
        return dict(zip(pairs[code_column], pairs[name_column]))
