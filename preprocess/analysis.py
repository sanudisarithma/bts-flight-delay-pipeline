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
    """Answers questions like "which carriers are late most often?" on cleaned data.

    Only rows with Usable == True are used. The rest are counted in excluded_rows.

    The delay rate is the share of ARRIVED flights that were 15+ minutes late:
    sum(arr_del15) / sum(arr_arrived), where arr_arrived = flights - cancelled - diverted.
    It is always sum / sum for a group (not the average of the row rates), so a big airport
    counts more than a tiny one.

    Contract: df must come from Cleaner.run() with the default steps.
    Raises ValueError if df is not a DataFrame or Usable is not True/False,
    SchemaError if a needed column is missing.
    """

    NEEDED = ("Usable", "year", "month", "carrier", "carrier_label", "airport", "airport_label",
              "arr_flights", "arr_arrived", "arr_del15", "arr_cancelled", "arr_delay") + CAUSE_MINUTE_COLUMNS

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
        """Rows in the cleaned data (usable and not usable)."""
        return self._total_rows

    @property
    def usable_rows(self):
        """Rows used in the analysis."""
        return len(self._rows)

    @property
    def excluded_rows(self):
        """Rows left out because they are not usable."""
        return self._total_rows - len(self._rows)

    def _select(self, first_year):
        if first_year is None:
            return self._rows
        if not isinstance(first_year, int):
            raise ValueError("first_year must be an int or None")
        return self._rows[self._rows["year"] >= first_year]

    def get_rows(self, columns, first_year=None):
        """A copy of some columns of the usable rows (optionally from first_year on)."""
        columns = list(columns)
        missing = set(columns) - set(self._rows.columns)
        if len(missing) > 0:
            raise SchemaError(missing)
        return self._select(first_year)[columns].copy()

    def months_per_year(self):
        """Number of different months we have usable rows for, per year."""
        return self._rows.groupby("year")["month"].nunique()

    def last_full_year(self):
        """The latest year that has all 12 months."""
        counts = self.months_per_year()
        full = counts[counts == 12]
        if len(full) == 0:
            raise ValueError("no year in the data has all 12 months")
        return int(full.index.max())

    def overall_delay_rate(self, first_year=None):
        """Share of arrived flights that were 15+ minutes late."""
        rows = self._select(first_year)
        arrived = rows["arr_arrived"].sum()
        if arrived <= 0:
            raise ValueError("no flights arrived in the selected rows")
        return float(rows["arr_del15"].sum() / arrived)

    def overall_cancel_rate(self, first_year=None):
        """Share of scheduled flights (arr_flights) that were cancelled."""
        rows = self._select(first_year)
        scheduled = rows["arr_flights"].sum()
        if scheduled <= 0:
            raise ValueError("no flights in the selected rows")
        return float(rows["arr_cancelled"].sum() / scheduled)

    def rate_by(self, column, min_flights=1000, first_year=None):
        """Delay rate for every value of `column`, worst first.

        Groups with fewer than min_flights ARRIVED flights are left out because their rates are too noisy.
        """
        if column not in self._rows.columns:
            raise ValueError("unknown column: " + str(column))
        if not isinstance(min_flights, int) or min_flights < 0:
            raise ValueError("min_flights must be a whole number >= 0")
        rows = self._select(first_year)
        g = rows.groupby(column)[["arr_flights", "arr_arrived", "arr_del15", "arr_delay", "arr_cancelled"]].sum()
        g = g[g["arr_arrived"] >= max(min_flights, 1)]      # also avoids dividing by zero
        g["delay_rate"] = g["arr_del15"] / g["arr_arrived"]
        g["minutes_per_late_flight"] = (g["arr_delay"] / g["arr_del15"]).replace([np.inf, -np.inf], np.nan)
        g["cancelled_per_100_flights"] = 100 * g["arr_cancelled"] / g["arr_flights"]   # per 100 scheduled flights
        return g.sort_values("delay_rate", ascending=False)

    def cause_share(self, first_year=None):
        """Share of all delay minutes that each cause is responsible for."""
        rows = self._select(first_year)
        totals = rows[list(CAUSE_MINUTE_COLUMNS)].sum()
        return totals / totals.sum()

    def cause_share_by_year(self):
        """Share of delay minutes per cause for every year (each row adds up to 1)."""
        per_year = self._rows.groupby("year")[list(CAUSE_MINUTE_COLUMNS)].sum()
        return per_year.div(per_year.sum(axis=1), axis=0)

    def names_for(self, code_column, name_column):
        """A dict code -> name (one name per code after cleaning)."""
        pairs = self._rows[[code_column, name_column]].drop_duplicates(subset=code_column)
        return dict(zip(pairs[code_column], pairs[name_column]))
