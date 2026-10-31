import time

import pandas as pd

from .exceptions import CleaningError, FlightDataError, SchemaError

TEXT_COLUMNS = ("carrier", "carrier_name", "airport", "airport_name")
CODE_COLUMNS = ("carrier", "airport")

COUNT_COLUMNS = ("arr_flights", "arr_del15", "carrier_ct", "weather_ct", "nas_ct",
                 "security_ct", "late_aircraft_ct", "arr_cancelled", "arr_diverted")
MINUTE_COLUMNS = ("arr_delay", "carrier_delay", "weather_delay", "nas_delay",
                  "security_delay", "late_aircraft_delay")
NUMBER_COLUMNS = COUNT_COLUMNS + MINUTE_COLUMNS

CAUSE_COUNT_COLUMNS = ("carrier_ct", "weather_ct", "nas_ct", "security_ct", "late_aircraft_ct")
CAUSE_MINUTE_COLUMNS = ("carrier_delay", "weather_delay", "nas_delay", "security_delay",
                        "late_aircraft_delay")

KEY_COLUMNS = ("year", "month", "carrier", "airport")   # one row per carrier + airport + month

COUNT_TOLERANCE = 0.5      # cause counts are rounded to 2 decimals, so a gap of half a flight is a real error
MINUTE_TOLERANCE = 1.0     # allowed gap (minutes) when the cause minutes are added up
FIRST_YEAR = 2000          # years outside this range are treated as impossible
LAST_YEAR = 2030


class CleaningStep:
    
    name = "base step"
    needs = ()

    def __init__(self):
        self._affected = 0
        self._note = ""

    @property
    def affected(self):
        return self._affected

    @property
    def note(self):
        return self._note

    def check_columns(self, df):
        missing = set(self.needs) - set(df.columns)
        if len(missing) > 0:
            raise SchemaError(missing)

    def apply(self, df):
        raise NotImplementedError("each step must write its own apply()")

    def summary(self):
        return {"step": self.name, "records_affected": int(self._affected), "note": self._note}


class TidyText(CleaningStep):
    name = "tidy text columns"
    needs = TEXT_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        changed_rows = pd.Series(False, index=out.index)
        details = []
        for col in TEXT_COLUMNS:
            before = out[col].fillna("").astype(str)
            after = before.str.strip().str.replace(r"\s+", " ", regex=True)
            if col in CODE_COLUMNS:
                after = after.str.upper()
            changed = before != after
            changed_rows = changed_rows | changed
            details.append(col + "=" + str(int(changed.sum())))
            out[col] = after
        self._affected = int(changed_rows.sum())
        self._note = "rows changed (spaces / upper case): " + ", ".join(details)
        return out


class ConvertNumbers(CleaningStep):
    name = "convert number columns"
    needs = ("year", "month") + NUMBER_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        new_missing = 0
        for col in self.needs:
            was_missing = out[col].isna()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            now_missing = out[col].isna()
            new_missing = new_missing + int((now_missing & ~was_missing).sum())
        self._affected = new_missing
        self._note = "values that were not numbers and are now missing: " + str(new_missing)
        return out


class RemoveExactDuplicates(CleaningStep):
    name = "remove exact duplicates"

    def apply(self, df):
        is_copy = df.duplicated(keep="first")
        self._affected = int(is_copy.sum())
        self._note = "rows identical to an earlier row, removed"
        return df[~is_copy].copy()


class FillBlankLateCount(CleaningStep):
    name = "fill blank arr_del15 when no flight arrived"
    needs = NUMBER_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        other_numbers = [c for c in NUMBER_COLUMNS if c != "arr_del15"]
        others_present = out[other_numbers].notna().all(axis=1)
        nothing_arrived = (out["arr_cancelled"] + out["arr_diverted"]) == out["arr_flights"]
        blank = out["arr_del15"].isna()
        fill = blank & others_present & nothing_arrived
        out.loc[fill, "arr_del15"] = 0.0
        self._affected = int(fill.sum())
        self._note = ("arr_del15 blank because every flight was cancelled/diverted, set to 0: " + str(int(fill.sum()))
                      + "; arr_del15 blank for another reason (left blank): " + str(int((blank & ~fill).sum())))
        return out


class AddNameLabels(CleaningStep):
    name = "add one name label per carrier / airport code"
    needs = ("year", "month", "carrier", "carrier_name", "airport", "airport_name")

    def _label(self, out, code_col, name_col, label_col):
        ok = (out[code_col] != "") & (out[name_col] != "") & out["year"].notna() & out["month"].notna()
        known = out[ok]
        many = int((known.groupby(code_col)[name_col].nunique() > 1).sum())

        known = known.sort_values(["year", "month", name_col])
        latest_rows = known.drop_duplicates(subset=code_col, keep="last")
        latest = dict(zip(latest_rows[code_col], latest_rows[name_col]))

        label = out[code_col].map(latest)
        label = label.fillna(out[name_col])   # keep the row's own name if the code has no latest name
        differs = int((label != out[name_col]).sum())
        out[label_col] = label
        return many, differs

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        many_c, diff_c = self._label(out, "carrier", "carrier_name", "carrier_label")
        many_a, diff_a = self._label(out, "airport", "airport_name", "airport_label")
        self._affected = diff_c + diff_a
        self._note = ("carrier codes with more than one name: " + str(many_c) + " (rows whose label differs: " + str(diff_c)
                      + "); airport codes with more than one name: " + str(many_a)
                      + " (rows whose label differs: " + str(diff_a) + ")")
        return out


class MakeDate(CleaningStep):
    
    name = "make date column"
    needs = ("year", "month")

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        parts = pd.DataFrame({"year": out["year"], "month": out["month"], "day": 1})
        out["date"] = pd.to_datetime(parts, errors="coerce")
        bad = int(out["date"].isna().sum())
        self._affected = bad
        self._note = "rows where year/month do not make a valid date: " + str(bad)
        return out


class AddArrivedFlights(CleaningStep):
    
    name = "add arrived flights"
    needs = ("arr_flights", "arr_cancelled", "arr_diverted")

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        out["arr_arrived"] = out["arr_flights"] - out["arr_cancelled"] - out["arr_diverted"]
        some_not_arrived = (out["arr_cancelled"] + out["arr_diverted"]) > 0
        none_arrived = out["arr_arrived"] == 0
        self._affected = int(some_not_arrived.sum())
        self._note = ("rows with at least one cancelled or diverted flight: " + str(int(some_not_arrived.sum()))
                      + "; rows where no flight arrived: " + str(int(none_arrived.sum())))
        return out


class FlagMissing(CleaningStep):
    
    name = "flag missing values"
    needs = ("year", "month", "carrier", "airport") + NUMBER_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        number_cols = ["year", "month"] + list(NUMBER_COLUMNS)
        missing_number = out[number_cols].isna().any(axis=1)
        missing_flights = out["arr_flights"].isna()
        missing_code = (out["carrier"] == "") | (out["airport"] == "")
        out["MissingValues"] = missing_number | missing_code
        self._affected = int(out["MissingValues"].sum())
        self._note = ("arr_flights missing: " + str(int(missing_flights.sum()))
                      + "; any number missing: " + str(int(missing_number.sum()))
                      + "; carrier or airport code missing: " + str(int(missing_code.sum())))
        return out


class FlagDuplicateKeys(CleaningStep):


    name = "flag duplicate keys"
    needs = KEY_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        out["DuplicateKey"] = out.duplicated(subset=list(KEY_COLUMNS), keep=False)
        self._affected = int(out["DuplicateKey"].sum())
        self._note = "rows sharing year+month+carrier+airport with another row (all copies flagged)"
        return out


class FlagImpossible(CleaningStep):
    
    name = "flag impossible values"
    needs = ("year", "month") + NUMBER_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        negative = (out[list(NUMBER_COLUMNS)] < 0).any(axis=1)
        # late + cancelled + diverted flights can not be more than all the flights
        too_many = (out["arr_del15"] + out["arr_cancelled"] + out["arr_diverted"]) > out["arr_flights"]
        bad_month = (out["month"] < 1) | (out["month"] > 12)
        bad_year = (out["year"] < FIRST_YEAR) | (out["year"] > LAST_YEAR)
        out["Impossible"] = negative | too_many | bad_month | bad_year
        self._affected = int(out["Impossible"].sum())
        self._note = ("negative number: " + str(int(negative.sum()))
                      + "; late + cancelled + diverted > arr_flights: " + str(int(too_many.sum()))
                      + "; month not 1-12: " + str(int(bad_month.sum()))
                      + "; year outside " + str(FIRST_YEAR) + "-" + str(LAST_YEAR) + ": " + str(int(bad_year.sum())))
        return out


class FlagCauseMismatch(CleaningStep):

    name = "flag cause totals that do not add up"
    needs = ("arr_del15", "arr_delay") + CAUSE_COUNT_COLUMNS + CAUSE_MINUTE_COLUMNS

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        # min_count means: if any of the five is missing the sum is missing, so no flag
        count_sum = out[list(CAUSE_COUNT_COLUMNS)].sum(axis=1, min_count=len(CAUSE_COUNT_COLUMNS))
        minute_sum = out[list(CAUSE_MINUTE_COLUMNS)].sum(axis=1, min_count=len(CAUSE_MINUTE_COLUMNS))
        count_bad = (count_sum - out["arr_del15"]).abs() > COUNT_TOLERANCE
        minute_bad = (minute_sum - out["arr_delay"]).abs() > MINUTE_TOLERANCE
        out["CauseMismatch"] = count_bad | minute_bad
        self._affected = int(out["CauseMismatch"].sum())
        self._note = ("cause counts != arr_del15: " + str(int(count_bad.sum()))
                      + "; cause minutes != arr_delay: " + str(int(minute_bad.sum())))
        return out


class MarkUsable(CleaningStep):
    

    name = "mark usable rows"
    needs = ("arr_flights", "MissingValues", "DuplicateKey", "Impossible")

    def apply(self, df):
        self.check_columns(df)
        out = df.copy()
        flagged = out["MissingValues"] | out["DuplicateKey"] | out["Impossible"]
        zero_flights = out["arr_flights"] <= 0
        out["Usable"] = ~(flagged | zero_flights)
        not_usable = int((~out["Usable"]).sum())
        only_zero = int((zero_flights & ~flagged).sum())
        self._affected = not_usable
        self._note = ("rows left out of the analysis: " + str(not_usable)
                      + " (of these, only because of zero flights: " + str(only_zero) + ")")
        return out


def default_steps():
    return [
        TidyText(),
        ConvertNumbers(),
        RemoveExactDuplicates(),
        FillBlankLateCount(),
        AddNameLabels(),
        MakeDate(),
        AddArrivedFlights(),
        FlagMissing(),
        FlagDuplicateKeys(),
        FlagImpossible(),
        FlagCauseMismatch(),
        MarkUsable(),
    ]


class Cleaner:
    
    def __init__(self, steps):
        if not isinstance(steps, (list, tuple)) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list of CleaningStep objects")
        for s in steps:
            if not isinstance(s, CleaningStep):
                raise ValueError(repr(s) + " is not a CleaningStep")
        self._steps = list(steps)   # a list, because the order of the steps matters
        self._report = []
        self._seconds = 0.0

    @property
    def report(self):
        return pd.DataFrame(self._report)

    @property
    def seconds(self):
        return self._seconds

    def run(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("run() needs a pandas DataFrame, got " + type(df).__name__)
        if df.empty:
            raise ValueError("the DataFrame is empty, nothing to clean")

        new_report = []
        current = df
        step_name = "(no step yet)"
        start = time.time()
        try:
            for step in self._steps:
                step_name = step.name
                rows_before = len(current)
                current = step.apply(current)
                info = step.summary()
                info["rows_before"] = rows_before
                info["rows_after"] = len(current)
                new_report.append(info)
        except FlightDataError:
            raise   # already one of our own errors with a good message
        except Exception as e:
            raise CleaningError("step '" + step_name + "' failed: " + str(e)) from e
        else:
            self._report = new_report   # only saved if every step worked
        finally:
            self._seconds = round(time.time() - start, 2)
        return current
