"""Behaviour tests using small synthetic inputs, independent of the BTS download."""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from preprocess.loaders import CsvLoader, REQUIRED_COLUMNS
from preprocess.cleaning import Cleaner, CleaningStep, TidyText, default_steps
from preprocess.exceptions import SchemaError, CleaningError


@pytest.fixture
def raw():
    row = dict.fromkeys(REQUIRED_COLUMNS, 0)
    row.update(year=2024, month=1, carrier=' aa ', carrier_name=' American  Airlines ',
               airport=' jfk ', airport_name=' New York ', arr_flights=100,
               arr_del15=10, carrier_ct=10, arr_delay=200, carrier_delay=200)
    return pd.DataFrame([row])


def test_csv_loads_values_and_records_row_count(tmp_path, raw):
    path = tmp_path / 'valid.csv'
    raw.to_csv(path, index=False)
    loader = CsvLoader(str(path))
    assert_frame_equal(loader.load(), raw)
    assert loader.rows_loaded == 1


def test_schema_failure_preserves_last_successful_count(tmp_path, raw):
    path = tmp_path / 'input.csv'
    raw.to_csv(path, index=False)
    loader = CsvLoader(str(path))
    loader.load()
    raw.drop(columns=['airport', 'arr_flights']).to_csv(path, index=False)
    with pytest.raises(SchemaError) as error:
        loader.load()
    assert error.value.missing == ['airport', 'arr_flights']
    assert loader.rows_loaded == 1


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        CsvLoader(str(tmp_path / 'missing.csv')).load()


@pytest.mark.parametrize('path', ['', None, 'input.json'])
def test_invalid_loader_path_raises_value_error(path):
    with pytest.raises(ValueError):
        CsvLoader(path)


@pytest.mark.parametrize('data', [None, [], pd.DataFrame()])
def test_cleaner_rejects_invalid_input(data):
    with pytest.raises(ValueError):
        Cleaner([TidyText()]).run(data)


def test_default_pipeline_cleans_without_mutating_input(raw):
    original = raw.copy(deep=True)
    cleaner = Cleaner(default_steps())
    output = cleaner.run(raw)
    assert_frame_equal(raw, original)
    assert output.loc[0, 'carrier'] == 'AA'
    assert output.loc[0, 'airport'] == 'JFK'
    assert output.loc[0, 'carrier_name'] == 'American Airlines'
    assert output.loc[0, 'arr_arrived'] == 100
    assert bool(output.loc[0, 'Usable'])
    assert cleaner.report.iloc[-1]['rows_after'] == 1


def test_negative_counts_are_retained_but_excluded(raw):
    raw.loc[0, 'arr_flights'] = -1
    output = Cleaner(default_steps()).run(raw)
    assert len(output) == 1
    assert bool(output.loc[0, 'Impossible'])
    assert not bool(output.loc[0, 'Usable'])


def test_domain_error_is_preserved():
    with pytest.raises(SchemaError) as error:
        Cleaner([TidyText()]).run(pd.DataFrame({'carrier': ['AA']}))
    assert error.value.missing == ['airport', 'airport_name', 'carrier_name']


def test_failed_step_preserves_input_report_and_updates_elapsed(raw, monkeypatch):
    class FailableStep(CleaningStep):
        name = 'controlled failure'
        fail = False

        def apply(self, frame):
            if self.fail:
                raise RuntimeError('simulated failure')
            return frame.copy()

    step = FailableStep()
    cleaner = Cleaner([TidyText(), step])
    cleaner.run(raw)
    old_report = cleaner.report.copy(deep=True)
    original = raw.copy(deep=True)
    step.fail = True
    clock = iter([100.0, 102.5])
    monkeypatch.setattr('preprocess.cleaning.time.time', lambda: next(clock))
    with pytest.raises(CleaningError, match='controlled failure') as error:
        cleaner.run(raw)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert_frame_equal(raw, original)
    assert_frame_equal(cleaner.report, old_report)
    assert cleaner.seconds == 2.5


def test_report_return_value_cannot_change_saved_report(raw):
    cleaner = Cleaner([TidyText()])
    cleaner.run(raw)
    exposed = cleaner.report
    exposed.loc[0, 'records_affected'] = 999
    assert cleaner.report.loc[0, 'records_affected'] == 1


def test_recent_window_excludes_partial_future_year(raw):
    from preprocess.analysis import DelayAnalyzer
    from scripts.osemn import recent_period_analyzer
    frames = []
    for year, late in [(2021, 1), (2022, 10), (2024, 20), (2025, 90)]:
        frame = raw.copy()
        frame["year"] = year
        frame["arr_del15"] = late
        frames.append(frame)
    clean = Cleaner(default_steps()).run(pd.concat(frames, ignore_index=True))
    recent = recent_period_analyzer(DelayAnalyzer(clean), 2022, 2024)
    assert recent.get_rows(["year"])["year"].tolist() == [2022, 2024]
    assert recent.overall_delay_rate() == pytest.approx(0.15)
    assert recent.rate_by("carrier", min_flights=0).loc["AA", "delay_rate"] == pytest.approx(0.15)
    assert recent.rate_by("airport", min_flights=0).loc["JFK", "delay_rate"] == pytest.approx(0.15)
