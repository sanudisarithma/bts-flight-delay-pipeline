class FlightDataError(Exception):
    """Parent of all the errors in this package."""


class SchemaError(FlightDataError):


    def __init__(self, missing):
        self.missing = sorted(missing)
        FlightDataError.__init__(self, "missing columns: " + ", ".join(self.missing))


class CleaningError(FlightDataError):
    """A cleaning step failed (wrong order, bad data, ...).

    When Cleaner.run() raises this, the input DataFrame is unchanged
    and the old report is kept.
    """
