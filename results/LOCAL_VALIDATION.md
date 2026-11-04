# Local validation results

Date: 2026-11-04
Branch: local/integration-validation-20261104
Merged commit: 766530a

All four supplied remote-tracking branches are ancestors of the merge. The main branch was not moved. No push was performed.

- pytest: exit 5; zero tests collected. This is an unresolved test coverage gap.
- ds_demo.py: passed.
- data_summary.py: passed; raw checksum matched MANIFEST.txt.
- osemn.py: passed. Year/month aggregates, scrub report and model result tables match the pre-run branch results exactly.
- spark.py: passed for factors 1, 5 and 25. Its built-in comparison confirmed matching pandas/Spark outputs for 439 groups on the largest input. Shutdown emitted process-termination warnings after results were produced.
- storage_formats.py: first attempt failed with a memory allocation error. A separate full retry passed all nine format/compression combinations and the YAML side test. Current storage tables/charts come from the successful retry.

See local_run_summary.json for exit codes and durations, *_run.txt for output, and storage_first_attempt.txt for the original failure.

Environment: Python 3.14.2, Java 21.0.9, Windows 11, Intel processor, 12 logical cores, 34.0 GB RAM. Direct Python dependencies are pinned in requirements.txt. Full benchmark hardware metadata is in the respective system JSON files.

The Word report was not modified. Its old AMD/16-core environment and timings must not be described as this run. Appendix B3 needs correction from delay_by_year.csv; the CSV rates are valid fractions in [0,1]. The download date and licence in the source provenance remain unverified. This integration does not certify that all coursework requirements are met.

Existing outputs were backed up locally in .local-validation/before-run before execution. No existing source analysis logic was rewritten; merge conflicts retained main's equivalent implementation and documentation.

## Follow-up test validation
On 2026-11-04, tests/test_processing.py was added at the user's request. All 14 collected cases passed in 1.35 seconds. This supersedes the earlier missing-tests status; local_run_summary.json retains the historical initial run. See pytest_output.txt for the current test evidence. Tests use synthetic fixtures and cover loading, input/schema errors, cleaning, input immutability, exclusion flags, failure chaining, report preservation and elapsed-time recording.

## Date-window correction and final document
The recent carrier and airport analysis now restricts data to 2022-2024 inclusive. All 15 tests passed, including a regression case excluding 2025. Analysis outputs were regenerated. The Word document in docs records these corrected results and discloses AI assistance. Student-specific and unverified provenance fields remain to be completed.
