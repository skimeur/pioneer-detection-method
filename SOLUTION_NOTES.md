# Reference solution — PDM inflation exercise

This is the accepted reference solution, merged from **PR #29** (author **@medkamel16-star**), the strongest
of the cohort: it completes Parts A.1, A.2, B.1 and B.2, uses all 8 `pdm.py` methods, and produces the four
tables and six figures with a clear discussion (including the Part B.2d caveat that the PDM is not a forecaster).

Two small instructor improvements were applied when merging:

1. **Self-computing discussion.** The written discussion now derives its figures (global RMSE and best method,
   the no-pioneer share, the sample size, and each subperiod's coverage) from the actual results, so the prose
   always matches the script's own tables. (The original draft had been written against an earlier data vintage.)
2. **Offline, deterministic by default.** `python exercise_pdm_inflation.py` now builds the panel from the
   committed CSVs, so it reproduces the committed tables/figures exactly on every run, with no network. Set
   `PDM_REFRESH=1` to instead re-fetch live from the ECB/SSSU SDMX APIs (note: those sources are revised over
   time, which shifts the numbers).

Run: `python exercise_pdm_inflation.py`  (outputs: `figure1..6_*.png`, `table1..4_*.csv`).
