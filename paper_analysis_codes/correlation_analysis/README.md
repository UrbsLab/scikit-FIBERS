# ASHI correlation analysis

Run three Python submission files in order. All analysis runs in LSF jobs; the
submission files only read the small configuration and call `bsub`.

```text
main_fibers.py       -> bsub run_fibers.py --imputation I --fold F --seed S
main_correlation.py  -> bsub run_correlation.py --imputation I --fold F
main_plots.py        -> bsub run_plots.py
```

There are no manifests, generated shell scripts, numbered stages, installation
steps, or analysis `src` package. `common.py`, `data.py`, `methods.py` and
`plotting.py` are ordinary neighboring modules shared by these workers. FIBERS
itself is imported from this checkout's existing `src/skfibers` directory.

## Run on LSF

Start in `SIMPLE/paper_analysis_codes/correlation_analysis`, activate your existing
conda environment, and review `config.json`. It contains the input paths, `TX_ID`,
clinical/antigen covariates, FIBERS settings, correlation cutoffs and LSF resources.
The default runs imputation 1, folds 1-10, seed 1, and writes to
`/project/kamoun_shared/output_shared/ashi_2026/simple`.

```bash
conda activate scikit-fibers
python main_fibers.py --config config.json
```

Wait until every FIBERS job finishes successfully. Then:

```bash
conda activate scikit-fibers
python main_correlation.py --config config.json
```

Wait until every correlation job finishes successfully. Then:

```bash
conda activate scikit-fibers
python main_plots.py --config config.json
```

Each command returns after submitting its jobs. There are no blocking preparation
jobs or scheduler dependencies. `bjobs -u "$USER"` shows status; logs are in
`<output_root>/logs`. The default submits 10 fit jobs, 10 correlation jobs, then
one plotting job. Completion of a submission command does not mean the jobs have
finished. Starting the next stage too early produces a missing-results error.

Use `--dry-run` on any main file to print its exact `bsub` commands. `--queue`,
`--memory-gb`, `--cpus` and `--hours` override that stage's resources. To retry
selected jobs, use `--imputations 1 --folds 3 7` and, for FIBERS, `--seeds 1`.
These select a subset of the tasks already listed in the config. Completed,
matching results are skipped. `--force` reruns selected tasks; rerun their
downstream correlation and plot jobs afterward. Wait for active jobs before
changing settings or repeating a submission for the same task.

Workers use the absolute Python executable that submitted them, so the conda
environment must already contain FIBERS' scientific dependencies. No editable
installation is needed. The login node never scans patient CSV files.

## Analysis choices

- Training and held-out test files are selected explicitly from templates; every
  wildcard must match exactly one file. Compute jobs check `TX_ID` uniqueness and
  train/test disjointness, required columns, finite numbers, 0/1 events and 0/1/2
  mismatch counts. The plot job checks that held-out folds partition each cohort
  exactly once and that cohort IDs match across imputations before summarizing.
- The seven study loci and position ranges match the prior analysis: A/B/C
  positions 1-182, DRB1/DRB345/DQA1 positions 6-94, DQB1 positions 6-95. DPA1/DPB1
  are not included. The actual input header supplies 903 candidate positions.
  Constants and configured rare features are removed using training data only.
- FIBERS uses the supplied study settings, including product fitness
  (`log_rank_residuals`), 100 iterations, population 50, fixed threshold 0 and
  seed 1. Product fitness uses clinical-plus-antigen-adjusted training residuals.
- Each correlation job reads its training fold once, computes all pairwise
  Pearson coefficients with chunked matrix multiplication, and reuses them for
  all seeds and cutoffs. All 21 interlocus pairs for the seven loci are retained,
  including negative correlations. No Python process pool or local parallel
  jobs are launched; numerical libraries use the CPU slots allocated by LSF.
- Within-locus blocks use complete linkage and require every pair to satisfy
  `r > cutoff`. The hierarchy is built once per locus and reused. Interlocus
  correlations are descriptive; they do not merge loci in processed bins.
- Processing includes every member of a touched block. A block contributes its
  maximum mismatch value once, even if several original features touch it.
  With 0/1/2 inputs the contribution can be 2; it is not automatically binary.
- Original and processed bins are compared at the original threshold. A separate
  sensitivity analysis reselects the threshold for the top bin using training
  log-rank separation only. All test-set evaluation holds that choice fixed.
- Every bin gets train/test log-rank results. By default the top bin gets
  unadjusted HR, Adj HR (clinical plus antigen covariates), and Adj NoAg HR
  (clinical covariates only) for every scheme. All bins also get unadjusted test
  HR for the original and primary r > 0.95 schemes. Increase the `evaluation`
  top-bin counts to fit more Cox models. Cox warnings/failures are reported in
  status columns, never replaced with a successful-looking estimate. Residuals
  are used in training fitness, not as outcome-derived predictors in test Cox
  models. HR refers to above-threshold versus at/below-threshold patients.
- The exploratory adaptive scheme selects the highest candidate threshold per
  locus that provides at least one block and four positions in multi-position
  blocks; if none qualifies it uses the primary cutoff. This is a transparent
  structural heuristic, not an optimized clinical cutoff. `locus_thresholds`
  can instead specify an explicit mapping, for example `{"DRB1": 0.4}`.
- Consistency is pairwise Jaccard feature overlap within populations, across CV
  folds, and across imputations at the same fold/seed. Summaries are descriptive;
  overlapping training folds and imputations are not independent replications.
  Mismatch correlations do not directly estimate genetic linkage disequilibrium.

## Additional imputations

The default uses only imputation 1 because only its pre-split files are available.
To use more already-split imputations, add them to `imputations` and provide
matching templates. To create CV comparisons directly from all full datasets,
set `input.mode` to `full_dataset` and list those imputations. Each compute job
forms its train/test split in memory: IDs are sorted, shuffled with `split_seed`,
and divided into K balanced folds. Identical IDs therefore receive the same fold
in each imputation regardless of row order. The split does not use outcomes.
This mode rereads each full CSV for each fold; it avoids a separate split stage
or split files. Use a new `output_root` when switching input mode.

## Outputs and figures

```text
<output_root>/
  logs/
  imp_01/cv_01/
    seed_001/
      fibers.pkl                 # fitted upstream model
      population.csv             # full final population, ranked on training
      feature_filter.csv
      top_bin_metrics.csv
      completed.json             # inputs/settings/code fingerprint and checks
    correlation/
      correlations.csv.gz        # full within- and between-locus pairs
      blocks.json
      feature_filter.csv
      split_ids.csv.gz           # local identifier audit, not model features
      metrics.csv.gz             # original/fixed/reoptimized train/test metrics
      membership.csv.gz          # original versus added positions by bin
      processed_bins.json        # scoring sublists and chosen thresholds
      completed.json
  summary/
    cv_validation.csv
    risk_comparison.csv.gz
    consistency_pairs.csv.gz
    consistency_summary.csv
    interlocus_summary.csv.gz
    interlocus_cutoff_counts.csv
    locus_thresholds.csv
    figure_captions.txt
    interpretation.txt
    figures/
    completed.json
```

Figure 1 compares a selected full bin population before/after processing. Figure
2 compares top bins across CV folds. Figure 3 compares top bins across imputations
at the selected common fold and is generated only when multiple imputations are
available. Large feature-inclusion plots are split into readable pages with the
same row and column ordering before/after. Dark cells are original, light cells
are added alternatives, and block colors are explained in the separate captions.
There are no dots, embedded titles or captions.

Figure 4 compares consistency, position counts and paired test HR changes over
the full threshold grid starting at 0.10. Figure 5 provides separate interlocus
heatmaps for locus pairs with eligible correlations. It displays mean r across
all training folds and counts of folds exceeding the configured cutoff, capped
at a configurable number of positions per axis. Full numerical results are
always in the tables. No across-imputation result is invented for a one-imputation
configuration. PNG and PDF versions are produced with matplotlib.

The source checkout is on branch `ashi-correlation`, created from upstream
`main` at `cfa6a51f215b577aaea1f572defd82828c97300d`. Local model code is unchanged.
`completed.json` records source fingerprints so incompatible results cannot be
silently combined. `test_analysis.py` exercises numerical/block checks, input
leakage checks, direct LSF commands and an entire synthetic three-stage run.
