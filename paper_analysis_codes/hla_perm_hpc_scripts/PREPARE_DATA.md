Run from the repository root with Python 3.9 or later; no additional packages
are needed for preparation:

```bash
python3 paper_analysis_codes/hla_perm_hpc_scripts/job_prepare_fibers_data.py \
  --d /path/to/Best-Typing_AA_MM_9loc_matrix_1_aligned.tsv.gz \
  --o data/prepared/BestTyping_1.csv
```

Replace the input path with the complete combined best-typing file to prepare
the full dataset. Inputs may be `.tsv`, `.txt`, `.csv`, `.tsv.gz`, `.txt.gz`,
`.csv.gz`, plain `.gz`, or extensionless numbered files. Gzip compression and
tab/comma delimiters are detected automatically from the input contents.
Output is CSV (gzip if the output path ends in `.gz`). The script streams rows,
so the complete matrix does not need to fit in memory.

For a folder of inputs, the matching `run_prepare_fibers_data.py` writes and
submits one preparation job per file. Run it from the same folder as the other
HPC launchers, or use its full path:

```bash
python3 paper_analysis_codes/hla_perm_hpc_scripts/run_prepare_fibers_data.py \
  --d /path/to/combined_typing --w /path/to/results \
  --o /path/to/prepared_datasets \
  --pattern 'Best-Typing_AA_MM_9loc_matrix_*_aligned*' \
  --rc LSF --rm 4 --q i2c2_normal --dry-run
```

The runner accepts either one input file or an input folder. It reads the
imputation number immediately before `_aligned`, or the trailing number for
legacy filenames. For example, `Best-Typing_AA_MM_9loc_matrix_1_aligned.tsv.gz`
becomes `BestTyping_1.csv`, and `Best-Typing_AA_MM_9loc_matrix_10_aligned.txt`
becomes `BestTyping_10.csv`. The `9loc` part is not used as an imputation number.
Names such as `SampleImp1.txt` and `SampleImp10.txt.gz` continue to work.
`--prefix` changes `BestTyping`; `--pattern` selects input filenames within a
folder, and `--re` limits the number of inputs in natural filename order.
If inputs have no recognized imputation number, specify `--start-index 1` to assign numbers
in that order. Duplicate output names and accidental input replacement are
rejected before submission. Existing outputs require `--overwrite`.

`--dry-run` writes inspectable job files without submitting them. Remove it to
submit, or use `--rc SLURM` for SLURM. `--python-executable` selects the compute
node Python. The worker's `--o` names a file; the runner's `--o` names a folder.

The default output contains:

- Record IDs (`PX_ID`, `PERS_ID`, `TX_ID`, `DONOR_ID`) and dates
  (`REC_TX_DT`, `TFL_GRAFT_DT`, `TFL_DEATH_DT`, `TFL_LAFUDATE`) when present.
  These retain their source values, including blanks and leading zeros, pass
  through CV partitioning, and are ignored by the FIBERS workers. `DON_TY` is
  also retained unchanged for donor type.
- All three AA mismatch types across all nine loci: `MM_A_9` (position),
  `MM_A_9_Y` (residue), and `MM_A_9_rF_dY` (donor Y → recipient F permutation).
- The 43 base covariates used by the repository's HLA FIBERS jobs.
- All existing `Agmm*` columns, including the 0, 1, and 2 categories, plus
  the raw `REC_*_MM_EQUIV_CUR` antigen mismatch counts. Source values are
  copied exactly; no antigen indicators are created or recoded during preparation.
- `graftyrs` and `grf_fail`, required for FIBERS survival training.

Haplotypes, allele typing, raw clinical fields outside the
adjustment set, and `MM_*_COUNT` totals are excluded. Existing data values,
feature names, row order, and all rows are preserved. No rare-feature filtering,
constant-column filtering, binarization, or imputation is performed.

Preparation copies the source antigen fields without requiring or creating
additional indicators. Training uses the existing requested Agmm adjustment
columns and ignores raw antigen counts. Prepared data is never rewritten by
model fitting.

Missing or invalid required numeric values stop conversion with the column and
row identified. An unsuccessful conversion does not replace an existing output.

Optional arguments:

- `--mismatch-level position`, `residue`, or `permutation` keeps one type only;
  the default is `all`.
- `--cov-list` specifies antigen loci when reporting available model covariates;
  `None` reports only the base adjustment set. All source `Agmm*` fields and
  antigen counts are retained regardless of this option.
  `--cov-loci` remains an alias.
- `--overwrite` explicitly replaces an existing output after successful validation.

To run FIBERS on permutation features from the prepared data:

```bash
python paper_analysis_codes/hla_perm_hpc_scripts/job_fibers_hpc.py \
  --d data/prepared/BestTyping_1.csv --o /path/to/single_run \
  --r 0 --f log_rank_residuals --ra 0.1
```

Use a full cohort for model fitting. The nine-row sample is sufficient for
checking conversion but is too small for the configured survival adjustment.
The `job_fibers_hpc.py` and `job_fibers_hpc_cv.py` workers in this folder select
only permutation features from either the all-types or permutation-only output.
The original workers in `hla_hpc_scripts` still select position-only features.
See [FIBERS_run_commands.txt](FIBERS_run_commands.txt) and
[FIBERS_CV_run_commands.txt](FIBERS_CV_run_commands.txt) for the complete HPC
training and summary workflows.
