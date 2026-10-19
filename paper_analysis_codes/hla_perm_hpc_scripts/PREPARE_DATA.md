Run from the repository root with Python 3.9 or later. Preparation uses only
the Python standard library; pandas is needed by training, not preparation:

```bash
python3 paper_analysis_codes/hla_perm_hpc_scripts/job_prepare_fibers_data.py \
  --d /path/to/Best-Typing_AA_MM_9loc_matrix_1_aligned.tsv.gz \
  --o data/prepared/BestTyping_1.csv \
  --tx-year-min 2010 --tx-year-max 2025 \
  --kidney-only --deceased-donor-only \
  --recipient-age-min 18 --donor-age-min 9 \
  --censor-date 2025-12-31 --days-per-year 365.24 --drop-missing
```

This applies the requested analysis cohort and removes missing numeric rows.
Both preparation scripts accept these parameters, and the supplied CV and non-CV
command files enable them. General script defaults still preserve rows: cohort
filters, censoring, and missing-row removal are opt-in.

## Cohort and censoring parameters

| Parameter | Source field and behavior |
| --- | --- |
| `--tx-year-min 2010 --tx-year-max 2025` | Keep `REC_TX_DT` from January 1, 2010 through December 31, 2025, inclusive. Either bound can be set independently. |
| `--kidney-only` | Require `REC_TX_ORG_TY=KI`, including decoded `KI: Kidney`. Combined-organ codes such as `KP` are excluded. Uses the transplant-organ field, not `ORG_TY` or STROBE flags. |
| `--deceased-donor-only` | Require `DON_TY=C` (cadaveric/deceased), including decoded `C: ...` values. Living donor code `L` is excluded. |
| `--recipient-age-min 18` | Keep `REC_AGE_AT_TX >= 18`, in years. |
| `--donor-age-min 9` | Keep `DON_AGE >= 9`, in years. |
| `--censor-date 2025-12-31` | Cap existing survival follow-up at this date. Omit to preserve outcomes unchanged. |
| `--days-per-year 365.24` | Conversion used by source `graftyrs`. Default 365.24 matches all nine supplied sample rows; change it for a different upstream definition. |
| `--drop-missing` | Remove rows missing retained numeric fields. Default remains false. |

The boolean cohort flags also accept explicit `true` or `false`. Filters run
before column selection and before CV partitioning. A missing field needed for
an enabled filter excludes the row with an explicit reason, even with
`--drop-missing false`. An absent required source column or malformed nonmissing
date stops conversion. ISO dates/timestamps, US `MM/DD/YYYY`, and the sample's
`MM/DD/YY` are accepted. Two-digit years use Python's standard pivot: 69–99 mean
1969–1999; 00–68 mean 2000–2068. Use four-digit years outside that range.

Censoring preserves the existing endpoint definition. Maximum follow-up is
`(censor_date - REC_TX_DT).days / days_per_year`. If `graftyrs` exceeds it, the
output receives the shorter duration and `grf_fail=0`. Events exactly on the
cutoff remain events, allowing 0.00001 day for rounding in source `graftyrs`.
Earlier events and earlier censored follow-up remain unchanged. Missing outcomes
are not reconstructed from event dates. Follow-up is never extended: if outcomes
were censored earlier upstream, regenerate them upstream to recover later
follow-up. Transplants after the censor date are excluded. Source files are never
modified.

## Inputs and batch preparation

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
  --tx-year-min 2010 --tx-year-max 2025 \
  --kidney-only --deceased-donor-only \
  --recipient-age-min 18 --donor-age-min 9 \
  --censor-date 2025-12-31 --drop-missing \
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
- The 43 base covariates used by the permutation FIBERS jobs, including `shared`.
- Existing processed Agmm indicators for the 0, 1, and 2 categories across the
  HLA loci, plus `Agmm0`. Nonmissing indicator values are copied unchanged;
  no antigen indicators are created or recoded during preparation.
- `graftyrs` and `grf_fail`, required for FIBERS survival training.

Haplotypes, allele typing, raw `REC_*_MM_EQUIV_CUR` antigen counts, other raw
clinical fields outside the adjustment set, Agmm text annotations, and
`MM_*_COUNT` totals are excluded. Nonmissing data values, feature names, row
order, and all rows are preserved by default. No rare-feature filtering,
constant-column filtering, binarization, or imputation is performed.

Preparation keeps the processed base and Agmm covariates without requiring or
creating additional indicators. Training uses the existing requested Agmm
adjustment columns. Prepared data is never rewritten by model fitting.

Preparation validates every nonmissing covariate, outcome, and AA mismatch cell
as finite decimal/scientific numeric text while processing its source row.
It rejects text, underscores in numbers, Unicode digits, infinities, and invalid
counts/outcomes. IDs, dates, and donor type remain metadata. Missing numeric cells
are written blank; all other numeric values retain their source representation
unless censoring changes outcomes.

The JSON summary records streaming validation and numbers of output rows and
numeric columns checked. An empty output records zero checked rows. Preparation
never imports pandas, builds a DataFrame, or rereads the converted CSV. Data and
reports are written incrementally; memory depends on the column count and one
record's size, not the dataset's row count. Non-CV and CV workers still check
actual DataFrame dtypes after `pd.to_numeric`, when loading data for training.

Missing cells remain missing by default, including in `shared`, outcomes, Agmm
indicators, and AA mismatches. Numeric missing markers are written as empty CSV
cells so pandas loads those columns as numeric with `NaN`; reports retain the
original source markers. Every preparation run automatically writes four reports
under `missing_data_reports/` beside the output dataset. For `BestTyping_1.csv`
(or `BestTyping_1.csv.gz`), these are:

- `BestTyping_1_missing_cells.csv`: one entry per missing cell with
  `row_number`, `source_line_number`, `TX_ID`, `column`, the original
  `missing_value`, `retained_in_output` (column status), and `row_dropped`.
- `BestTyping_1_missing_columns.csv`: every input column, its
  `retained_in_output` status, `total_rows`, `missing_count`, and
  `missing_percent` before any row dropping, including columns with no missing cells.
- `BestTyping_1_missing_summary.json`: total rows, columns, missing cells,
  rows with missing cells, and columns with missing cells, separately for
  the source and prepared dataset, plus all filter settings and `rows_dropped`.
  Counts separate cohort exclusions, subsequent missing-numeric exclusions,
  shortened follow-up, and events changed to censored observations.
- `BestTyping_1_row_exclusions.csv`: one entry per excluded source record with
  `row_number`, `source_line_number`, `PX_ID`, `TX_ID`, and semicolon-separated
  reasons. Individual reason counts can overlap. Cohort exclusions and subsequent
  missing-numeric exclusions are disjoint and sum to `rows_dropped`.

`row_number` starts at 1 for the first data record, excluding the header and
blank lines. `source_line_number` is the record's starting line in the
uncompressed input, with the header on line 1. `TX_ID` is copied as text,
preserving leading zeros and duplicate IDs; it is blank if the source lacks
that column. Reports scan every source column, including columns excluded
from the prepared dataset. The separate report folder keeps report CSVs out
of the training launchers' dataset discovery.

Missing markers include empty or whitespace-only cells, `NA`, `N/A`, `NaN`,
`null`, `None`, `<NA>`, `NaT`, `.`, `\N`, and common pandas missing markers.
`998` (including `998.0` and `9.98e2`) is treated as an ordinary numeric value,
so it is not reported or dropped as missing. Field-specific validation still applies.
Detection ignores case and surrounding whitespace; the complete marker list
is saved in the JSON summary. Zero is valid. Nonmissing numeric values and all
metadata values are copied unchanged unless censoring changes outcomes.
With no missing data, the detail report contains only its header
and the summary contains zero counts.

Absent required columns, malformed rows, invalid nonmissing numeric values,
infinities, and invalid counts or outcomes in cohort-eligible rows stop conversion. Validation
errors leave existing data and reports unchanged. Training still requires
complete numeric model inputs; use the reports to resolve missing values
before fitting.

Add `--drop-missing` (or `--drop-missing true`) to omit rows missing any retained
numeric field: processed covariates, outcomes, Agmm indicators, and selected
AA mismatch columns. Other metadata and excluded source columns do not trigger
missing-numeric drops; enabled cohort filters separately require their eligibility
fields. Omit the flag or use `--drop-missing false` to disable missing-numeric
exclusions. Both the `run_` and `job_` preparation
scripts accept this option.

Reports always include missing cells in dropped rows and preserve the original
source row numbers and `TX_ID`. Column counts describe the complete source;
the JSON `prepared` totals describe only rows actually written. If every row
is dropped, the output contains its header and the reports record zero output rows.

Optional arguments:

- `--mismatch-level position`, `residue`, or `permutation` keeps one type only;
  the default is `all`.
- `--cov-list` specifies antigen loci when reporting available model covariates;
  `None` reports only the base adjustment set. All existing processed Agmm 0/1/2
  indicators are retained regardless of this option; raw antigen fields are excluded.
  `--cov-loci` remains an alias.
- `--overwrite` replaces the output dataset and its reports after successful validation.
- `--drop-missing [true|false]` drops rows with missing retained numeric fields;
  default `false`, or `true` when supplied without a value.

To run FIBERS on permutation features from the prepared data:

```bash
python paper_analysis_codes/hla_perm_hpc_scripts/job_fibers_hpc.py \
  --d data/prepared/BestTyping_1.csv --o /path/to/single_run \
  --r 0 --f log_rank_residuals --ra 0.1 --mismatch-level permutation
```

Use a full cohort for model fitting. The nine-row sample is sufficient for
checking conversion but is too small for the configured survival adjustment.
The `run_fibers_hpc.py`, `run_fibers_hpc_cv.py`, and corresponding `job_` workers
accept `--mismatch-level mismatch` (alias `position`), `residue`, `permutation`,
or `all`. Training defaults to `permutation`; preparation defaults to `all`.
Keep all three types during preparation to switch training modes without
preparing again. `all` trains with all available types in the prepared input.
The original workers in `hla_hpc_scripts` still select position-only features.
See [FIBERS_run_commands.txt](FIBERS_run_commands.txt) and
[FIBERS_CV_run_commands.txt](FIBERS_CV_run_commands.txt) for the complete HPC
training and summary workflows.
