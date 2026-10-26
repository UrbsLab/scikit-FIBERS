This folder follows `hla_hpc_scripts`: each `run_*.py` launcher writes and submits
jobs to its matching standalone `job_*.py` worker. Preparation, training,
partitioning, summaries, master summaries, and table makers are all included.
There are no shared custom launcher, argument, or data-processing modules.
The comparison notebook and cleanup utilities are also retained. Edit the
notebook paths for your experiments. `cleanup.py --w /path/to/results` removes
only that results root's logs and scratch contents; `killjobsgt.sh` retains the
original LSF job-cancellation behavior.

Use [FIBERS_run_commands.txt](FIBERS_run_commands.txt) for imputation/seed runs,
or [FIBERS_CV_run_commands.txt](FIBERS_CV_run_commands.txt) for cross-validation.
Run commands from this folder in a Python environment with the repository's
FIBERS dependencies. Replace the example paths and run one stage at a time,
waiting for successful completion before starting its summaries. Workers import
FIBERS from this checkout, so the checkout must also exist on compute nodes.

| Stage | Non-CV run/job pair | CV run/job pair |
| --- | --- | --- |
| Prepare combined typing data | `run_prepare_fibers_data.py` / `job_prepare_fibers_data.py` | Same preparation pair |
| Partition folds | — | `run_HLA_CV_Partitioner.py` / `job_HLA_CV_Partitioner.py` |
| Train FIBERS | `run_fibers_hpc.py` / `job_fibers_hpc.py` | `run_fibers_hpc_cv.py` / `job_fibers_hpc_cv.py` |
| Summarize runs | `run_sum_fibers_hpc.py` / `job_sum_fibers_hpc.py` | `run_sum_fibers_hpc_cv.py` / `job_sum_fibers_hpc_cv.py` |
| Master summaries | `run_master_sum_fibers_hpc.py` / `job_master_sum_fibers_hpc.py` | `run_master_sum_fibers_hpc_cv.py` / `job_master_sum_fibers_hpc_cv.py` |
| Comparison tables | `run_table_maker_hpc.py` / `job_table_maker_hpc.py` | `run_table_maker_hpc_cv.py` / `job_table_maker_hpc_cv.py` |

Preparation retains all three AA mismatch types, the 43 processed base covariates,
existing Agmm 0/1/2 indicators, and both survival outcomes. Raw antigen counts and
Agmm text annotations are excluded. Record IDs, dates, and `DON_TY` are also
retained when present. Nonmissing Agmm values are copied without recoding.
`shared` is retained. By default, missing cells remain missing without dropping
rows; numeric missing markers are written as blanks so pandas infers numeric
columns. Preparation validates decimal/scientific numeric values one row at a
time and records the result in the JSON summary. It uses only the Python standard
library and never loads a DataFrame or rereads the output. Memory depends on the
number of columns and the size of one record, rather than the total row count.
Each dataset gets reports under `missing_data_reports/`:
every missing cell with its row number, `TX_ID`, and column; counts and
percentages by column; and overall totals for the source and prepared data.
Reports cover all input columns and identify which are retained in the output.
Both preparation scripts accept `--drop-missing` or `--drop-missing true` to
drop rows missing retained numeric fields. The default is `false`; metadata
and excluded source columns do not cause drops. Reports retain source row
numbers and `TX_ID`, identify dropped rows, and summarize the resulting dataset.
Metadata stays with the rows through CV partitioning and is excluded by the
workers' `IGNORE_COLUMNS` lists. See [PREPARE_DATA.md](PREPARE_DATA.md).
The preparation commands now enable the requested cohort: 2010–2025 transplants,
kidney only, deceased donors, recipient age at least 18, donor age at least 9,
and missing-numeric row removal. Each condition has a parameter. Filters use
`REC_TX_DT`, `REC_TX_ORG_TY`, `DON_TY`, `REC_AGE_AT_TX`, and `DON_AGE` before
column selection and CV splitting; they do not use STROBE flags. Missing
eligibility fields exclude rows and are recorded in the row-exclusion report.
The `--censor-date 2025-12-31` option caps existing follow-up and censors later
events without extending earlier follow-up. `--days-per-year 365.24` matches
the supplied sample's survival-year units. General script defaults remain
unfiltered; the run-command files explicitly enable the requested settings.
The reports also record exclusions by source row, `PX_ID`, `TX_ID`, and reason,
plus counts for the cohort, missing-data, and censoring steps.
Both training launchers and workers accept `--mismatch-level`:

| Value | AA features used for training |
| --- | --- |
| `mismatch` (or `position`) | Position counts, such as `MM_A_9` |
| `residue` | Residue counts, such as `MM_A_9_Y` |
| `permutation` (default) | Donor-to-recipient counts, such as `MM_A_9_rF_dY` |
| `all` | All three available types together |

For example, add `--mismatch-level all` to `run_fibers_hpc.py` or
`run_fibers_hpc_cv.py`. The same option works when calling either `job_` worker
directly. Preparation defaults to keeping all three types, so one prepared
dataset can support each training mode. Training cannot use columns omitted
during preparation. Use a separate `--w` directory for each mode and use that
same directory for its summaries and comparison tables.

Workers select the requested types and loci while reading the CSV, so excluded
columns are not loaded into the model DataFrame. CV selects features from the
training fold and loads exactly the retained feature/covariate schema for testing.
`MM_A_9_rF_dY` means **donor Y → recipient F** at A position 9. The reverse pair
`MM_A_9_rY_dF` is a separate feature. Names and integer counts are preserved.
Locus totals and metadata never enter bins. Position and residue columns enter
bins only when their type or `all` is selected.

Training uses every observed AA position for the selected loci. The old
`locus_range_dict` and its filtering condition remain commented in both workers.
The default loci are A,B,C,DRB1,DRB345,DQA1,DQB1; `--loci-list` can also include
DPA1 and DPB1. `--all-positions` remains accepted for command compatibility.
Outcomes default to `graftyrs` and `grf_fail`.

Training requires the 43 base covariates and adds only existing source Agmm
0/1 indicators for the requested `--cov-list` loci. Raw antigen counts are
ignored, and no missing indicators are reconstructed. The default requested
antigen loci are A,B,C,DRB1,DQA1,DQB1; the exact adjustment set depends on which
source indicators are present and is saved with the fitted model. CV chooses
this set from the training schema and requires the same fields in its test fold.
`--cov-list None` uses only the base covariates; DRB1 and DRB345 share indicators.
`--de default`, `--de protective`, and `--de high_risk` are supported.

At `--ra 0`, all-zero features are removed. A positive `--ra` removes features
whose nonzero fraction is below that threshold. CV uses training rows to choose
features and applies exactly that list to the test fold. Missing required
columns, non-finite values, and invalid counts or outcomes stop the worker.
Workers explicitly check numeric DataFrame dtypes after conversion.
`998` is treated as an ordinary numeric value in preparation and training.
Preparation's missing-data reports can be used to resolve missing model inputs
before training; preparation does not impute them.

The non-CV launchers expect one dataset prefix with consecutive imputation
numbers: `BestTyping_1.csv` through `BestTyping_<re>.csv`. Prefixes may contain
underscores. Preparation accepts TXT/TSV/CSV and gzip files, including
`Best-Typing_AA_MM_9loc_matrix_<number>_aligned.tsv.gz` and extensionless
aligned files. It preserves the number before `_aligned` (or a legacy trailing
number); `--start-index 1` can assign consecutive numbers when needed. `--rs` controls
the number of seeds, from 0 through `rs-1`. The CV launcher accepts one pair
of `*_<fold>_Train.csv` and `*_<fold>_Test.csv` files per fold and checks all
pairs before submission. `--r` sets the fixed CV seed.

All launchers accept `--rc LSF` or `--rc SLURM`, `--q`, `--rm`,
`--python-executable`, and `--dry-run`. Dry runs write shell files without
submitting; normal runs submit immediately. Generated scripts, logs, and
results live under `--w` in `scratch/`, `logs/`, and `output/`.
Use separate `--w` roots for CV and non-CV analyses.

Output naming follows the original scripts:

- Non-CV training and seed summaries use `--o hla_perm_7locus_product_DP3`.
  They add `Fibers2.0_`, with each dataset in its own subfolder.
- Non-CV master summaries use the full experiment folder name:
  `--o Fibers2.0_hla_perm_7locus_product_DP3`.
- The non-CV table launcher uses `--o tables` to place comparisons in
  `<w>/output/tables/`.
- CV training also adds `Fibers2.0_`, with one numbered subfolder per fold.
  CV summary/master/table launchers take `--w` and `--cv` and scan the
  experiment folders under `<w>/output/`.
- The partitioning launcher appends `_<cv>` to its `--o` folder prefix;
  the direct partitioning worker uses `--o` exactly.

Partitioning copies CSV records without pandas, preserving numeric text, IDs,
dates, and original row order. RRS is the only partitioning method.
Each job loads the plain CSV into RAM as bytes once, then writes
its folds from RAM. Memory use is approximately the CSV size plus row bookkeeping and buffers,
without expanding every cell into a pandas object.
NumPy supplies the time sort and random fold assignment. Fold outputs remain CSV.
Logs show loading/writing progress every 10,000 rows
and the expected output size. Fold files and reports are written to unique
`.partial` files next to their final paths, then renamed after writing succeeds;
there is no temporary directory or second data copy. Handled failures clean up
that job's partial files. A forced kill can leave partial files, which are not
finished fold CSVs. Renames are atomic per file, not across the complete set;
wait for all partition jobs to finish before starting training.

`run_HLA_CV_Partitioner.py --cv 10 --parallel-folds --max-concurrent-folds 10`
submits ten independent fold jobs in one array, allowing all ten to run together.
Both LSF and SLURM are supported, with separate logs for each array task.
The run-command file enables
this mode with 16 GB requested per job. Each task reads the source once into its
own RAM and writes only its own train/test pair. The initial reads still access
shared storage; no source reads occur during writing in this mode.
Omitting `--parallel-folds` writes all folds from RAM in one job.
The direct worker's `--fold 3 --cv 10` writes only fold 3 for a targeted retry.
Wait for the entire array to succeed before submitting CV training. Ten folds
still write roughly ten times the input's size across all pairs.
The concurrency limits use the schedulers' documented array controls:
[LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=arrays-job-array-job-slot-limit)
and [SLURM](https://slurm.schedmd.com/job_array.html).

### RRS survival partitioning

Both scripts now use representative random sampling (RRS) only. There is no
`--stratify` or `--time-bins` option.

1. Separate events (`grf_fail=1`) from censored rows (`grf_fail=0`).
2. Shuffle each status group to randomize ties, then stably sort by `graftyrs`.
3. Take consecutive blocks of K rows, where K is `--cv`. Randomly send one row
   from each full block to each fold. Assign the final incomplete block to
   distinct random folds.

This implements the allocation in
[Rendleman et al. (2022), Algorithm 1](https://link.springer.com/article/10.1007/s11222-022-10138-7),
separately by status. Equal times can cross block boundaries. Each fold receives
nearly the same count for each status (difference at most one). Independent
remainders from the two statuses can make total fold sizes differ by up to two.
Every row appears in exactly one test fold and K-1 training folds. Original
continuous times stay unchanged; for censored rows these are times until censoring.

| Option | Meaning |
| --- | --- |
| `--cv 10` | Number of folds and RRS block size. |
| `--event-col grf_fail` (alias `--c`) | Binary status: 0=censored, 1=event. |
| `--time-col graftyrs` (alias `--ol`) | Finite, nonnegative observed follow-up. |
| `--seed 42` | Partition seed, separate from FIBERS training seed `--r`. |
| `--fold 3` | Worker only: write a single fold, using the same assignment. |

All array jobs reproduce the same assignment from identical input row order,
options, and seed. To keep each observed status in every fold, the worker stops
if that status has fewer than K rows. Missing/invalid outcomes stop the job with
the data-row number and TX_ID when present. Partitioning does not drop rows.
Other columns are copied without interpreting missing-value codes. Rows from the
same patient or donor are not grouped by this splitter.

Each fold writes two small files under `partition_reports/`:

- `<input>_CV_<fold>_balance.csv`: train/test counts, event fractions, and observed
  time min/quartiles/max separately for each status. These quartiles summarize
  the output only; they do not determine folds.
- `<input>_CV_<fold>_partition.json`: source identity, method (`rrs`), K, fold,
  seed, column names, and the same balance summary.

The commands in `FIBERS_CV_run_commands.txt` write data to
`PermutationData/BestTyping_1_CV_rrs_10`, with runs/logs/summaries under
`output_shared/Best_Center_Data_Typing_Deceased/cv_rrs`.
