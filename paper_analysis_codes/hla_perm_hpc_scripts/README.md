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

Preparation retains all three AA mismatch types, the 42 base covariates, all
existing `Agmm*` columns (including 0/1/2 categories), raw antigen mismatch
counts, and both survival outcomes. Record IDs, dates, and `DON_TY` are also
retained when present. Antigen values are copied without recoding.
`shared` is excluded from preparation and training; missing values in that
column do not remove rows or stop processing.
Metadata stays with the rows through CV partitioning and is excluded by the
workers' `IGNORE_COLUMNS` lists. See [PREPARE_DATA.md](PREPARE_DATA.md).
The FIBERS workers then select only donor-to-recipient permutation features.
`MM_A_9_rF_dY` means **donor Y → recipient F** at A position 9. The reverse pair
`MM_A_9_rY_dF` is a separate feature. Names and integer counts are preserved.
Position-only columns, residue-only columns, locus totals, and metadata do not
enter permutation bins.

Training uses every observed AA position for the selected loci. The old
`locus_range_dict` and its filtering condition remain commented in both workers.
The default loci are A,B,C,DRB1,DRB345,DQA1,DQB1; `--loci-list` can also include
DPA1 and DPB1. `--all-positions` remains accepted for command compatibility.
Outcomes default to `graftyrs` and `grf_fail`.

Training requires the 42 base covariates and adds only existing source Agmm
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

Outputs include populations, fitted-model pickles, parameters, Cox reports,
Kaplan–Meier plots, feature-frequency audits, and retained-feature counts.
Summary jobs produce seed/fold statistics, master summaries, and heatmaps.
A failed secondary Cox fit is recorded in a `_failed.txt` file while the fitted
model is saved; resolve missing required Cox reports before summarizing that run.

Table makers retain the original inline experiment comparisons, updated to
`hla_perm_` names. The command files include the three fitness experiments
needed for the DP3 comparison. Comparisons whose input summaries are missing
are skipped. Edit the inline experiment lists in the table workers to add your
own comparisons, including directional analyses or different locus sets.
