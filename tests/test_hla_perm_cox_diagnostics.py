"""Check that diagnosis distinguishes model failure from covariate evidence."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / 'paper_analysis_codes/hla_perm_hpc_scripts/job_diagnose_cox.py'
spec = importlib.util.spec_from_file_location('cox_diagnostics', SCRIPT)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def example_data():
    rng = np.random.default_rng(713)
    return pd.DataFrame({'Bin_0': rng.integers(0, 2, 250), 'x': rng.normal(size=250),
                         'constant': np.zeros(250), 'graftyrs': rng.exponential(4, 250),
                         'grf_fail': rng.integers(0, 2, 250)})


def make_exports(tmp_path):
    results = tmp_path / 'results'
    fold = results / '1'
    fold.mkdir(parents=True)
    (fold / '1_run_parameters.txt').write_text(
        "covariates: ['x', 'constant']\noutcome_label: graftyrs\ncensor_label: grf_fail\nn_groups: 2\n")
    pd.DataFrame([{'feature_list': "['MM_A_9_rF_dY']", 'group_threshold': 1}]).to_csv(fold / '1_pop.csv', index=False)
    for split in ('train', 'test'):
        for model, template in diagnostics.MODELS.items():
            stem = f'1_{template.format(split=split)}'
            if split == 'test' and model != 'unadjusted':
                (fold / (stem + '_failed.txt')).write_text('Generic failure\n')
            else:
                pd.DataFrame({'covariate': ['Bin_0'] + ([] if model == 'unadjusted' else ['x', 'constant'])}).to_csv(fold / (stem + '.csv'), index=False)
    return results


def test_detect_constant_and_joint_dependence():
    data = example_data()
    data['duplicate_x'] = 2 * data['x'] + 3
    checks, _ = diagnostics.check_columns(data, 'graftyrs', 'grf_fail')
    assert checks.set_index('covariate').loc['constant', 'flags'] == 'constant'
    dependencies = diagnostics.check_dependencies(data[['Bin_0', 'x', 'constant', 'duplicate_x']])
    assert dependencies['constants'] == ['constant']
    assert dependencies['centered_scaled_rank'] == 2
    assert len(dependencies['dependencies']) == 1
    assert dependencies['dependencies'][0]['max_abs_residual'] < 1e-10


def test_capture_real_failure_and_success_after_omission(tmp_path):
    data = example_data()
    failed = diagnostics.fit_cox(data, 'graftyrs', 'grf_fail', tmp_path, 'constant_model')
    assert failed['status'] == 'failed'
    assert failed['error_type'] and failed['error']
    assert 'constant' in failed['warnings']
    passed = diagnostics.fit_cox(data.drop(columns='constant'), 'graftyrs', 'grf_fail', tmp_path, 'omitted')
    assert passed['status'] == 'ok'
    assert (tmp_path / 'constant_model_error.txt').exists()


def test_reconstruct_threshold_and_preserve_invalid_cell_identity(tmp_path):
    results = make_exports(tmp_path)
    data = pd.DataFrame({'MM_A_9_rF_dY': [0, 1, 2, np.nan], 'x': [1, 2, 3, 4],
                         'constant': 0, 'graftyrs': [1, 2, 3, 4], 'grf_fail': [0, 1, 1, 0],
                         'TX_ID': ['001', '002', '003', '004']})
    source = tmp_path / 'fold.csv'
    data.to_csv(source, index=False)
    design, _, _, _, invalid, _ = diagnostics.load_design(results / '1', 1, source)
    assert design.Bin_0.iloc[:3].tolist() == [0, 0, 1]
    assert pd.isna(design.Bin_0.iloc[3])
    assert invalid.iloc[0]['TX_ID'] == '004'
    assert invalid.iloc[0]['data_row'] == 4


def test_cli_failed_fold_diagnosis_and_roster_not_attribution(tmp_path):
    results = make_exports(tmp_path)
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    data = example_data()
    data['MM_A_9_rF_dY'] = 2 * data.pop('Bin_0')
    data.to_csv(data_dir / 'sample_CV_1_Test.csv', index=False)
    output = tmp_path / 'diagnostics'
    subprocess.run([sys.executable, str(SCRIPT), '--results', str(results),
                    '--data', str(data_dir), '--o', str(output), '--leave-one-out', '--univariate'], check=True)
    fits = pd.read_csv(output / 'fit_results.csv')
    assert len(fits) == 10  # 3 baselines + 4 omission fits + 3 univariate fits.
    omission = fits[fits.omitted_covariate == 'constant']
    assert len(omission) == 2 and set(omission.status) == {'ok'}
    roster = pd.read_csv(output / 'failed_model_covariates.csv')
    assert roster.role.str.contains('not proven responsible').all()
    assert pd.read_csv(output / 'diagnostic_errors.csv').empty
