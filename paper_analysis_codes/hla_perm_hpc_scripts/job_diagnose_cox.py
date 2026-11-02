"""Diagnose saved FIBERS Cox failures without changing the fitted models or data.

Requires numpy, pandas, scipy and lifelines. Reads population/parameter CSV/TXT
exports, never pickles. Without --data, reports only what the saved files show.
With --data, reconstructs Bin_0 and fits CoxPHFitter directly so errors survive.
Omission/univariate fits are diagnostics, not recommendations to remove covariates.
"""

import argparse
import ast
import csv
import json
from pathlib import Path
import platform
import traceback
import warnings

import lifelines
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from scipy.linalg import qr


MODELS = {'unadjusted': 'coxph_unadj_bin_{split}_0',
          'base': 'coxph_adj_bin_{split}_0_NoAg',
          'full': 'coxph_adj_bin_{split}_0'}


def read_parameters(folder, fold):
    return dict(line.split(': ', 1) for line in
                (folder / f'{fold}_run_parameters.txt').read_text().splitlines()
                if ': ' in line)


def covariate_sets(folder, fold, parameters):
    full = ast.literal_eval(parameters['covariates'])
    for split in ('train', 'test'):
        path = folder / f'{fold}_coxph_adj_bin_{split}_0_NoAg.csv'
        if path.exists():
            with path.open(newline='') as handle:
                base = [row['covariate'] for row in csv.DictReader(handle)
                        if row['covariate'] != 'Bin_0']
            if not set(base) <= set(full):
                raise ValueError('Base covariates are not a subset of the saved full model')
            return {'unadjusted': [], 'base': base, 'full': full}
    raise ValueError('Need one successful NoAg export to recover its exact covariate list')


def inventory(results):
    records, rosters = [], []
    for folder in sorted((p for p in results.iterdir() if p.is_dir() and p.name.isdigit()),
                         key=lambda p: int(p.name)):
        fold = int(folder.name)
        sets = covariate_sets(folder, fold, read_parameters(folder, fold))
        for split in ('train', 'test'):
            for model, template in MODELS.items():
                stem = f'{fold}_{template.format(split=split)}'
                success, failure = folder / (stem + '.csv'), folder / (stem + '_failed.txt')
                status = ('conflicting_exports' if success.exists() and failure.exists()
                          else 'ok' if success.exists() else 'failed' if failure.exists() else 'missing')
                records.append({'fold': fold, 'split': split, 'model': model, 'status': status,
                                'failure_reason': failure.read_text().strip() if failure.exists() else '',
                                'covariates': json.dumps(sets[model]),
                                'exact_failure_covariate': '',
                                'diagnosis': 'Not identifiable from saved model-level status alone'})
                if status != 'ok':
                    for column in ['Bin_0'] + sets[model]:
                        rosters.append({'fold': fold, 'split': split, 'model': model,
                                        'covariate': column,
                                        'role': 'Included in failed/unavailable model; not proven responsible'})
    if not records:
        raise ValueError('No numbered CV result directories found')
    return pd.DataFrame(records), pd.DataFrame(rosters, columns=['fold', 'split', 'model', 'covariate', 'role'])


def load_design(folder, fold, csv_path):
    params = read_parameters(folder, fold)
    if int(params.get('n_groups', '2')) != 2:
        raise ValueError('This diagnostic supports the current two-group runs only')
    sets = covariate_sets(folder, fold, params)
    with (folder / f'{fold}_pop.csv').open(newline='') as handle:
        top = next(csv.DictReader(handle))
    features = ast.literal_eval(top['feature_list'])
    threshold = float(top['group_threshold'])
    duration, event = params['outcome_label'], params['censor_label']
    required = list(dict.fromkeys(features + sets['full'] + [duration, event]))
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError('Missing source columns: ' + ', '.join(missing))
    columns = required + (['TX_ID'] if 'TX_ID' in header else [])
    raw = pd.read_csv(csv_path, usecols=columns, dtype={'TX_ID': str})
    numeric = raw[required].apply(pd.to_numeric, errors='coerce')
    values = numeric.to_numpy(dtype=float)
    bad_rows, bad_columns = np.where(~np.isfinite(values))
    invalid = pd.DataFrame([{'data_row': int(i) + 1,
                             'TX_ID': raw.at[i, 'TX_ID'] if 'TX_ID' in raw else '',
                             'column': required[j], 'value': str(raw.at[i, required[j]])}
                            for i, j in zip(bad_rows, bad_columns)],
                           columns=['data_row', 'TX_ID', 'column', 'value'])
    design = numeric[sets['full'] + [duration, event]].copy()
    sums = numeric[features].sum(axis=1, skipna=False)
    design.insert(0, 'Bin_0', np.where(np.isfinite(sums), (sums > threshold).astype(float), np.nan))
    return design, sets, duration, event, invalid, threshold


def check_columns(design, duration, event):
    rows, levels = [], []
    events = design[event] == 1
    censored = design[event] == 0
    for column in design.columns.drop([duration, event]):
        x = design[column].replace([np.inf, -np.inf], np.nan)
        finite = x.dropna()
        variance = finite.var()
        event_var, censored_var = x[events].var(), x[censored].var()
        flags = []
        if x.isna().any(): flags.append('missing_or_nonfinite')
        if finite.nunique() <= 1: flags.append('constant')
        elif variance < 1e-4: flags.append('low_variance_raw_scale')
        if finite.nunique() > 1 and (event_var == 0 or censored_var == 0):
            flags.append('constant_within_event_status_check_separation')
        counts = finite.value_counts()
        if len(counts) > 1 and counts.iloc[0] / len(finite) > .999:
            flags.append('very_rare_nonmodal_values')
        rows.append({'covariate': column, 'n': len(x), 'nonfinite': int(x.isna().sum()),
                     'unique_values': finite.nunique(), 'variance': variance,
                     'variance_events': event_var, 'variance_censored': censored_var,
                     'nonzero_events': int(((x != 0) & x.notna() & events).sum()),
                     'nonzero_censored': int(((x != 0) & x.notna() & censored).sum()),
                     'minimum': finite.min(), 'maximum': finite.max(),
                     'flags': '; '.join(flags)})
        if 0 < finite.nunique() <= 5:
            for value in sorted(finite.unique()):
                mask = x == value
                levels.append({'covariate': column, 'value': value, 'rows': int(mask.sum()),
                               'events': int((mask & events).sum()),
                               'censored': int((mask & censored).sum())})
    return pd.DataFrame(rows), pd.DataFrame(levels)


def check_dependencies(frame):
    """Report dependencies jointly; the QR-selected column is not a unique culprit."""
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        return {'status': 'nonfinite_design', 'dependencies': []}
    constants = frame.columns[frame.nunique() <= 1].tolist()
    variable = frame.drop(columns=constants)
    if variable.empty:
        return {'status': 'all_constant', 'constants': constants, 'dependencies': []}
    z = (variable - variable.mean()) / variable.std(ddof=0)
    _, r, pivots = qr(z.to_numpy(), mode='economic', pivoting=True)
    singular = np.linalg.svd(r, compute_uv=False)
    tolerance = max(z.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tolerance).sum())
    names = variable.columns.tolist()
    independent = list(pivots[:rank])
    dependencies = []
    for index in pivots[rank:]:
        coefficients = np.linalg.lstsq(z.iloc[:, independent], z.iloc[:, index], rcond=None)[0]
        residual = z.iloc[:, index].to_numpy() - z.iloc[:, independent].to_numpy() @ coefficients
        dependencies.append({'dependent_column': names[index],
                             'standardized_relation': {names[i]: float(c) for i, c in zip(independent, coefficients) if abs(c) > 1e-8},
                             'max_abs_residual': float(np.abs(residual).max())})
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    return {'status': 'checked', 'constants': constants, 'nonconstant_columns': len(names),
            'centered_scaled_rank': rank, 'condition_number': condition,
            'dependencies': dependencies,
            'note': 'Linear dependence is joint. Which column QR labels dependent depends on ordering/scaling.'}


def fit_cox(data, duration, event, output, label):
    record = {'fit': label, 'status': 'failed', 'error_type': '', 'error': '', 'warnings': '',
              'n': len(data), 'n_predictors': len(data.columns) - 2}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            if not np.isfinite(data.to_numpy(dtype=float)).all():
                raise ValueError('Missing or nonfinite values; no rows were dropped')
            if (data[duration] < 0).any() or not data[event].isin([0, 1]).all():
                raise ValueError('Invalid survival time or event encoding')
            model = CoxPHFitter()  # Match the original unpenalized fit.
            model.fit(data, duration_col=duration, event_col=event)
            model.summary.to_csv(output / f'{label}_coefficients.csv')
            if not np.isfinite(model.summary[['coef', 'se(coef)']].to_numpy()).all():
                raise ValueError('Fit returned nonfinite coefficient estimates or standard errors')
            record['status'] = 'ok'
        except Exception as error:
            record.update(error_type=type(error).__name__, error=str(error))
            (output / f'{label}_error.txt').write_text(traceback.format_exc())
    record['warnings'] = '\n'.join(f'{w.category.__name__}: {w.message}' for w in caught)
    if record['status'] == 'ok' and any(issubclass(w.category, ConvergenceWarning) for w in caught):
        record['status'] = 'convergence_warning'
    if caught:
        (output / f'{label}_warnings.txt').write_text(record['warnings'] + '\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Directory containing matching *_CV_N_Train/Test.csv files')
    parser.add_argument('--o', type=Path, required=True)
    parser.add_argument('--fold', nargs='+', type=int, help='Default: folds with failed adjusted models in the selected split')
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--univariate', action='store_true', help='Also fit each predictor alone')
    parser.add_argument('--leave-one-out', action='store_true', help='For failed/warning adjusted models, omit each covariate in turn')
    args = parser.parse_args()
    results, output = args.results.resolve(), args.o.resolve()
    if output == results or results in output.parents:
        parser.error('Write diagnostics outside the original results folder')
    output.mkdir(parents=True, exist_ok=True)
    saved, roster = inventory(results)
    saved.to_csv(output / 'saved_model_status.csv', index=False)
    roster.to_csv(output / 'failed_model_covariates.csv', index=False)
    (output / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'lifelines': lifelines.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__,
        'results': str(results), 'data': str(args.data.resolve()) if args.data else None,
        'penalizer': 0, 'split': args.split,
        'univariate': args.univariate, 'leave_one_out': args.leave_one_out,
    }, indent=2) + '\n')
    guidance = ('Saved failures are model-level; their covariate lists do not identify a culprit.\n'
                'Raw fold CSVs are required for covariate diagnosis. No pickle is loaded.\n'
                'Column flags, univariate failures and omission successes identify suspects, not proof of a unique cause.\n'
                'Several variables may cause joint problems. Do not use test refits to choose the final adjustment set.\n'
                'No source rows/covariates are removed, no outcomes are changed, and no penalized replacement model is saved.\n'
                'Reference: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model\n')
    (output / 'README.txt').write_text(guidance)
    if args.data is None:
        print('Saved-status inventory written. Supply --data for actual covariate diagnostics.')
        return
    selected = args.fold or sorted(saved.loc[(saved.split == args.split) &
                      saved.model.isin(['base', 'full']) & (saved.status == 'failed'), 'fold'].unique())
    all_fits, all_checks, errors = [], [], []
    for fold in selected:
        fold = int(fold)
        destination = output / f'CV{fold}_{args.split}'
        destination.mkdir(exist_ok=True)
        try:
            matches = sorted(args.data.glob(f'*_{fold}_{args.split.title()}.csv'))
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one fold CSV, found {len(matches)}: {matches}')
            design, sets, duration, event, invalid, threshold = load_design(results / str(fold), fold, matches[0])
            invalid.to_csv(destination / 'invalid_cells.csv', index=False)
            checks, levels = check_columns(design, duration, event)
            checks.to_csv(destination / 'column_checks.csv', index=False)
            levels.to_csv(destination / 'category_event_counts.csv', index=False)
            all_checks.append(checks.assign(fold=fold, split=args.split))
            for model_name, covariates in sets.items():
                predictors = ['Bin_0'] + covariates
                frame = design[predictors + [duration, event]]
                dependencies = check_dependencies(frame[predictors])
                (destination / f'{model_name}_dependencies.json').write_text(json.dumps(dependencies, indent=2) + '\n')
                print(f'CV{fold} {args.split}: {model_name}, n={len(frame)}, threshold={threshold}', flush=True)
                baseline = fit_cox(frame, duration, event, destination, model_name)
                all_fits.append(dict(baseline, fold=fold, split=args.split, model=model_name, omitted_covariate=''))
                if args.leave_one_out and model_name != 'unadjusted' and baseline['status'] != 'ok':
                    for index, column in enumerate(covariates):
                        trial = fit_cox(frame.drop(columns=column), duration, event, destination, f'{model_name}_omit_{index:02}')
                        all_fits.append(dict(trial, fold=fold, split=args.split, model=model_name, omitted_covariate=column))
            if args.univariate:
                for index, column in enumerate(['Bin_0'] + sets['full']):
                    trial = fit_cox(design[[column, duration, event]], duration, event, destination, f'univariate_{index:02}')
                    all_fits.append(dict(trial, fold=fold, split=args.split, model='univariate', covariate=column, omitted_covariate=''))
        except Exception as error:
            errors.append({'fold': fold, 'split': args.split, 'error': str(error)})
            (destination / 'diagnostic_error.txt').write_text(traceback.format_exc())
        # Save after each fold, including partial fit results if later diagnostics fail.
        pd.DataFrame(all_fits).to_csv(output / 'fit_results.csv', index=False)
        if all_checks:
            pd.concat(all_checks, ignore_index=True).to_csv(output / 'all_column_checks.csv', index=False)
        pd.DataFrame(errors, columns=['fold', 'split', 'error']).to_csv(output / 'diagnostic_errors.csv', index=False)
    print(f'Diagnostics: {output}', flush=True)
    if errors:
        raise SystemExit('Some folds could not be diagnosed; see diagnostic_errors.csv')


if __name__ == '__main__':
    main()
