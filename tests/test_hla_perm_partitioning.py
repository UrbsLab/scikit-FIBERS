"""RRS memberships, CSV preservation, failure cleanup, and independent HPC tasks."""

import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pytest

from paper_analysis_codes.hla_perm_hpc_scripts import job_HLA_CV_Partitioner as worker
from paper_analysis_codes.hla_perm_hpc_scripts import run_HLA_CV_Partitioner as launcher


def write_source(path, counts=(83, 97), event_col='grf_fail', time_col='graftyrs', ties=False):
    rows = []
    for status, count in enumerate(counts):
        for i in range(count):
            time = 5 if ties else (100 + i if status == 0 else i / 100)
            rows.append([f'{len(rows):06}', str(float(status)), str(time), 'comma,quoted\ntext', '998'])
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\r\n')
        writer.writerow(['TX_ID', event_col, time_col, 'note', 'shared'])
        writer.writerows(rows)
    return rows


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.reader(handle))[1:]


@pytest.mark.parametrize('counts', [(83, 97), (20, 20), (10, 10), (0, 23)])
def test_rrs_has_one_member_per_full_time_block_per_fold(tmp_path, counts):
    source, output = tmp_path / 'data.csv', tmp_path / 'folds'
    rows = write_source(source, counts)
    result = worker.load_and_split_dataset(source, output, 10)
    assigned, sizes = {}, []
    for fold in range(1, 11):
        train = read_rows(output / f'data_CV_{fold}_Train.csv')
        test = read_rows(output / f'data_CV_{fold}_Test.csv')
        assert sorted(train + test) == rows
        assert {r[0] for r in train}.isdisjoint(r[0] for r in test)
        assert train == sorted(train) and test == sorted(test)
        sizes.append(len(test))
        for row in test:
            assert row[0] not in assigned
            assigned[row[0]] = fold
        report = json.loads((output / 'partition_reports' / f'data_CV_{fold}_partition.json').read_text())
        assert report['method'] == 'rrs' and report['block_size'] == 10
        assert report['seed'] == 42 and report['fold'] == fold
        for summary in report['balance']:
            actual = test if summary['split'] == 'Test' else train
            times = [float(r[2]) for r in actual if int(float(r[1])) == summary['event_status']]
            assert summary['rows'] == len(actual)
            assert summary['events'] == sum(int(float(r[1])) for r in actual)
            assert summary['status_rows'] == len(times)
            assert summary['time_median'] == float(np.median(times))
    assert len(assigned) == len(rows)
    for status in (0, 1):
        ordered = sorted((r for r in rows if int(float(r[1])) == status), key=lambda r: float(r[2]))
        for start in range(0, len(ordered), 10):
            block = ordered[start:start + 10]
            destinations = [assigned[r[0]] for r in block]
            assert len(set(destinations)) == len(block)  # Includes the remainder.
            if len(block) == 10:
                assert sorted(destinations) == list(range(1, 11))
        status_counts = [sum(assigned[r[0]] == f for r in ordered) for f in range(1, 11)]
        assert max(status_counts) - min(status_counts) <= 1
    assert max(sizes) - min(sizes) <= 2
    assert result['output_bytes'] == sum(p.stat().st_size for p in output.glob('*.csv'))
    assert not list(output.rglob('*.partial'))


@pytest.mark.parametrize('ties', [False, True])
def test_seed_reproducible_and_randomizes_ties(tmp_path, ties):
    source = tmp_path / 'data.csv'
    write_source(source, ties=ties)
    for folder, seed in [('first', 73), ('same', 73), ('changed', 74)]:
        worker.load_and_split_dataset(source, tmp_path / folder, 10, seed=seed)
    for path in (tmp_path / 'first').rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (tmp_path / 'same' / path.relative_to(tmp_path / 'first')).read_bytes()
    assert (tmp_path / 'first/data_CV_1_Test.csv').read_bytes() != (tmp_path / 'changed/data_CV_1_Test.csv').read_bytes()


def test_reads_source_only_once(tmp_path, monkeypatch):
    source = tmp_path / 'data.csv'
    write_source(source)
    opened = []
    original = Path.open

    def track(path, *args, **kwargs):
        if path == source:
            opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', track)
    worker.load_and_split_dataset(source, tmp_path / 'folds', 10, fold=3)
    assert opened == [source]


def test_preserves_csv_text_blank_lines_and_final_record(tmp_path):
    source = tmp_path / 'data.csv'
    source.write_bytes(b'\xef\xbb\xbfTX_ID,grf_fail,graftyrs,shared,note\r\n\r\n'
                       b'0001,1,0,998,"two\nlines"\r\n   \r\n'
                       b'0002,1,1,1.2300000000000001,"comma,quote"""\r\n0003,1,2,,text')
    output = tmp_path / 'folds'
    worker.load_and_split_dataset(source, output, 3)
    expected = [['0001', '1', '0', '998', 'two\nlines'],
                ['0002', '1', '1', '1.2300000000000001', 'comma,quote"'],
                ['0003', '1', '2', '', 'text']]
    seen = []
    for fold in range(1, 4):
        train = read_rows(output / f'data_CV_{fold}_Train.csv')
        test = read_rows(output / f'data_CV_{fold}_Test.csv')
        assert sorted(train + test) == expected
        assert len(test) == 1
        seen += test
    assert sorted(seen) == expected


@pytest.mark.parametrize('content', ['', 'grf_fail,graftyrs\n', 'grf_fail,graftyrs\n1,2\n',
                                     'grf_fail,grf_fail\n1,2\n3,4\n',
                                     'grf_fail,graftyrs\n1,2\n1,3,4\n',
                                     'grf_fail,graftyrs\n1,"unterminated\n'])
def test_invalid_source_preserves_existing_folds(tmp_path, content):
    source, output = tmp_path / 'data.csv', tmp_path / 'folds'
    write_source(source)
    worker.load_and_split_dataset(source, output, 10, fold=1)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    source.write_text(content)
    with pytest.raises((ValueError, csv.Error)):
        worker.load_and_split_dataset(source, output, 10, fold=1)
    assert {p: p.read_bytes() for p in before} == before
    assert not list(output.rglob('*.partial'))


@pytest.mark.parametrize('column,value', [('grf_fail', ''), ('grf_fail', '998'), ('grf_fail', 'nan'),
                                        ('graftyrs', ''), ('graftyrs', 'NA'), ('graftyrs', 'inf'),
                                        ('graftyrs', '-0.1')])
def test_invalid_survival_values_report_row_and_tx_id(tmp_path, column, value):
    source, output = tmp_path / 'data.csv', tmp_path / 'folds'
    write_source(source)
    with source.open(newline='') as handle:
        records = list(csv.reader(handle))
    records[2][records[0].index(column)] = value
    with source.open('w', newline='') as handle:
        csv.writer(handle).writerows(records)
    with pytest.raises(ValueError, match=r'Data row 2 \(TX_ID='):
        worker.load_and_split_dataset(source, output, 10)
    assert not output.exists()


def test_missing_column_and_rare_status_fail_before_writing(tmp_path):
    source, output = tmp_path / 'data.csv', tmp_path / 'folds'
    source.write_text('grf_fail,other\n0,1\n')
    with pytest.raises(ValueError, match='Missing required survival column'):
        worker.load_and_split_dataset(source, output, 10)
    write_source(source, counts=(20, 9))
    with pytest.raises(ValueError, match='only 9 rows for 10 folds'):
        worker.load_and_split_dataset(source, output, 10)
    assert not output.exists()


@pytest.mark.parametrize('kwargs', [{'partitions': 1}, {'partitions': 3, 'fold': 0},
                                   {'partitions': 3, 'fold': 4}, {'partitions': 3, 'fold': True},
                                   {'partitions': 3, 'seed': -1},
                                   {'partitions': 3, 'event_col': 'same', 'time_col': 'same'}])
def test_bad_options_fail_before_reading(tmp_path, kwargs):
    with pytest.raises(ValueError):
        worker.load_and_split_dataset(tmp_path / 'absent.csv', tmp_path / 'folds', **kwargs)


@pytest.mark.parametrize('failed_file', ['_Test.csv.', '_partition.json.', '_balance.csv.'])
def test_write_failure_preserves_completed_files_and_cleans_own_partials(tmp_path, monkeypatch, failed_file):
    source, output = tmp_path / 'data.csv', tmp_path / 'folds'
    write_source(source)
    worker.load_and_split_dataset(source, output, 10, fold=1)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    other_job = output / '.other_job.csv.unique.partial'
    other_job.write_bytes(b'belongs to a different running job')
    original_open = Path.open

    def fail_write(path, *args, **kwargs):
        if path.suffix == '.partial' and failed_file in path.name:
            raise OSError('Simulated output write failure')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', fail_write)
    with pytest.raises(OSError, match='Simulated output write failure'):
        worker.load_and_split_dataset(source, output, 10, fold=1)
    assert {p: p.read_bytes() for p in before} == before
    assert list(output.rglob('*.partial')) == [other_job]
    assert other_job.read_bytes() == b'belongs to a different running job'


def test_ten_independent_jobs_match_single_job_without_pandas_or_sklearn(tmp_path):
    source = tmp_path / 'data.csv'
    write_source(source)
    reference, output = tmp_path / 'reference', tmp_path / 'parallel'
    worker.load_and_split_dataset(source, reference, 10, seed=7)
    code = "import sys,runpy; sys.modules['pandas']=None; sys.modules['sklearn']=None; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    jobs = []
    try:
        for fold in range(1, 11):
            jobs.append(subprocess.Popen([
                sys.executable, '-c', code, worker.__file__, '--d', str(source), '--o', str(output),
                '--cv', '10', '--fold', str(fold), '--seed', '7'],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
        for job in jobs:
            _, stderr = job.communicate(timeout=60)
            assert job.returncode == 0, stderr
    finally:
        for job in jobs:
            if job.poll() is None:
                job.kill()
            job.communicate()
    for path in reference.rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (output / path.relative_to(reference)).read_bytes()
    assert len(list(output.glob('*.csv'))) == 20
    assert len(list((output / 'partition_reports').glob('*'))) == 20
    assert not list(output.rglob('*.partial'))


@pytest.mark.parametrize('cluster', ['LSF', 'SLURM'])
@pytest.mark.parametrize('limit', [2, 8])
def test_array_script_routes_fold_id_and_limits_concurrency(tmp_path, monkeypatch, cluster, limit):
    source = tmp_path / 'data.csv'
    write_source(source, counts=(5, 6))
    work, output = tmp_path / 'jobs', tmp_path / 'folds'
    with monkeypatch.context() as patch:
        patch.setattr(launcher.subprocess, 'run', lambda *a, **kw: pytest.fail('Dry run submitted'))
        launcher.main(['run', '--d', str(source), '--w', str(work), '--o', str(output),
                       '--cv', '3', '--rc', cluster, '--parallel-folds', '--max-concurrent-folds', str(limit),
                       '--python-executable', sys.executable, '--dry-run'])
    script = next((work / 'scratch').glob('*.sh'))
    content = script.read_text()
    if cluster == 'LSF':
        assert f'[1-3]%{min(limit, 3)}' in content
        monkeypatch.setenv('LSB_JOBINDEX', '2')
    else:
        assert f'#SBATCH --array=1-3%{min(limit, 3)}' in content
        monkeypatch.setenv('SLURM_ARRAY_TASK_ID', '2')
        binary = tmp_path / 'bin'
        binary.mkdir()
        stub = binary / 'srun'
        stub.write_text('#!/bin/sh\nexec "$@"\n')
        stub.chmod(0o755)
        monkeypatch.setenv('PATH', str(binary) + os.pathsep + os.environ['PATH'])
    subprocess.run(['bash', '-n', str(script)], check=True)
    subprocess.run(['bash', str(script)], check=True, capture_output=True, text=True)
    assert sorted(p.name for p in (tmp_path / 'folds_3').glob('*.csv')) == ['data_CV_2_Test.csv', 'data_CV_2_Train.csv']
    worker.load_and_split_dataset(source, tmp_path / 'reference', 3, fold=2)
    for path in (tmp_path / 'folds_3').rglob('*'):
        if path.is_file():
            assert path.read_bytes() == (tmp_path / 'reference' / path.relative_to(tmp_path / 'folds_3')).read_bytes()


@pytest.mark.parametrize('cluster', ['LSF', 'SLURM'])
def test_array_defaults_to_ten_jobs_16gb_and_forwards_custom_columns(tmp_path, monkeypatch, cluster):
    source = tmp_path / 'survival data.csv'
    write_source(source, event_col='event status', time_col='follow up')
    work, output = tmp_path / 'jobs', tmp_path / 'folds'
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **kw: pytest.fail('Dry run submitted'))
    launcher.main(['run', '--d', str(source), '--w', str(work), '--o', str(output), '--cv', '10',
                   '--rc', cluster, '--parallel-folds', '--c', 'event status', '--ol', 'follow up',
                   '--seed', '21', '--python-executable', sys.executable, '--dry-run'])
    script = next((work / 'scratch').glob('*.sh'))
    content = script.read_text()
    assert ('[1-10]%10' if cluster == 'LSF' else '--array=1-10%10') in content
    assert ('rusage[mem=16G]' if cluster == 'LSF' else '--mem=16G') in content
    command = shlex.split(content.splitlines()[-1])
    if cluster == 'SLURM':
        assert command.pop(0) == 'srun'
    assert command[command.index('--event-col') + 1] == 'event status'
    assert command[command.index('--time-col') + 1] == 'follow up'
    assert '--time-bins' not in command and '--stratify' not in command
    command[-1] = '2'
    worker.main(command[1:])
    report = json.loads((tmp_path / 'folds_10/partition_reports/survival data_CV_2_partition.json').read_text())
    assert report['fold'] == 2 and report['seed'] == 21 and report['method'] == 'rrs'
    assert report['event_col'] == 'event status' and report['time_col'] == 'follow up'


@pytest.mark.parametrize('option', [['--stratify', 'event_time'], ['--time-bins', '4']])
def test_removed_methods_are_rejected(tmp_path, option):
    with pytest.raises(SystemExit) as exc:
        worker.main(['job', '--d', str(tmp_path / 'data.csv'), '--o', str(tmp_path / 'folds'), *option])
    assert exc.value.code == 2
