"""Load a plain CSV into RAM and write survival CV folds using RRS."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np


def _records(source):
    """Parse records while retaining their exact text, including quoted newlines."""
    lines = []

    def tracked_lines():
        for line in source:
            lines.append(line)
            yield line

    for fields in csv.reader(tracked_lines(), strict=True):
        record = ''.join(lines)
        lines.clear()
        if record.strip():
            yield fields, record


def _payload(record):
    # A final record without a newline may become an interior row in a fold.
    if not record.endswith(('\n', '\r')):
        record += '\n'
    return record.encode('utf-8')


def load_and_split_dataset(csv_file, output_folder, partitions, fold=None,
                           event_col='grf_fail', time_col='graftyrs', seed=42):
    """Read CSV rows once into RAM as bytes, then write the selected folds."""
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions < 2:
        raise ValueError('--cv must be an integer of at least 2')
    if fold is not None and (isinstance(fold, bool) or not isinstance(fold, int) or not 1 <= fold <= partitions):
        raise ValueError('--fold must be between 1 and --cv')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('--seed must be an integer from 0 through 4294967295')
    if event_col == time_col:
        raise ValueError('--event-col and --time-col must be different columns')
    selected_folds = list(range(partitions)) if fold is None else [fold - 1]
    source_path, output_path = Path(csv_file), Path(output_folder)
    initial_stat = source_path.stat()
    source_identity = (initial_stat.st_ino, initial_stat.st_size, initial_stat.st_mtime_ns)
    started = time.monotonic()
    print(f'Loading into RAM {source_path} '
          'to count and validate CSV records...', flush=True)
    rows = 0
    source_bytes = 0
    cached_rows = []
    events, times = [], []
    with source_path.open('r', encoding='utf-8-sig', newline='') as source:
        records = _records(source)
        try:
            header, header_record = next(records)
        except StopIteration:
            raise ValueError('Input is empty') from None
        if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
            raise ValueError('Input has empty or duplicate column names')
        for name in (event_col, time_col):
            if name not in header:
                raise ValueError(f'Missing required survival column: {name}')
        event_index, time_index = header.index(event_col), header.index(time_col)
        tx_index = header.index('TX_ID') if 'TX_ID' in header else None
        header_bytes = _payload(header_record)
        source_bytes += len(header_bytes)
        for fields, record in records:
            rows += 1
            if len(fields) != len(header):
                raise ValueError(f'Data row {rows}: expected {len(header)} fields, got {len(fields)}')
            location = f'Data row {rows}'
            if tx_index is not None:
                location += f' (TX_ID={fields[tx_index]!r})'
            try:
                event_value = float(fields[event_index])
            except ValueError:
                raise ValueError(f'{location}: {event_col} must be numeric 0 or 1') from None
            if event_value not in (0.0, 1.0):
                raise ValueError(f'{location}: {event_col} must be 0 (censored) or 1 (event)')
            try:
                time_value = float(fields[time_index])
            except ValueError:
                raise ValueError(f'{location}: {time_col} must be finite, numeric, and nonnegative') from None
            if not np.isfinite(time_value) or time_value < 0:
                raise ValueError(f'{location}: {time_col} must be finite and nonnegative')
            events.append(int(event_value))
            times.append(time_value)
            payload = _payload(record)
            source_bytes += len(payload)
            cached_rows.append(payload)
            if rows % 10000 == 0:
                print(f'Loaded {rows:,} rows ({time.monotonic() - started:.1f}s)', flush=True)
    if rows < partitions:
        raise ValueError(f'Cannot create {partitions} folds from {rows} rows')

    def check_source():
        current = source_path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns) != source_identity:
            raise ValueError('Source changed during partitioning; no completed folds were replaced')

    check_source()
    rng = np.random.RandomState(seed)
    test_fold = np.empty(rows, dtype=np.min_scalar_type(partitions - 1))
    events = np.asarray(events, dtype=np.uint8)
    times = np.asarray(times, dtype=np.float64)
    # RRS: sort time within each status, then randomly send one row from
    # every consecutive K-row block to each of the K folds.
    # https://doi.org/10.1007/s11222-022-10138-7 (Algorithm 1)
    for status in np.unique(events):
        members = np.flatnonzero(events == status)
        if len(members) < partitions:
            raise ValueError(f'{event_col}={status}: only {len(members)} rows for '
                             f'{partitions} folds; reduce --cv to preserve this status in every fold')
        rng.shuffle(members)  # Randomize ties before the stable time sort.
        members = members[np.argsort(times[members], kind='stable')]
        for start in range(0, len(members), partitions):
            block = members[start:start + partitions]
            test_fold[block] = rng.permutation(partitions)[:len(block)]
    fold_sizes = np.bincount(test_fold.astype(np.int64), minlength=partitions)
    print(f'RRS: block size={partitions}; seed={seed}; '
          f'test fold sizes {fold_sizes.tolist()}', flush=True)

    file_name = source_path.stem
    # Every train/test pair contains two copies of the header and one of the data.
    expected_bytes = len(selected_folds) * (source_bytes + len(header_bytes))
    print(f'Cached {source_bytes / 1024 ** 3:.2f} GiB of CSV bytes in RAM; '
          'no second source read is needed.', flush=True)
    print(f'Writing folds {[index + 1 for index in selected_folds]} of {partitions}: '
          f'{rows:,} rows x {len(header):,} columns; '
          f'approximately {expected_bytes / 1024 ** 3:.2f} GiB of CSV output. '
          'Each selected fold writes one train/test pair.', flush=True)
    output_path.mkdir(parents=True, exist_ok=True)
    destinations = []
    write_id = uuid4().hex
    try:
        for fold_index in selected_folds:
            train_destination = output_path / f'{file_name}_CV_{fold_index + 1}_Train.csv'
            test_destination = output_path / f'{file_name}_CV_{fold_index + 1}_Test.csv'
            train_path = train_destination.with_name(f'.{train_destination.name}.{write_id}.partial')
            test_path = test_destination.with_name(f'.{test_destination.name}.{write_id}.partial')
            destinations.extend(((train_path, train_destination), (test_path, test_destination)))
            with train_path.open('wb', buffering=1024 * 1024) as train_handle, \
                    test_path.open('wb', buffering=1024 * 1024) as test_handle:
                train_handle.write(header_bytes)
                test_handle.write(header_bytes)
                written_rows = 0
                written_tests = 0
                for written_rows, payload in enumerate(cached_rows, start=1):
                    if test_fold[written_rows - 1] == fold_index:
                        test_handle.write(payload)
                        written_tests += 1
                    else:
                        train_handle.write(payload)
                    if written_rows % 10000 == 0:
                        print(f'Fold {fold_index + 1}: partitioned {written_rows:,}/{rows:,} rows '
                              f'({100 * written_rows / rows:.1f}%; {time.monotonic() - started:.1f}s)', flush=True)
                if written_rows != rows or written_tests != fold_sizes[fold_index]:
                    raise ValueError('Fold row counts do not match the validated source')
            check_source()
        # Each array task owns its reports, just as it owns its train/test pair.
        report_dir = output_path / 'partition_reports'
        report_dir.mkdir(exist_ok=True)
        for fold_index in selected_folds:
            summary = []
            for split in ('Train', 'Test'):
                in_split = test_fold == fold_index
                if split == 'Train':
                    in_split = ~in_split
                n = int(in_split.sum())
                failures = int(events[in_split].sum())
                for status in np.unique(events):
                    observed = times[in_split & (events == status)]
                    quantiles = np.quantile(observed, [0, .25, .5, .75, 1]).tolist()
                    summary.append({'split': split, 'rows': n, 'events': failures,
                                    'censored': n - failures, 'event_fraction': failures / n,
                                    'event_status': int(status), 'status_rows': len(observed),
                                    'time_min': quantiles[0], 'time_q25': quantiles[1],
                                    'time_median': quantiles[2], 'time_q75': quantiles[3],
                                    'time_max': quantiles[4]})
            metadata = {'source': str(source_path.resolve()), 'rows': rows,
                        'source_size_bytes': initial_stat.st_size,
                        'source_mtime_ns': initial_stat.st_mtime_ns,
                        'cv': partitions, 'fold': fold_index + 1, 'seed': seed,
                        'method': 'rrs', 'block_size': partitions,
                        'event_col': event_col, 'time_col': time_col,
                        'event_encoding': '0=censored, 1=event',
                        'balance': summary}
            report_name = f'{file_name}_CV_{fold_index + 1}'
            json_destination = report_dir / f'{report_name}_partition.json'
            csv_destination = report_dir / f'{report_name}_balance.csv'
            json_path = json_destination.with_name(f'.{json_destination.name}.{write_id}.partial')
            csv_path = csv_destination.with_name(f'.{csv_destination.name}.{write_id}.partial')
            destinations.extend(((json_path, json_destination), (csv_path, csv_destination)))
            json_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            with csv_path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
        check_source()
        # Adjacent files are renamed, not copied. Publish after all writes succeed.
        for temporary, destination in destinations:
            os.replace(temporary, destination)
    finally:
        # Only clean up this invocation's files; concurrent fold jobs own theirs.
        for temporary, _ in destinations:
            temporary.unlink(missing_ok=True)
    print(f'Finished {len(selected_folds)} folds in {time.monotonic() - started:.1f}s: {output_path}', flush=True)
    return {'rows': rows, 'columns': len(header), 'folds': len(selected_folds),
            'fold_indices': [index + 1 for index in selected_folds],
            'test_rows': [int(fold_sizes[index]) for index in selected_folds], 'output_bytes': expected_bytes}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d', dest='datafile', help='Prepared plain CSV input', required=True)
    parser.add_argument('--o', dest='outputfolder', help='Directory for fold CSVs', required=True)
    parser.add_argument('--cv', dest='partitions', help='Number of CV partitions', type=int, default=10)
    parser.add_argument('--fold', type=int, help='Write only this 1-based fold (default: all folds)')
    parser.add_argument('--event-col', '--c', default='grf_fail', help='0=censored, 1=event')
    parser.add_argument('--time-col', '--ol', default='graftyrs', help='Observed event/censoring follow-up time')
    parser.add_argument('--seed', type=int, default=42, help='Partition seed (default: 42)')
    options = parser.parse_args(argv[1:])
    try:
        load_and_split_dataset(options.datafile, options.outputfolder, options.partitions,
                               fold=options.fold, event_col=options.event_col,
                               time_col=options.time_col, seed=options.seed)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
