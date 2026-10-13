"""Label-informed, full-budget allocation. Never use for strict low-label claims.

The same constrained allocation is used for outer Test and inner Validation.
Binary marginal counts must round their population expectation up or down;
absolute deviations of the (up to eight) joint strata are minimized. No scores,
images, model outputs, seed search, or patient exclusion enter allocation.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from glioma_meta.data.transfer_protocol import (
    LABEL_DEFINITION, _digest, _validate_split, validate_binary_manifest, validate_task_labels,
)
from glioma_meta.data.who_labels import manifest_who_definition

VERSION = 'patient-disjoint-balanced-100pct-v1'
RULE = 'same_marginal_balanced_allocator_outer_and_inner'
COLUMNS = ('idh_label', 'who_label', 'survival_event')
USAGE = 'full_cohort_labels_100pct_only'


def label_digest(frame):
    rows = frame.sort_values('patient_id')[['patient_id', *COLUMNS, 'survival_time']].copy()
    rows['patient_id'] = rows.patient_id.astype(str)
    for name in (*COLUMNS, 'survival_time'):
        rows[name] = rows[name].astype(float)
    return _digest({'rows': rows.values.tolist()})


def balanced_partition(frame, sizes, seed):
    """Allocate all patients once; fail rather than silently relaxing quotas."""
    frame = frame.sort_values('patient_id').reset_index(drop=True)
    n, k = len(frame), len(sizes)
    if sum(sizes) != n or min(sizes) <= 0:
        raise ValueError('Partition sizes must be positive and sum to patient count')
    labels, inverse, counts = np.unique(frame[list(COLUMNS)].to_numpy(int),
                                        axis=0, return_inverse=True, return_counts=True)
    g, m = len(labels), len(labels)*k
    ideal = counts[:, None]*np.asarray(sizes)[None, :]/n
    constraints, lower, upper = [], [], []
    def add(row, lo, hi):
        constraints.append(row)
        lower.append(lo)
        upper.append(hi)
    # x[g,k] integers, d[g,k] nonnegative absolute deviations from ideal.
    for group in range(g):
        row = np.zeros(2*m)
        row[group*k:(group+1)*k] = 1
        add(row, counts[group], counts[group])
    for part, size in enumerate(sizes):
        row = np.zeros(2*m)
        row[part:m:k] = 1
        add(row, size, size)
        for col in range(len(COLUMNS)):
            row = np.zeros(2*m)
            row[part:m:k] = labels[:, col]
            # Integer arithmetic avoids floating rounding at exact integers.
            numerator = int(frame[COLUMNS[col]].sum())*size
            add(row, numerator//n, (numerator+n-1)//n)
    for j, expected in enumerate(ideal.ravel()):
        row = np.zeros(2*m)
        row[j], row[m+j] = 1, -1
        add(row, -np.inf, expected)
        row = np.zeros(2*m)
        row[j], row[m+j] = -1, -1
        add(row, -np.inf, -expected)
    answer = milp(np.r_[np.zeros(m), np.ones(m)],
                  integrality=np.r_[np.ones(m), np.zeros(m)],
                  bounds=Bounds(np.zeros(2*m), np.full(2*m, np.inf)),
                  constraints=LinearConstraint(np.asarray(constraints), lower, upper),
                  options={'time_limit': 30, 'mip_rel_gap': 0.0})
    if not answer.success:
        raise ValueError(f'Cannot satisfy fixed marginal quotas: {answer.message}')
    allocation = np.rint(answer.x[:m]).astype(int).reshape(g, k)
    rng, result = np.random.default_rng(seed), [[] for _ in sizes]
    for group in range(g):
        ids = rng.permutation(frame.loc[inverse == group, 'patient_id'].tolist()).tolist()
        start = 0
        for part, count in enumerate(allocation[group]):
            result[part].extend(ids[start:start+count])
            start += count
    return [sorted(ids) for ids in result]


def build_balanced_protocol(manifest, n_splits=5, validation_fraction=.2, seed=2027):
    validate_binary_manifest(manifest)
    for task in ('idh', 'who', 'survival'):
        validate_task_labels(manifest, task, 'stratified cohort')
    if n_splits != 5 or not 0 < validation_fraction < .5:
        raise ValueError('Use five folds and validation_fraction in (0, 0.5)')
    for name in COLUMNS:
        if len(manifest[name].unique()) != 2 or manifest[name].value_counts().min() < n_splits:
            raise ValueError(f'Not enough patients in both {name} classes for five balanced folds')
    ids = set(manifest.patient_id)
    n = len(ids)
    sizes = [n//n_splits + (i < n % n_splits) for i in range(n_splits)]
    tests = balanced_partition(manifest, sizes, seed)
    outer_records, records = [], []
    for fold, test in enumerate(tests):
        train = sorted(ids - set(test))
        val_size = math.ceil(len(train)*validation_fraction)
        fit, validation = balanced_partition(manifest[manifest.patient_id.isin(train)],
                                             [len(train)-val_size, val_size], seed+fold+1)
        outer = dict(fold=fold, train_ids=train, test_ids=test)
        split = dict(fold=fold, fit_ids=fit, validation_ids=validation, test_ids=test)
        _validate_split(split, outer)
        for role in ('fit_ids', 'validation_ids', 'test_ids'):
            for task in ('idh', 'who', 'survival'):
                validate_task_labels(manifest[manifest.patient_id.isin(split[role])], task, role)
        outer_records.append(outer)
        records.append(split)
    payload = dict(version=VERSION, seed=seed, validation_fraction=validation_fraction,
                   split_rule=RULE, allocation_label_usage=USAGE, minimum_label_fraction=1.0,
                   balanced_columns=list(COLUMNS), allocation_labels_sha256=label_digest(manifest),
                   label_definition={**LABEL_DEFINITION, 'who': manifest_who_definition(manifest)},
                   folds=records)
    return (dict(seed=seed, n_splits=n_splits, split_rule=RULE, folds=outer_records),
            {**payload, 'protocol_id': _digest(payload)})


def validate_balanced_payload(payload, manifest):
    for task in ('idh', 'who', 'survival'):
        validate_task_labels(manifest, task, 'stratified cohort')
    if (payload.get('split_rule') != RULE or payload.get('allocation_label_usage') != USAGE
            or payload.get('minimum_label_fraction') != 1.0
            or payload.get('balanced_columns') != list(COLUMNS)):
        raise ValueError('Invalid full-budget balanced protocol metadata')
    if payload.get('allocation_labels_sha256') != label_digest(manifest):
        raise ValueError('Allocation labels changed; generate a separate protocol')
    fraction = payload.get('validation_fraction', 0)
    if not 0 < fraction < .5:
        raise ValueError('Invalid validation fraction')
    # Verify coverage and quotas on read, not merely a checksum on arbitrary JSON.
    seen = []
    for index, fold in enumerate(payload['folds']):
        train_ids = sorted(set(manifest.patient_id) - set(fold['test_ids']))
        _validate_split(fold, dict(train_ids=train_ids, test_ids=fold['test_ids']))
        expected_test = len(manifest)//5 + (index < len(manifest) % 5)
        if (fold['fold'] != index or len(fold['test_ids']) != expected_test
                or len(fold['validation_ids']) != math.ceil(len(train_ids)*fraction)):
            raise ValueError('Balanced protocol fold index or patient counts invalid')
        pools = [manifest[manifest.patient_id.isin(fold[key])]
                 for key in ('fit_ids', 'validation_ids', 'test_ids')]
        seen.extend(fold['test_ids'])
        if set().union(*(set(p.patient_id) for p in pools)) != set(manifest.patient_id):
            raise ValueError('Balanced protocol must cover the complete cohort')
        train = pd.concat(pools[:2])
        for parent, parts in ((manifest, [pools[2]]), (train, pools[:2])):
            for part in parts:
                for col in COLUMNS:
                    numerator = int(parent[col].sum())*len(part)
                    lo, hi = numerator//len(parent), (numerator+len(parent)-1)//len(parent)
                    if not lo <= part[col].sum() <= hi:
                        raise ValueError('Balanced protocol marginal quota violated')
    if len(payload['folds']) != 5 or len(seen) != len(set(seen)) or set(seen) != set(manifest.patient_id):
        raise ValueError('Every patient must occur in exactly one outer Test fold')


def distribution_report(manifest, protocol):
    rows = []
    for fold in protocol['folds']:
        for role in ('fit', 'validation', 'test'):
            part = manifest[manifest.patient_id.isin(fold[f'{role}_ids'])]
            row = dict(fold=fold['fold'], role=role, n=len(part))
            for col in COLUMNS:
                for label in (0, 1):
                    count = int((part[col] == label).sum())
                    row[f'{col}_{label}_n'] = count
                    row[f'{col}_{label}_fraction'] = count/len(part)
            rows.append(row)
    return pd.DataFrame(rows)
