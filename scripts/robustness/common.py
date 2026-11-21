"""Shared routing for supplementary analyses; research inputs are read only."""
from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from project_config import input_path, setting, work_dir
from scripts.analysis import common as analysis_common
from scripts.analysis import explanatory as ex

ROOT = work_dir('robustness')
MEASUREMENT = work_dir('measurement')
ANALYSIS = work_dir('analysis')
CONTROLS = analysis_common.CONTROLS
NOVELTY = analysis_common.NOVELTY
OUTCOMES = analysis_common.OUTCOMES


def output(relative):
    result = (ROOT / relative).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError('Output escapes the robustness work directory')
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def save(relative, value):
    target = output(relative)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                 default=analysis_common.default), encoding='utf-8')


def log(message):
    print(message, flush=True)


def source(key, default):
    configured = input_path(key, required=False)
    return Path(configured) if configured is not None else Path(default)


def panel(all_rows=False):
    path = source('panel', MEASUREMENT / 'data/panel_model_ready.parquet')
    if not path.is_file():
        raise FileNotFoundError(f'Measured panel is required: {path}')
    data = pd.read_parquet(path)
    data['patent_id'] = data.patent_id.astype(str)
    if not all_rows:
        data = data.loc[data.is_focal_candidate].copy()
    for col in ['grant_date', 'label_window_end', 'label_available_date']:
        if col in data:
            data[col] = pd.to_datetime(data[col])
    return data.reset_index(drop=True)


@contextmanager
def analysis_outputs():
    """Reuse the unchanged estimator while writing to a separate output tree."""
    previous = analysis_common.ROOT
    analysis_common.ROOT = ROOT
    try:
        yield
    finally:
        analysis_common.ROOT = previous


def fit_variant(data, label, targets=None, extra=None, models=('M2', 'M5'),
                terms_override=None, fe_assignee=False):
    data = data.loc[data.overall_novelty_raw.notna()].copy().reset_index(drop=True)
    if len(data) < 10:
        raise ValueError(f'{label}: fewer than ten eligible records')
    with analysis_outputs():
        x, specs, fe, _ = ex.prepare(data, label)
        extra = extra or []
        for col in extra:
            values = pd.to_numeric(data[col], errors='coerce')
            missing = values.isna()
            values = values.fillna(values.median() if values.notna().any() else 0.)
            scale = float(values.std(ddof=0))
            x[col] = (values - values.mean()) / (scale if scale > 1e-12 else 1.)
            if missing.any():
                x[col + '_missing5'] = missing.astype(float)
        additions = extra + [c + '_missing5' for c in extra if c + '_missing5' in x]
        if fe_assignee:
            counts = data.assignee_cluster_key.value_counts()
            group = data.assignee_cluster_key.where(
                data.assignee_cluster_key.map(counts) >= 100, 'OTHER')
            dummy = pd.get_dummies(group, prefix='assigneeFE', drop_first=True, dtype=float)
            dummy.index = x.index
            x = pd.concat([x, dummy], axis=1)
            fe = fe + list(dummy.columns)
        targets = targets or {k: OUTCOMES[k] for k in ['citations', 'semantic', 'newfield']}
        results = []
        for target, (col, kind) in targets.items():
            original_outcomes = ex.OUTCOMES
            ex.OUTCOMES = {**original_outcomes, target: (col, kind)}
            try:
                keep = data[col].notna().to_numpy().copy()
                if kind == 'count':
                    while True:
                        old = keep.copy()
                        groups = ['grant_year', 'technical_subfield']
                        if fe_assignee:
                            groups += ['assignee_cluster_key']
                        for group in groups:
                            sums = data.loc[keep, col].groupby(data.loc[keep, group]).sum()
                            keep &= ~data[group].isin(sums[sums == 0].index).to_numpy()
                        if np.array_equal(old, keep):
                            break
                save(f'checks/separation_{label}_{target}.json', {
                    'input_n': len(data), 'fit_n': int(keep.sum()),
                    'excluded_n': int((~keep).sum())})
                for model in models:
                    terms = terms_override[model] if terms_override else specs[model]
                    results.append(ex.fitmodel(data.loc[keep], x.loc[keep], fe,
                                              terms + additions, target, kind, label, model))
            finally:
                ex.OUTCOMES = original_outcomes
    return results
