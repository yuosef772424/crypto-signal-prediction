"""
PURPOSE:  Full pandas_ta_classic survey: dummy OHLCV, parameter sweeps, run_survey classifier (ok / constant / all_nan / error) and generation of candidate dicts for the lab.
TAGS:     robustness_agg, build_robust_candidate_dicts, run_survey, make_dummy_ohlcv, param_variants, build_candidate_dicts, LENGTH_SWEEP, FAST_SLOW_SWEEP, pandas_ta_classic, survey, indicators
PITFALLS: robustness_agg / build_robust_candidate_dicts (sections 9-10) were nested in the cells' if-blocks and are lifted verbatim (robustness_agg was _agg). `ta` (pandas_ta_classic) and CATEGORY_HYPOTHESES come from the survey notebook's namespace at call time; the dummy data is random, so this checks that indicators run, not that they predict. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from pandas_ta_full_survey.ipynb cells 4, 6 and 14 (sections 2, 3, 6).
"""
import inspect
import numpy as np
import pandas as pd


def make_dummy_ohlcv(n=300, seed=0):
    """مسيرة عشوائية واقعية بما يكفي (اتجاه + تقلّب + حجم) لاختبار أن كل
    مؤشر يُنتج قيماً متغيّرة فعلاً، لا لقياس أي قوة تنبّئية — ذلك يحتاج
    بيانات حقيقية عبر محور signal_evaluation_axis، خطوة لاحقة منفصلة."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    openp = np.r_[close[0], close[:-1]]
    high = np.maximum(openp, close) * (1 + rng.random(n) * 0.01)
    low = np.minimum(openp, close) * (1 - rng.random(n) * 0.01)
    volume = rng.random(n) * 1000 + 100
    return pd.DataFrame({'open': openp, 'high': high, 'low': low, 'close': close,
                         'volume': volume}, index=idx)
LENGTH_SWEEP = [7, 14, 21, 50]
FAST_SLOW_SWEEP = [(12, 26, 9), (6, 13, 5), (24, 52, 18)]


def param_variants(sig_params):
    names = set(sig_params)
    if 'length' in names:
        return [{'length': L} for L in LENGTH_SWEEP]
    if {'fast', 'slow'}.issubset(names):
        variants = []
        for f, s, sg in FAST_SLOW_SWEEP:
            kw = {'fast': f, 'slow': s}
            if 'signal' in names:
                kw['signal'] = sg
            variants.append(kw)
        return variants
    return [{}]


def run_survey(df=None, indicators=None):
    """يستدعي كل مؤشر (بكل توليفة فترات) على df، ويُصنِّف الناتج:
    'ok' (قيم صالحة متغيّرة)، 'constant' (نجح لكن بلا معلومة فعلية)،
    'all_nan'، أو 'error' (استثناء صريح — عادة معامل خارجي إلزامي مفقود)."""
    df = df if df is not None else DUMMY_DF
    names = (sorted({nm for cat in ta.Category.values() for nm in cat})
             if indicators is None else indicators)
    cat_of = {nm: cat for cat, lst in ta.Category.items() for nm in lst}
    rows = []
    for name in names:
        fn = getattr(df.ta, name, None)
        if fn is None:
            rows.append(dict(indicator=name, category=cat_of.get(name, '?'), params={},
                             status='no_attr', n_cols=0, n_valid=0, is_constant=None,
                             error='', columns=[]))
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as e:
            rows.append(dict(indicator=name, category=cat_of.get(name, '?'), params={},
                             status='no_signature', n_cols=0, n_valid=0, is_constant=None,
                             error=str(e), columns=[]))
            continue
        for kw in param_variants(sig.parameters):
            try:
                out = fn(**kw)
            except Exception as e:
                rows.append(dict(indicator=name, category=cat_of.get(name, '?'), params=kw,
                                 status='error', n_cols=0, n_valid=0, is_constant=None,
                                 error=f'{type(e).__name__}: {e}', columns=[]))
                continue
            if out is None:
                rows.append(dict(indicator=name, category=cat_of.get(name, '?'), params=kw,
                                 status='returned_none', n_cols=0, n_valid=0, is_constant=None,
                                 error='', columns=[]))
                continue
            out_df = out.to_frame() if isinstance(out, pd.Series) else out
            n_valid = int((~out_df.isna().all(axis=1)).sum())
            col_nunique = out_df.apply(lambda c: c.dropna().nunique())
            is_constant = bool((col_nunique <= 1).all()) if n_valid > 0 else None
            status = ('ok' if (n_valid > 0 and not is_constant)
                     else ('all_nan' if n_valid == 0 else 'constant'))
            rows.append(dict(indicator=name, category=cat_of.get(name, '?'), params=kw,
                             status=status, n_cols=out_df.shape[1], n_valid=n_valid,
                             is_constant=is_constant, error='',
                             columns=list(out_df.columns)))
    return pd.DataFrame(rows)
def build_candidate_dicts(results_df):
    ok = results_df[results_df.status == 'ok'].copy()
    candidates = []
    for _, row in ok.iterrows():
        param_suffix = '_'.join(f'{k}{v}' for k, v in row['params'].items()) or 'default'
        candidates.append({
            'name': f"{row['indicator'].upper()}_{param_suffix}",
            'track': 'literature_mining',
            'source_indicator': row['indicator'],
            'category': row['category'],
            'params': row['params'],
            'primary_column': row['columns'][0] if row['columns'] else None,
            'all_columns': row['columns'],
            'hypothesis': CATEGORY_HYPOTHESES.get(row['category'], 'بلا فرضية فئة محدَّدة.'),
        })
    return candidates


def robustness_agg(g):
    n = len(g)
    n_ok = int((g['status'] == 'ok').sum())
    ok_cols = g.loc[g['status'] == 'ok', 'columns']
    return pd.Series({
        'n_assets_tested': n,
        'n_assets_ok': n_ok,
        'frac_ok': n_ok / n,
        'statuses_seen': sorted(g['status'].unique().tolist()),
        'columns': ok_cols.iloc[0] if len(ok_cols) else [],
    })


def build_robust_candidate_dicts(robustness_df):
    candidates = []
    for _, row in robustness_df.iterrows():
        params = dict(row['params_key'])
        param_suffix = '_'.join(f'{k}{v}' for k, v in params.items()) or 'default'
        candidates.append({
            'name': f"{row['indicator'].upper()}_{param_suffix}",
            'track': 'literature_mining',
            'source_indicator': row['indicator'],
            'category': row['category'],
            'params': params,
            'primary_column': row['columns'][0] if row['columns'] else None,
            'all_columns': row['columns'],
            'n_assets_validated': int(row['n_assets_tested']),
            'hypothesis': CATEGORY_HYPOTHESES.get(row['category'], 'بلا فرضية فئة محدَّدة.'),
        })
    return candidates
