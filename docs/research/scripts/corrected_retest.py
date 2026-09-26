import json, ast, time, warnings, os, glob, sys, pickle
warnings.filterwarnings("ignore")
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODE = sys.argv[1]            # 'absolute' | 'relative'
assert MODE in ('absolute', 'relative')
REPO = os.environ.get('REPO_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
SCRATCH = os.environ.get('OUT_DIR', '.')  # where result pickles are written
DATA_DIR = os.environ.get('DATA_DIR', '/tmp/realdata')  # csv/ and csv_expanded/ daily OHLCV files


def load_notebook_defs(path):
    nb = json.load(open(path, encoding="utf-8"))
    code_text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code_text = "\n".join(l for l in code_text.split("\n") if not l.strip().startswith(("%", "!")))
    tree = ast.parse(code_text)
    keep_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    literal_types = (ast.Tuple, ast.List, ast.Constant, ast.Dict, ast.Set)
    kept, futures = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(node)
        elif isinstance(node, keep_types):
            kept.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, literal_types):
            kept.append(node)
    mod = ast.Module(body=futures[:1] + kept, type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod)


nb = json.load(open(f'{REPO}/crypto_data_pipeline_v6.ipynb', encoding='utf-8'))
code = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
code = '\n'.join(l for l in code.split('\n') if not l.strip().startswith(('!', '%')))
ns = {}
exec(compile(code, 'pipeline', 'exec'), ns)
log('pipeline loaded')
CONFIG = ns['CONFIG']; update_config = ns['update_config']
resample_timeframes = ns['resample_timeframes']
build_dataset = ns['build_dataset']; rolling_splits = ns['rolling_splits']


def by_asset(pkl, col):
    df = pd.read_pickle(f'{SCRATCH}/{pkl}')
    return {a: g[['date', col]].reset_index(drop=True) for a, g in df.groupby('asset')}


update_config({
    'exclude_from_features': [],
    'funding_rate': {'enabled': False}, 'open_interest': {'enabled': False},
    'custom_settings': {
        'supertrend_windows': [10], 'psar_enabled': True,
        'vol_asymmetry_windows': [20], 'direction_consistency_windows': [20],
        'risk_adj_momentum_windows': [20], 'ath_distance_enabled': True,
        'vol_term_structure_pairs': [(3, 30)],
    },
    'market_context': {'enabled': True, 'reference_symbol': 'BTCUSDT', 'return_periods': [1],
                       'corr_windows': [20], 'lag_windows': [2], 'beta_windows': [20]},
    'momentum_rank': {'enabled': True, 'horizons': [6, 24]},
    'market_breadth': {'enabled': True, 'horizons': [24]},
    'intraday_vwap': {'enabled': True, 'as_feature': True,
                      'data': by_asset('hourly_vwap_deviation.pkl', 'vwap_deviation')},
    'intraday_efficiency': {'enabled': True, 'as_feature': True,
                            'data': by_asset('hourly_efficiency_ratio.pkl', 'efficiency_ratio_24h')},
    'intraday_volume_concentration': {'enabled': True, 'as_feature': True,
                                      'data': by_asset('hourly_volume_concentration.pkl',
                                                       'volume_concentration_hhi')},
})
CONFIG['feature_order'] = None

csv_files = sorted(glob.glob(DATA_DIR + '/csv_expanded/*.csv') + glob.glob(DATA_DIR + '/csv/*.csv'))
data = {}
for path in csv_files:
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['datetime_utc'], utc=True)
    data[name] = resample_timeframes(df.set_index('timestamp').sort_index(), config=CONFIG)
log(f'{len(data)} assets loaded')

dataset = build_dataset([{'name': n} for n in data], data=data, config=CONFIG)
FO = dataset['feature_order']
log(f'build_dataset OK: n_features={len(FO)}')
classify = ns['classify_feature']

windows = rolling_splits(dataset, test_span="30D", val_span="15D", initial_train_span="365D",
                         step="30D", max_windows=30, keep_asset_test_separate=False, config=CONFIG)
del dataset
log(f'{len(windows)} windows')

g = {'__name__': '__main__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
exec(compile(load_notebook_defs(f'{REPO}/signal_discovery_lab.ipynb'), 'lab', 'exec'), g)
LAST_COLUMNS = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
g['LAST_COLUMNS'] = LAST_COLUMNS
evaluate_windows = g['evaluate_windows']; extract = g['extract_feature_last_value']
concat_splits = g['concat_splits']; clean_reg_target = g['clean_reg_target']

tests = [dict(window=f'نافذة {i+1}', flat=concat_splits(te)) for i, (_, _, te) in enumerate(windows)]
del windows


def get(flat, f):
    return np.asarray(extract(flat, f, feature_order=FO), dtype='float64')


SINGLE = ['VWAP_DEVIATION', 'SUPERT_DIR_10', 'SUPERT_STRETCH_10', 'PSAR_DIR', 'MOM_RANK_6', 'MOM_RANK_24',
          'VOL_ASYMMETRY_20', 'MKT_BREADTH_24', 'MKT_CORR_20', 'DIR_CONSISTENCY_20', 'EFF_RATIO_24H',
          'PCT_FROM_ATH', 'VOL_TERM_3_30', 'RISK_ADJ_MOM_20', 'MKT_BETA_20', 'MKT_LAG_RET_2', 'VOL_CONC_HHI',
          'NATR_14', 'RET_1', 'RSI_14']
missing = [f for f in SINGLE if f not in FO]
assert not missing, f'missing from feature_order: {missing}'
print('kinds:', {f: classify(f) for f in SINGLE})

CONSTRUCTED = {
    # RSI/BBP arrive already centred by process_windows ((x-50)/50 and (x-0.5)*2),
    # so the correct interaction centres at 0, not at 50 / 0.5.
    'NATR14_x_RSI14_fixed': lambda fl: get(fl, 'NATR_14') * get(fl, 'RSI_14'),
    'NATR14_x_BBP20_fixed': lambda fl: get(fl, 'NATR_14') * get(fl, 'BBP_20_2.0'),
    'MOMENTUM_AGREEMENT_ZSCORE': lambda fl: np.mean(
        [r / (np.nanstd(r) if np.nanstd(r) > 1e-9 else 1.0)
         for r in (get(fl, f'RET_{h}') for h in (1, 3, 6, 12, 24))], axis=0),
}
OLD_BUGGY = {
    'NATR14_x_RSI14_CENTERED(old)': lambda fl: get(fl, 'NATR_14') * (get(fl, 'RSI_14') - 50.0),
    'NATR14_x_BBPCTB20_CENTERED(old)': lambda fl: get(fl, 'NATR_14') * (get(fl, 'BBP_20_2.0') - 0.5),
}

preds = {f: [get(t['flat'], f) for t in tests] for f in SINGLE}
for name, fn in {**CONSTRUCTED, **OLD_BUGGY}.items():
    preds[name] = [fn(t['flat']) for t in tests]
log('predictions extracted')


def actuals(flat, target):
    y = np.asarray(clean_reg_target(flat, target), dtype='float64')
    if MODE == 'absolute':
        return y
    ts = np.asarray(flat['last_candles'])[:, LAST_COLUMNS.index('timestamp')]
    s = pd.Series(y)
    return (s - s.groupby(ts).transform('mean')).to_numpy()


if MODE == 'relative':
    ts0 = np.asarray(tests[5]['flat']['last_candles'])[:, LAST_COLUMNS.index('timestamp')]
    print('assets per timestamp (window 6):', pd.Series(ts0).value_counts().describe().round(1).to_dict())

ACT = {tg: [actuals(t['flat'], tg) for t in tests] for tg in ('close', 'high', 'low')}

# Diagnostics (absolute mode only): how much is each candidate just recent return?
if MODE == 'absolute':
    ret1 = np.concatenate(preds['RET_1']); ret24 = np.concatenate([get(t['flat'], 'RET_24') for t in tests])
    natr = np.concatenate(preds['NATR_14'])
    diag = {}
    for name, p in preds.items():
        allp = np.concatenate(p)
        diag[name] = dict(rho_RET_1=spearmanr(allp, ret1)[0], rho_RET_24=spearmanr(allp, ret24)[0],
                          rho_NATR_14=spearmanr(allp, natr)[0])
    diag_df = pd.DataFrame(diag).T.round(3)
    print(diag_df.to_string())
    diag_df.to_pickle(f'{SCRATCH}/corrected_retest_diag.pkl')

rows, per_window = [], {}
for name in list(SINGLE) + list(CONSTRUCTED) + list(OLD_BUGGY):
    for tg in ('close', 'high', 'low'):
        wr = [(t['window'], preds[name][i], ACT[tg][i]) for i, t in enumerate(tests)]
        rep = evaluate_windows(wr, n_shuffles=1000, min_samples=10, seed=42, verbose=False)
        per_window[(name, tg)] = rep['per_window']
        rows.append(dict(name=name, target=tg, **{k: rep.get(k) for k in
                         ('mean_ic', 'std_ic', 'frac_significant', 'consistent_sign', 'n_ok')}))
        log(f'{name}/{tg}: mean_ic={rep["mean_ic"]:.4f} frac={rep["frac_significant"]:.2f} '
            f'consistent={rep["consistent_sign"]}')

res = pd.DataFrame(rows)
res.to_pickle(f'{SCRATCH}/corrected_retest_{MODE}.pkl')
with open(f'{SCRATCH}/corrected_retest_{MODE}_per_window.pkl', 'wb') as f:
    pickle.dump(per_window, f)
log('DONE')
pd.set_option('display.width', 220)
print(res.to_string(index=False))
