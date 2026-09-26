"""Screen every pipeline feature (defaults + optional + commented-out indicators)
on the same axis: per-window pooled Spearman IC vs abs/rel high/low/close,
over 68 consecutive 30-day blocks (30 in-sample + 38 hold-out)."""
import json, ast, time, warnings, os, glob, pickle, sys
warnings.filterwarnings("ignore")
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)
import numpy as np, pandas as pd
from scipy.stats import rankdata
REPO = os.environ.get('REPO_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
S = os.environ.get('OUT_DIR', '.')  # hourly archive pickles in, results out
QUICK = len(sys.argv) > 1 and sys.argv[1] == 'quick'
# momentum_orth_natr is evaluated from its pre-registered definition in feature_screen_eval.py


def load_notebook_defs(path):
    nb = json.load(open(path, encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code = "\n".join(l for l in code.split("\n") if not l.strip().startswith(("%", "!")))
    tree = ast.parse(code)
    keep = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    lit = (ast.Tuple, ast.List, ast.Constant, ast.Dict, ast.Set)
    body = [n for n in tree.body if (isinstance(n, keep) and not (isinstance(n, ast.ImportFrom) and n.module == '__future__'))
            or (isinstance(n, ast.Assign) and isinstance(n.value, lit))]
    mod = ast.Module(body=body, type_ignores=[]); ast.fix_missing_locations(mod); return ast.unparse(mod)

nb = json.load(open(f'{REPO}/crypto_data_pipeline_v6.ipynb', encoding='utf-8'))
code = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
code = '\n'.join(l for l in code.split('\n') if not l.strip().startswith(('!', '%')))
ns = {}; exec(compile(code, 'pipeline', 'exec'), ns)
CONFIG = ns['CONFIG']; update_config = ns['update_config']
log('pipeline loaded')

def by_asset(pkl, col):
    df = pd.read_pickle(f'{S}/{pkl}')
    return {a: g[['date', col]].reset_index(drop=True) for a, g in df.groupby('asset')}

ind = dict(CONFIG['indicator_settings'])
EXTRA = {'ema': [9, 26], 'stoch': [(14, 3)], 'macd': [(12, 26, 9)], 'cmf': [20],   # pre-selection defaults
         'sma': [20], 'ppo': [(12, 26, 9)], 'mom': [10], 'roc': [10], 'cci': [20], 'aroon': [14],
         'willr': [14], 'er': [10], 'tsi': [13], 'trix': [14], 'fisher': [9], 'slope': [14], 'chop': [14],
         'atr': [14], 'stdev': [14], 'zscore': [20], 'ui': [14], 'efi': [13], 'adosc': [3], 'qstick': [14],
         'kurtosis': [20], 'rvi': [14]}
ind.update(EXTRA)
update_config({
    'exclude_from_features': [],
    'funding_rate': {'enabled': False}, 'open_interest': {'enabled': False},
    'indicator_settings': ind,
    'static_indicators': ['obv'],
    'custom_settings': {
        # pre-selection defaults (the screen compares every candidate on equal footing)
        'returns': [1, 3, 6, 12, 24], 'range_windows': [14, 50], 'vol_windows': [(6, 24), (12, 48)],
        'volume_window': 20, 'candle_structure': True, 'time_features': True,
        'supertrend_windows': [10], 'psar_enabled': True,
        'vol_asymmetry_windows': [20], 'direction_consistency_windows': [20],
        'risk_adj_momentum_windows': [20], 'ath_distance_enabled': True,
        'vol_term_structure_pairs': [(3, 30)],
    },
    'market_context': {'enabled': True, 'reference_symbol': 'BTCUSDT', 'return_periods': [1],
                       'corr_windows': [20, 60], 'lag_windows': [2], 'beta_windows': [20]},
    'momentum_rank': {'enabled': True, 'horizons': [6, 24]},
    'market_breadth': {'enabled': True, 'horizons': [24]},
    'intraday_vwap': {'enabled': True, 'as_feature': True, 'data': by_asset('hourly_vwap_deviation.pkl', 'vwap_deviation')},
    'intraday_efficiency': {'enabled': True, 'as_feature': True, 'data': by_asset('hourly_efficiency_ratio.pkl', 'efficiency_ratio_24h')},
    'intraday_volume_concentration': {'enabled': True, 'as_feature': True,
                                      'data': by_asset('hourly_volume_concentration.pkl', 'volume_concentration_hhi')},
})
CONFIG['feature_order'] = None

data = {}
paths = sorted(glob.glob(os.environ.get('DATA_DIR', '/tmp/realdata') + '/csv_expanded/*.csv') + glob.glob(os.environ.get('DATA_DIR', '/tmp/realdata') + '/csv/*.csv'))
if QUICK: paths = [p for p in paths if 'BTCUSDT' in p or 'SOLUSDT' in p or 'ETHUSDT' in p]
for path in paths:
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path); df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['datetime_utc'], utc=True)
    data[name] = ns['resample_timeframes'](df.set_index('timestamp').sort_index(), config=CONFIG)
dataset = ns['build_dataset']([{'name': n} for n in data], data=data, config=CONFIG)
del data
FO = list(dataset['feature_order'])
log(f'{len(FO)} features: {FO}')
if QUICK: sys.exit(0)

ts_all = ns['sample_timestamps'](dataset)
CUT = pd.Timestamp('2023-07-20', tz='UTC')
blocks = [(CUT - pd.Timedelta(days=30 * (k + 1)), CUT - pd.Timedelta(days=30 * k)) for k in range(30)][::-1]
k = 0
while CUT + pd.Timedelta(days=30 * (k + 1)) <= ts_all.max():
    blocks.append((CUT + pd.Timedelta(days=30 * k), CUT + pd.Timedelta(days=30 * (k + 1)))); k += 1
log(f'{len(blocks)} blocks')

g = {'__name__': '__main__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
exec(compile(load_notebook_defs(f'{REPO}/signal_discovery_lab.ipynb'), 'lab', 'exec'), g)
LC = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
g['LAST_COLUMNS'] = LC
extract = g['extract_feature_last_value']; clean_reg_target = g['clean_reg_target']

X, TG, TS = {f: [] for f in FO}, {}, []
for lo, hi in blocks:
    m = np.asarray((ts_all > lo) & (ts_all <= hi))
    fl = ns['add_y_prefix'](ns['_take'](dataset, m, dataset['timeframes'], dataset['targets']))
    ts = np.asarray(fl['last_candles'])[:, LC.index('timestamp')]; TS.append(ts)
    last = np.asarray(fl['X_1D'])[:, -1, :].astype('float64')
    for j, f in enumerate(FO): X[f].append(last[:, j])
    for t in ('high', 'low', 'close'):
        y = pd.Series(np.asarray(clean_reg_target(fl, t), dtype='float64'))
        TG.setdefault(('abs', t), []).append(y.to_numpy())
        TG.setdefault(('rel', t), []).append((y - y.groupby(ts).transform('mean')).to_numpy())
del dataset
# sanity: extract == last step
fl_chk = None
log('extracted')
with open(f'{S}/feature_screen_inputs.pkl', 'wb') as fh:
    pickle.dump(dict(X=X, TG=TG, TS=TS, FO=FO, blocks=[(str(a), str(b)) for a, b in blocks]), fh)
log('saved inputs')
