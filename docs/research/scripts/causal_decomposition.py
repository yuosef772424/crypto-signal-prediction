"""Causal per-day cross-sectional residualization + control decomposition.

For each candidate feature x and each set of controls Z, within every single
timestamp (same-day assets only -- nothing from other days, fully causal),
x is replaced by the residual of rank(x) regressed on ranks of Z. The
residual is then evaluated against the market-neutral target exactly as
the axis does (per-window pooled Spearman IC + 1000 permutations).
"""
import json, ast, time, warnings, os, glob, pickle
warnings.filterwarnings("ignore")
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REPO = os.environ.get('REPO_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
SCRATCH = os.environ.get('OUT_DIR', '.')  # where result pickles are written
DATA_DIR = os.environ.get('DATA_DIR', '/tmp/realdata')  # csv/ and csv_expanded/ daily OHLCV files


def load_notebook_defs(path):
    nb = json.load(open(path, encoding="utf-8"))
    code_text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code_text = "\n".join(l for l in code_text.split("\n") if not l.strip().startswith(("%", "!")))
    tree = ast.parse(code_text)
    keep = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    lit = (ast.Tuple, ast.List, ast.Constant, ast.Dict, ast.Set)
    kept, fut = [], []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__": fut.append(n)
        elif isinstance(n, keep): kept.append(n)
        elif isinstance(n, ast.Assign) and isinstance(n.value, lit): kept.append(n)
    mod = ast.Module(body=fut[:1] + kept, type_ignores=[]); ast.fix_missing_locations(mod)
    return ast.unparse(mod)


nb = json.load(open(f'{REPO}/crypto_data_pipeline_v6.ipynb', encoding='utf-8'))
code = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
code = '\n'.join(l for l in code.split('\n') if not l.strip().startswith(('!', '%')))
ns = {}; exec(compile(code, 'pipeline', 'exec'), ns)
CONFIG = ns['CONFIG']; update_config = ns['update_config']
resample_timeframes = ns['resample_timeframes']; build_dataset = ns['build_dataset']; rolling_splits = ns['rolling_splits']
log('pipeline loaded')

update_config({
    'exclude_from_features': [], 'funding_rate': {'enabled': False}, 'open_interest': {'enabled': False},
    'custom_settings': {},
    'market_context': {'enabled': True, 'reference_symbol': 'BTCUSDT', 'return_periods': [1],
                       'corr_windows': [20, 60]},
})
CONFIG['feature_order'] = None

data = {}
for path in sorted(glob.glob(DATA_DIR + '/csv_expanded/*.csv') + glob.glob(DATA_DIR + '/csv/*.csv')):
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path); df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['datetime_utc'], utc=True)
    data[name] = resample_timeframes(df.set_index('timestamp').sort_index(), config=CONFIG)
dataset = build_dataset([{'name': n} for n in data], data=data, config=CONFIG)
FO = dataset['feature_order']
windows = rolling_splits(dataset, test_span="30D", val_span="15D", initial_train_span="365D",
                         step="30D", max_windows=30, keep_asset_test_separate=False, config=CONFIG)
del dataset
log(f'{len(windows)} windows')

g = {'__name__': '__main__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
exec(compile(load_notebook_defs(f'{REPO}/signal_discovery_lab.ipynb'), 'lab', 'exec'), g)
LC = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
g['LAST_COLUMNS'] = LC
evaluate_windows = g['evaluate_windows']; extract = g['extract_feature_last_value']
concat_splits = g['concat_splits']; clean_reg_target = g['clean_reg_target']
tests = [concat_splits(te) for _, _, te in windows]; del windows
TS = [np.asarray(fl['last_candles'])[:, LC.index('timestamp')] for fl in tests]


def get(fl, f): return np.asarray(extract(fl, f, feature_order=FO), dtype='float64')


def rel(i, tg):
    y = pd.Series(np.asarray(clean_reg_target(tests[i], tg), dtype='float64'))
    return (y - y.groupby(TS[i]).transform('mean')).to_numpy()


def momentum_agreement(fl):
    zs = []
    for h in (1, 3, 6, 12, 24):
        r = get(fl, f'RET_{h}'); sd = np.nanstd(r)
        zs.append(r / sd if sd > 1e-9 else np.zeros_like(r))
    return np.mean(zs, axis=0)


def per_day_residual(x, controls, ts):
    """Rank-residual of x on controls, fitted separately within each timestamp."""
    out = np.full(len(x), np.nan)
    for t in np.unique(ts):
        m = ts == t
        if m.sum() < 5:
            out[m] = 0.0
            continue
        rx = rankdata(x[m])
        if not controls:
            out[m] = rx - rx.mean()
            continue
        RX = np.column_stack([np.ones(m.sum())] + [rankdata(c[m]) for c in controls])
        beta, *_ = np.linalg.lstsq(RX, rx, rcond=None)
        out[m] = rx - RX @ beta
    return out


FEAT = {i: {'NATR_14': get(fl, 'NATR_14'), 'RET_24': get(fl, 'RET_24'), 'RET_1': get(fl, 'RET_1'),
            'MKT_CORR_20': get(fl, 'MKT_CORR_20'), 'MKT_CORR_60': get(fl, 'MKT_CORR_60'),
            'MOMENTUM_AGREEMENT_ZSCORE': momentum_agreement(fl)} for i, fl in enumerate(tests)}
log('features extracted')

# cross-sectional (within-day) rank correlation of each candidate with each control, averaged
for cand in ('MKT_CORR_20', 'MKT_CORR_60', 'MOMENTUM_AGREEMENT_ZSCORE'):
    for ctrl in ('NATR_14', 'RET_24', 'RET_1'):
        rhos = []
        for i in range(len(tests)):
            for t in np.unique(TS[i]):
                m = TS[i] == t
                if m.sum() >= 5:
                    rhos.append(spearmanr(FEAT[i][cand][m], FEAT[i][ctrl][m])[0])
        print(f'within-day rho({cand}, {ctrl}) = {np.nanmean(rhos):+.3f}')

VARIANTS = {'none': [], 'NATR': ['NATR_14'], 'RET_24': ['RET_24'], 'RET_1': ['RET_1'],
            'all': ['NATR_14', 'RET_24', 'RET_1']}
CANDS = ['MKT_CORR_20', 'MKT_CORR_60', 'MOMENTUM_AGREEMENT_ZSCORE']
rows, per_window = [], {}
for cand in CANDS:
    for vname, ctrls in VARIANTS.items():
        if cand == 'MOMENTUM_AGREEMENT_ZSCORE' and vname in ('RET_24', 'RET_1', 'all'):
            continue  # its own ingredients
        P = [per_day_residual(FEAT[i][cand], [FEAT[i][c] for c in ctrls], TS[i]) for i in range(len(tests))]
        for tg in ('low', 'high', 'close'):
            wr = [(f'نافذة {i+1}', P[i], rel(i, tg)) for i in range(len(tests))]
            rep = evaluate_windows(wr, n_shuffles=1000, min_samples=10, seed=42, verbose=False)
            pw = rep['per_window']; ok = pw[pw.status == 'ok']
            per_window[(cand, vname, tg)] = pw
            passed = bool(rep['consistent_sign'] and rep['frac_significant'] >= 0.34 and rep['n_ok'] >= 5)
            rows.append(dict(candidate=cand, controls=vname, target=tg, mean_ic=rep['mean_ic'],
                             frac_significant=rep['frac_significant'], consistent_sign=rep['consistent_sign'],
                             same_sign=int((np.sign(ok.ic) == np.sign(rep['mean_ic'])).sum()),
                             n_ok=rep['n_ok'], weakest=float(ok.ic.abs().min()), passes=passed))
            log(f'{cand} | ctrl={vname} | {tg}: ic={rep["mean_ic"]:+.4f} frac={rep["frac_significant"]:.2f} '
                f'same_sign={rows[-1]["same_sign"]}/{rep["n_ok"]} pass={passed}')

res = pd.DataFrame(rows)
res.to_pickle(f'{SCRATCH}/causal_decomposition.pkl')
with open(f'{SCRATCH}/causal_decomposition_per_window.pkl', 'wb') as f: pickle.dump(per_window, f)
log('DONE')
pd.set_option('display.width', 220); print(res.round(3).to_string(index=False))
