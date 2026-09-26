"""Pre-registered out-of-sample test (docs/research/preregistration_momentum_orth_natr_holdout.md).

Hold-out = consecutive complete 30-day blocks after 2023-07-20 (end of the last
in-sample test window). Features/targets/evaluation identical to
causal_decomposition.py; only the sample masks differ.
"""
import json, ast, time, warnings, os, glob, pickle
warnings.filterwarnings("ignore")
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
resample_timeframes = ns['resample_timeframes']; build_dataset = ns['build_dataset']
sample_timestamps = ns['sample_timestamps']; _take = ns['_take']; add_y_prefix = ns['add_y_prefix']
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
del data
FO = dataset['feature_order']
ts_all = sample_timestamps(dataset)
log(f'dataset: {len(ts_all):,} samples, {ts_all.min():%Y-%m-%d} -> {ts_all.max():%Y-%m-%d}')

CUT = pd.Timestamp('2023-07-20', tz='UTC')   # end of last in-sample test window (inclusive)
def blocks(start, end_limit, direction):
    out, k = [], 0
    while True:
        if direction > 0:
            lo, hi = start + pd.Timedelta(days=30 * k), start + pd.Timedelta(days=30 * (k + 1))
            if hi > end_limit: break
        else:
            hi, lo = start - pd.Timedelta(days=30 * k), start - pd.Timedelta(days=30 * (k + 1))
            if k >= end_limit: break
        out.append((lo, hi)); k += 1
    return out if direction > 0 else out[::-1]

HOLD = blocks(CUT, ts_all.max(), +1)          # (lo, hi]: strictly after 2023-07-20
INS = blocks(CUT, 30, -1)                     # last 30 in-sample blocks, sanity reference only
log(f'hold-out blocks: {len(HOLD)} ({HOLD[0][0]:%Y-%m-%d} .. {HOLD[-1][1]:%Y-%m-%d}); '
    f'in-sample reference blocks: {len(INS)} ({INS[0][0]:%Y-%m-%d} .. {INS[-1][1]:%Y-%m-%d})')


def take_block(lo, hi):
    m = np.asarray((ts_all > lo) & (ts_all <= hi))
    s = _take(dataset, m, dataset['timeframes'], dataset['targets'])
    return add_y_prefix(s)

g = {'__name__': '__main__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
exec(compile(load_notebook_defs(f'{REPO}/signal_discovery_lab.ipynb'), 'lab', 'exec'), g)
LC = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
g['LAST_COLUMNS'] = LC
evaluate_windows = g['evaluate_windows']; extract = g['extract_feature_last_value']
clean_reg_target = g['clean_reg_target']


def get(fl, f): return np.asarray(extract(fl, f, feature_order=FO), dtype='float64')


def momentum_agreement(fl):
    zs = []
    for h in (1, 3, 6, 12, 24):
        r = get(fl, f'RET_{h}'); sd = np.nanstd(r)
        zs.append(r / sd if sd > 1e-9 else np.zeros_like(r))
    return np.mean(zs, axis=0)


def per_day_residual(x, controls, ts):
    out = np.full(len(x), np.nan)
    for t in np.unique(ts):
        m = ts == t
        if m.sum() < 5:
            out[m] = 0.0
            continue
        rx = rankdata(x[m])
        if not controls:
            out[m] = rx - rx.mean(); continue
        RX = np.column_stack([np.ones(m.sum())] + [rankdata(c[m]) for c in controls])
        beta, *_ = np.linalg.lstsq(RX, rx, rcond=None)
        out[m] = rx - RX @ beta
    return out


def prepare(block_list):
    P = []
    for lo, hi in block_list:
        fl = take_block(lo, hi)
        ts = np.asarray(fl['last_candles'])[:, LC.index('timestamp')]
        natr = get(fl, 'NATR_14'); mom = momentum_agreement(fl)
        tg = {}
        for t in ('high', 'low', 'close'):
            y = pd.Series(np.asarray(clean_reg_target(fl, t), dtype='float64'))
            tg[('abs', t)] = y.to_numpy()
            tg[('rel', t)] = (y - y.groupby(ts).transform('mean')).to_numpy()
        P.append(dict(label=f'{lo:%Y-%m-%d}→{hi:%Y-%m-%d}', n=len(ts),
                      n_assets_per_day=float(pd.Series(ts).value_counts().mean()),
                      MOM_ORTH_NATR=per_day_residual(mom, [natr], ts), NATR_14=natr,
                      MOMENTUM_AGREEMENT_ZSCORE=mom, MKT_CORR_60=get(fl, 'MKT_CORR_60'), tg=tg))
    return P


def run(P, tag, specs):
    rows, pws = [], {}
    for feat, kind, t in specs:
        wr = [(p['label'], p[feat], p['tg'][(kind, t)]) for p in P]
        rep = evaluate_windows(wr, n_shuffles=1000, min_samples=10, seed=42, verbose=False)
        pw = rep['per_window']; ok = pw[pw.status == 'ok']
        passed = bool(rep['consistent_sign'] and rep['frac_significant'] >= 0.34 and rep['n_ok'] >= 5)
        same = int((np.sign(ok.ic) == np.sign(rep['mean_ic'])).sum())
        rows.append(dict(period=tag, feature=feat, target=f'{kind}/{t}', mean_ic=rep['mean_ic'],
                         frac_significant=rep['frac_significant'], same_sign=same, n_ok=rep['n_ok'],
                         consistent_sign=rep['consistent_sign'], passes=passed))
        pws[(tag, feat, kind, t)] = pw
        log(f'{tag} | {feat} | {kind}/{t}: ic={rep["mean_ic"]:+.4f} frac={rep["frac_significant"]:.2f} '
            f'same_sign={same}/{rep["n_ok"]} pass={passed}')
    return rows, pws

SPECS = [('MOM_ORTH_NATR', 'rel', 'high'), ('MOM_ORTH_NATR', 'rel', 'low'), ('MOM_ORTH_NATR', 'rel', 'close'),
         ('NATR_14', 'rel', 'high'), ('NATR_14', 'rel', 'low'), ('NATR_14', 'rel', 'close'),
         ('NATR_14', 'abs', 'high'), ('NATR_14', 'abs', 'low'), ('NATR_14', 'abs', 'close'),
         ('MOMENTUM_AGREEMENT_ZSCORE', 'rel', 'high'), ('MOMENTUM_AGREEMENT_ZSCORE', 'rel', 'low'),
         ('MOMENTUM_AGREEMENT_ZSCORE', 'rel', 'close'),
         ('MKT_CORR_60', 'rel', 'low'), ('MKT_CORR_60', 'rel', 'high'), ('MKT_CORR_60', 'rel', 'close')]

PH = prepare(HOLD)
log('hold-out prepared: ' + ', '.join(f"{p['label']}:{p['n']}({p['n_assets_per_day']:.0f}/d)" for p in PH[:3]) + ' ...')
rows, pws = run(PH, 'holdout', SPECS)
PI = prepare(INS)
r2, p2 = run(PI, 'insample_blocks', SPECS[:6])
rows += r2; pws.update(p2)

res = pd.DataFrame(rows)
res.to_pickle(f'{SCRATCH}/holdout.pkl')
with open(f'{SCRATCH}/holdout_per_window.pkl', 'wb') as f: pickle.dump(pws, f)
with open(f'{SCRATCH}/holdout_blocks.json', 'w') as f:
    json.dump([dict(label=p['label'], n=p['n'], assets_per_day=p['n_assets_per_day']) for p in PH], f, indent=1)
log('DONE')
pd.set_option('display.width', 220); print(res.round(3).to_string(index=False))
