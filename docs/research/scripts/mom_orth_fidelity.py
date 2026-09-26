"""Does the pipeline's MOM_ORTH_NATR column reproduce the pre-registered feature?"""
import json, ast, time, warnings, os, glob, pickle
warnings.filterwarnings("ignore")
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
REPO = os.environ.get('REPO_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
S = os.environ.get('OUT_DIR', '.')  # hourly archive pickles in, results out
def load_notebook_defs(path):
    nb = json.load(open(path, encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    code = "\n".join(l for l in code.split("\n") if not l.strip().startswith(("%", "!")))
    tree = ast.parse(code)
    keep = (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    lit = (ast.Tuple, ast.List, ast.Constant, ast.Dict, ast.Set)
    body = [n for n in tree.body if (isinstance(n, keep) and not (isinstance(n, ast.ImportFrom) and n.module == '__future__'))
            or (isinstance(n, ast.Assign) and isinstance(n.value, lit))]
    mod = ast.Module(body=body, type_ignores=[]); ast.fix_missing_locations(mod); return ast.unparse(mod)
nb = json.load(open(f'{REPO}/crypto_data_pipeline_v6.ipynb', encoding='utf-8'))
code = '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
code = '\n'.join(l for l in code.split('\n') if not l.strip().startswith(('!', '%')))
ns = {}; exec(compile(code, 'pipeline', 'exec'), ns)
CONFIG = ns['CONFIG']
ns['update_config']({'exclude_from_features': [], 'funding_rate': {'enabled': False}, 'open_interest': {'enabled': False}})
CONFIG['feature_order'] = None
assert CONFIG['momentum_orth_natr']['enabled']
data = {}
for path in sorted(glob.glob(os.environ.get('DATA_DIR', '/tmp/realdata') + '/csv_expanded/*.csv') + glob.glob(os.environ.get('DATA_DIR', '/tmp/realdata') + '/csv/*.csv')):
    name = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path); df.columns = df.columns.str.lower()
    df['timestamp'] = pd.to_datetime(df['datetime_utc'], utc=True)
    data[name] = ns['resample_timeframes'](df.set_index('timestamp').sort_index(), config=CONFIG)
dataset = ns['build_dataset']([{'name': n} for n in data], data=data, config=CONFIG)
FO = list(dataset['feature_order']); j = FO.index('MOM_ORTH_NATR'); jn = FO.index('NATR_14')
log(f'{len(FO)} features; MOM_ORTH_NATR at {j}')
ts_all = ns['sample_timestamps'](dataset)
CUT = pd.Timestamp('2023-07-20', tz='UTC')
blocks = [(CUT - pd.Timedelta(days=30 * (k + 1)), CUT - pd.Timedelta(days=30 * k)) for k in range(30)][::-1]
k = 0
while CUT + pd.Timedelta(days=30 * (k + 1)) <= ts_all.max():
    blocks.append((CUT + pd.Timedelta(days=30 * k), CUT + pd.Timedelta(days=30 * (k + 1)))); k += 1
g = {'__name__': '__main__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
exec(compile(load_notebook_defs(f'{REPO}/signal_discovery_lab.ipynb'), 'lab', 'exec'), g)
LC = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
g['LAST_COLUMNS'] = LC
P, REF, TG, TS = [], [], {}, []
for lo, hi in blocks:
    m = np.asarray((ts_all > lo) & (ts_all <= hi))
    fl = ns['add_y_prefix'](ns['_take'](dataset, m, dataset['timeframes'], dataset['targets']))
    ts = np.asarray(fl['last_candles'])[:, LC.index('timestamp')]; TS.append(ts)
    last = np.asarray(fl['X_1D'])[:, -1, :].astype('float64')
    P.append(last[:, j])
    zs = []
    for h in (1, 3, 6, 12, 24):
        r = last[:, FO.index(f'RET_{h}')]; sd = np.nanstd(r); zs.append(r / sd if sd > 1e-9 else 0 * r)
    mom = np.mean(zs, axis=0); natr = last[:, jn]; ref = np.zeros(len(ts))
    for t in np.unique(ts):
        mm = ts == t
        if mm.sum() < 5: continue
        rx = rankdata(mom[mm]); RX = np.column_stack([np.ones(mm.sum()), rankdata(natr[mm])])
        b, *_ = np.linalg.lstsq(RX, rx, rcond=None); ref[mm] = rx - RX @ b
    REF.append(ref)
    for t in ('high', 'low', 'close'):
        y = pd.Series(np.asarray(g['clean_reg_target'](fl, t), dtype='float64'))
        TG.setdefault(t, []).append((y - y.groupby(ts).transform('mean')).to_numpy())
rho = [spearmanr(P[i], REF[i])[0] for i in range(len(blocks))]
log(f'per-block spearman(pipeline MOM_ORTH_NATR, preregistered def): mean={np.mean(rho):.3f} min={np.min(rho):.3f}')
for t in ('low', 'high', 'close'):
    for name, sel in (('in', range(30)), ('out', range(30, len(blocks)))):
        wr = [(str(blocks[i][0])[:10], P[i], TG[t][i]) for i in sel]
        rep = g['evaluate_windows'](wr, n_shuffles=1000, min_samples=10, seed=42, verbose=False)
        pw = rep['per_window']; ok = pw[pw.status == 'ok']
        same = int((np.sign(ok.ic) == np.sign(rep['mean_ic'])).sum())
        log(f'pipeline MOM_ORTH_NATR rel/{t} {name}: ic={rep["mean_ic"]:+.4f} frac={rep["frac_significant"]:.2f} same={same}/{rep["n_ok"]}')
