import json, ast, pickle, time, warnings, os, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from multiprocessing import Pool
REPO = os.environ.get('REPO_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
S = os.environ.get('OUT_DIR', '.')  # hourly archive pickles in, results out
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.1f}s] {m}', flush=True)

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
g = {'__name__': '__axis__', 'np': np, 'pd': pd}
exec(compile(load_notebook_defs(f'{REPO}/signal_evaluation_axis (3).ipynb'), 'axis', 'exec'), g)
D = pickle.load(open(f'{S}/feature_screen_inputs.pkl', 'rb'))
X, TG, TS, FO, BL = D['X'], D['TG'], D['TS'], D['FO'], D['blocks']
NB = len(TS)

def per_day_residual(x, controls, ts):
    out = np.full(len(x), np.nan)
    for t in np.unique(ts):
        m = ts == t
        if m.sum() < 5: out[m] = 0.0; continue
        rx = rankdata(x[m])
        RX = np.column_stack([np.ones(m.sum())] + [rankdata(c[m]) for c in controls])
        beta, *_ = np.linalg.lstsq(RX, rx, rcond=None); out[m] = rx - RX @ beta
    return out
mom = []
for i in range(NB):
    zs = []
    for h in (1, 3, 6, 12, 24):
        r = X[f'RET_{h}'][i]; sd = np.nanstd(r); zs.append(r / sd if sd > 1e-9 else np.zeros_like(r))
    mom.append(np.mean(zs, axis=0))
X['MOM_ORTH_NATR*'] = [per_day_residual(mom[i], [X['NATR_14'][i]], TS[i]) for i in range(NB)]
FEATS = FO + ['MOM_ORTH_NATR*']

def job(args):
    f, kind, t = args
    wr = [(BL[i][0][:10], X[f][i], TG[(kind, t)][i]) for i in range(NB)]
    rep = g['evaluate_windows'](wr, n_shuffles=200, min_samples=10, seed=42, verbose=False)
    pw = rep['per_window']
    return (f, kind, t, pw)

def summarize(pw):
    ok = pw[pw.status == 'ok'] if 'status' in pw else pw.iloc[0:0]
    if len(ok) == 0: return dict(mean_ic=np.nan, frac_sig=0.0, same_sign=0, n_ok=0, passes=False)
    ics = ok.ic.to_numpy(); s = np.sign(ics.mean())
    frac = float(np.mean((ok.p_value.to_numpy() < 0.05) & (np.sign(ics) == s)))
    same = int((np.sign(ics) == s).sum())
    return dict(mean_ic=float(ics.mean()), frac_sig=frac, same_sign=same, n_ok=len(ok),
                passes=bool(same == len(ok) and frac >= 0.34 and len(ok) >= 5))

if __name__ == '__main__':
    jobs = [(f, k, t) for f in FEATS for k in ('abs', 'rel') for t in ('high', 'low', 'close')]
    log(f'{len(jobs)} jobs, {NB} blocks')
    rows, pws = [], {}
    with Pool(4) as p:
        for n, (f, k, t, pw) in enumerate(p.imap_unordered(job, jobs, chunksize=4)):
            pws[(f, k, t)] = pw
            idx = np.arange(len(pw))
            for period, sel in (('all', idx >= 0), ('in', idx < 30), ('out', idx >= 30)):
                rows.append(dict(feature=f, kind=k, target=t, period=period, **summarize(pw[sel])))
            if n % 50 == 0: log(f'{n}/{len(jobs)}')
    res = pd.DataFrame(rows)
    res.to_pickle(f'{S}/feature_screen_results.pkl')
    with open(f'{S}/feature_screen_per_window.pkl', 'wb') as fh: pickle.dump(pws, fh)
    log('DONE')
