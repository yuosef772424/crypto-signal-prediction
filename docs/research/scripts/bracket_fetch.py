"""لكل (عملة، يوم) لمس فيه السعر +B و-B معاً: أيهما أولاً؟ من شموع 1h في أرشيف Binance."""
import sys, pandas as pd, numpy as np, concurrent.futures as cf, pickle
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "..", "tools"))
import fetch_history_vision_colab as F
S = sys.argv[1]; B = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
F.choose_vision_base()
rows = []
for run in ("eval_timesplit_sig", "eval_volnorm"):
    for sp in ("val", "test"):
        d = pd.read_csv(f"{S}/{run}/signals_{sp}.csv.gz", usecols=["asset", "timestamp", "entry", "fut_high", "fut_low"])
        rows.append(d)
d = pd.concat(rows).drop_duplicates(["asset", "timestamp"])
d["day"] = pd.to_datetime(d["timestamp"]) + pd.Timedelta(days=1)
amb = d[(d.fut_high >= d.entry * (1 + B)) & (d.fut_low <= d.entry * (1 - B))]
print(f"rows {len(d):,} | ambiguous at ±{B:.0%}: {len(amb):,} ({len(amb)/len(d):.1%}) | coins {amb.asset.nunique()}", flush=True)
need = amb.assign(ym=amb.day.dt.strftime("%Y-%m")).groupby(["asset", "ym"])
jobs = []
for (a, ym), g in need:
    y, m = map(int, ym.split("-"))
    jobs.append((a, y, m, g[["day", "entry"]].values.tolist()))
import time
import urllib.parse
def _dl(url):
    base, path = url.split("/data/", 1)
    url = base + "/data/" + urllib.parse.quote(path)
    for i in range(5):
        try:
            return F.download_zip_rows(url)
        except Exception:
            time.sleep(1 + 2 * i)
    print("  [fail]", url, flush=True)
    return None
def first_hit(a, y, m, days):
    url = F.vision_klines_url(a, "1h", y, m)
    rows = _dl(url)
    if rows is None:   # الشهر غير منشور شهرياً ⇒ ملفات يومية
        rows = []
        for day, _ in days:
            r = _dl(F.vision_klines_url(a, "1h", day.year, day.month, day.day))
            rows += r or []
    if not rows:
        return [(a, day, "nodata") for day, _ in days]
    k = pd.DataFrame([r[:5] for r in rows], columns=["t", "o", "h", "l", "c"]).astype(float)
    k["day"] = pd.to_datetime(k.t, unit="ms").dt.normalize()
    out = []
    for day, entry in days:
        h = k[k.day == day].sort_values("t")
        res = "nodata"
        if len(h):
            res = "none"
            for _, x in h.iterrows():
                up, dn = x.h >= entry * (1 + B), x.l <= entry * (1 - B)
                if up and dn: res = "same_hour"; break
                if up: res = "up"; break
                if dn: res = "down"; break
        out.append((a, day, res))
    return out
res = []
with cf.ThreadPoolExecutor(16) as ex:
    for i, r in enumerate(ex.map(lambda j: first_hit(*j), jobs)):
        res += r
        if i % 500 == 0: print(f"  {i}/{len(jobs)}", flush=True)
res = pd.DataFrame(res, columns=["asset", "day", "first"])
print(res["first"].value_counts(), flush=True)
res.to_pickle(f"{S}/bracket_first_{int(B*100)}.pkl")
