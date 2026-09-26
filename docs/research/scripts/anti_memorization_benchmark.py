"""Anti-memorization benchmark (PR #7) — docs/research/anti_memorization_pr7.md.

Trains the project's real NIG-TimeNet v2 through the project's real trainer_framework (both loaded from the
repo notebooks, not copies) on daily crypto data, under stress scenarios that make memorization easy, and
reports train / val / test metrics measured the same way (inference mode) so memorization is visible.

Two targets, chosen so one has learnable signal and one is almost pure noise:
  mag : next-day |return| above the cross-sectional median that day   (volatility clustering: learnable)
  dir : next-day return above the cross-sectional median that day      (relative direction: ~noise)
Each gets an NIG regression head and a binary classification head (same as main.ipynb).

Scenarios (--kind):
  full      : all coins, all train days
  little    : 8 coins only in train (val/test keep all coins)
  badnorm   : + raw price level, raw USD volume, coin age, a near-constant flag (bad normalization)
  badnorm_little
Add :shuf to shuffle train labels (pure memorization capacity; test must stay at 0.5).

Configs (--cfg): baseline (MODEL_CONFIG + trainer defaults as main.ipynb before PR #7), robust
(ANTI_MEMORIZATION_CONFIG + ANTI_MEMORIZATION_TRAINER), or any name in EXTRA_CONFIGS below.

Data (one of):
  --history-dir DIR     Drive layout history_1d/<SYMBOL>.csv (timestamp, datetime_utc, open, high, low, close, volume)
  --coinmetrics-dir DIR github.com/coinmetrics/data csv/<asset>.csv files (what the PR numbers were produced on)

Example (Colab, your Drive data):
  python docs/research/scripts/anti_memorization_benchmark.py --history-dir /content/drive/MyDrive/crypto/history_1d \
      --run little:baseline:60 little:robust:60 little:baseline:60:shuf little:robust:60:shuf
Each run appends one JSON line to --out (default anti_memorization_results.jsonl).
"""
import argparse, contextlib, glob, io, json, os, re, shutil, sys, time
import numpy as np
import pandas as pd

REPO = os.environ.get("REPO_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
SEQ = 32
TRAIN_END = pd.Timestamp("2023-06-30")
VAL_END = pd.Timestamp("2024-06-30")
PURGE_DAYS = 2
TARGETS = ("mag", "dir")


# ───────────────────────────── notebooks ─────────────────────────────
_SKIP = re.compile(r"^(%run|drive\.mount\(|%cd)", re.M)


def load_notebook(path, ns, stop_at=None):
    nb = json.load(open(path, encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if stop_at is not None and i >= stop_at:
            break
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if c["cell_type"] != "code" or _SKIP.search(src):
            continue
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
        exec(compile(src, f"{os.path.basename(path)}#cell{i}", "exec"), ns)
    return ns


NS = {}


def project():
    if "GenericTrainer" not in NS:
        with contextlib.redirect_stdout(io.StringIO()):
            load_notebook(os.path.join(REPO, "model_v2 (1).ipynb"), NS)
            nb = json.load(open(os.path.join(REPO, "trainer_framework_v2.ipynb"), encoding="utf-8"))
            smoke = next(i for i, c in enumerate(nb["cells"])
                         if c["cell_type"] == "code" and "def _dummy_model_builder" in "".join(c["source"]))
            load_notebook(os.path.join(REPO, "trainer_framework_v2.ipynb"), NS, stop_at=smoke)
    return NS


# ───────────────────────────── data ─────────────────────────────
def load_history_csv(d, min_days=700):
    px, vol = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        df = pd.read_csv(f)
        t = pd.to_datetime(df["datetime_utc"]) if "datetime_utc" in df else pd.to_datetime(df["timestamp"], unit="ms")
        idx = pd.DatetimeIndex(t).normalize()
        keep = ~idx.duplicated()
        if keep.sum() < min_days:
            continue
        a = os.path.basename(f)[:-4]
        px[a] = pd.Series(df["close"].values[keep], index=idx[keep])
        qv = df["quote_volume"] if "quote_volume" in df else df["volume"] * df["close"]
        vol[a] = pd.Series(qv.values[keep], index=idx[keep])
    px = pd.DataFrame(px).sort_index()
    return px, pd.DataFrame(vol).reindex(index=px.index, columns=px.columns)


def load_coinmetrics(d, min_days=700, start="2018-01-01"):
    px, vol = {}, {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        a = os.path.basename(f)[:-4]
        if a in ("usdt", "usdc", "dai", "busd", "tusd"):
            continue
        df = pd.read_csv(f, low_memory=False)
        col = "PriceUSD" if "PriceUSD" in df else ("ReferenceRateUSD" if "ReferenceRateUSD" in df else None)
        if col is None:
            continue
        df = df.set_index(pd.to_datetime(df["time"]))
        s = pd.to_numeric(df[col], errors="coerce")
        s = s[(s.index >= start) & (s > 0)].dropna()
        if len(s) < min_days:
            continue
        px[a] = s
        if "volume_reported_spot_usd_1d" in df:
            vol[a] = pd.to_numeric(df["volume_reported_spot_usd_1d"], errors="coerce")
    px = pd.DataFrame(px).sort_index()
    return px, pd.DataFrame(vol).reindex(index=px.index, columns=px.columns)


def _rsi(r, n=14):
    up = r.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return up / (up + dn + 1e-12)


def build_features(px, vol, bad_norm=False):
    """Close+volume features only (both data sources have them). Causal: row t uses data up to t."""
    lr = np.log(px).diff()
    lr = lr.where(lr.abs() < 1.5)
    mkt = lr.mean(axis=1)
    btc = next((lr[c] for c in lr if c.lower() in ("btc", "btcusdt")), mkt)
    feats = {}
    for a in px.columns:
        r = lr[a]
        f = pd.DataFrame(index=px.index)
        f["RET_1"] = r
        f["RET_5"] = r.rolling(5).sum()
        f["RET_20"] = r.rolling(20).sum()
        f["VOL_20"] = r.rolling(20).std()
        f["NATR_PROXY_14"] = r.abs().rolling(14).mean()
        f["RSI_14"] = _rsi(r) - 0.5
        f["MA_DIST_20"] = px[a] / px[a].rolling(20).mean() - 1
        lv = np.log(vol[a].where(vol[a] > 0))
        f["VOLUME_Z"] = (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + 1e-6)
        f["MKT_RET_1"] = mkt
        f["BTC_RET_1"] = btc
        f["REL_RET_1"] = r - mkt
        if bad_norm:
            f["PRICE_LEVEL"] = px[a]
            f["VOLUME_USD"] = vol[a]
            f["AGE_DAYS"] = np.arange(len(f), dtype="float64") - np.argmax(px[a].notna().values)
            f["LISTED_1Y"] = (f["AGE_DAYS"] > 365).astype("float64")
        feats[a] = f
    fut = lr.shift(-1)
    mag = fut.abs()
    return feats, {"mag": mag.sub(mag.median(axis=1), axis=0), "dir": fut.sub(fut.median(axis=1), axis=0)}


def make_windows(feats, targets):
    Xs, ys, days, aid = [], {k: [] for k in targets}, [], []
    for ai, a in enumerate(feats):
        f = feats[a].replace([np.inf, -np.inf], np.nan)
        v = f.values.astype("float32")
        ok = ~np.isnan(v).any(1)
        tv = np.stack([targets[k][a].reindex(f.index).values for k in targets], 1)
        idx = np.arange(SEQ - 1, len(f))
        good = np.array([ok[i - SEQ + 1:i + 1].all() for i in idx]) & ~np.isnan(tv[idx]).any(1)
        idx = idx[good]
        if not len(idx):
            continue
        W = np.lib.stride_tricks.sliding_window_view(v, SEQ, axis=0).transpose(0, 2, 1)
        Xs.append(W[idx - SEQ + 1])
        for j, k in enumerate(targets):
            ys[k].append(tv[idx, j])
        days.append(f.index.values[idx])
        aid.append(np.full(len(idx), ai))
    return (np.concatenate(Xs).astype("float32"), {k: np.concatenate(v).astype("float32") for k, v in ys.items()},
            np.concatenate(days), np.concatenate(aid), list(next(iter(feats.values())).columns))


def split(X, y, days, aid):
    d64 = lambda t: t.to_datetime64()
    masks = {"train": days <= d64(TRAIN_END - pd.Timedelta(days=PURGE_DAYS)),
             "val": (days > d64(TRAIN_END)) & (days <= d64(VAL_END - pd.Timedelta(days=PURGE_DAYS))),
             "test": days > d64(VAL_END)}
    return {n: subset({"X": X, "y": y, "days": days, "aid": aid}, m) for n, m in masks.items()}


def subset(p, m):
    return {"X": p["X"][m], "y": {k: v[m] for k, v in p["y"].items()}, "days": p["days"][m], "aid": p["aid"][m]}


def scenario(px, vol, kind):
    feats, tg = build_features(px, vol, bad_norm=kind.startswith("badnorm"))
    X, y, days, aid, names = make_windows(feats, tg)
    S = split(X, y, days, aid)
    if kind.endswith("little"):
        coins = np.random.default_rng(7).choice(np.unique(S["train"]["aid"]), 8, replace=False)
        S["train"] = subset(S["train"], np.isin(S["train"]["aid"], coins))
    S["feature_names"] = names
    return S


# ───────────────────────────── metrics ─────────────────────────────
def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y) > 0
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def daily_ic(y, s, days):
    df = pd.DataFrame({"y": y, "s": s, "d": days})
    ic = df.groupby("d")[["y", "s"]].apply(lambda g: g["y"].rank().corr(g["s"].rank()) if len(g) >= 8 else np.nan)
    ic = ic.dropna()
    return float(ic.mean()), float(ic.mean() / (ic.std() + 1e-12) * np.sqrt(len(ic)))


def evaluate(model, part, max_n=None, seed=0):
    idx = np.arange(len(part["X"]))
    if max_n and len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), max_n, replace=False))
    out = model.predict(part["X"][idx], batch_size=2048, verbose=0)
    res = {}
    for t in TARGETS:
        yt = part["y"][t][idx]
        p = np.asarray(out[f"y_{t}_class_logits"]).ravel()
        res[f"{t}_auc"] = auc(yt, p)
        res[f"{t}_acc"] = float(np.mean((p >= 0.5) == (yt > 0)))
        res[f"{t}_ic"], res[f"{t}_ic_t"] = daily_ic(yt, np.asarray(out[f"y_{t}"]).ravel(), part["days"][idx])
        res[f"{t}_p_std"] = float(np.std(p))
    return res


# ───────────────────────────── configs ─────────────────────────────
def configs():
    ns = project()
    robust_trainer = {"optimizer": {"weight_decay": 0.05},
                      "callbacks": {"early_stopping": {"weights_snapshot": "ema_weights"}},
                      "_class": {"label_smoothing": 0.1}}
    am = dict(ns["ANTI_MEMORIZATION_CONFIG"])
    cfgs = {"baseline": ({}, {}), "robust": (am, robust_trainer)}
    cfgs.update(EXTRA_CONFIGS(am, robust_trainer))
    return cfgs


def EXTRA_CONFIGS(am, rt):
    """Ablations used in the PR doc: robust minus one component, and baseline plus one component."""
    import copy
    minus = lambda **kw: ({**am, **kw}, rt)
    nt = lambda f: (am, f(copy.deepcopy(rt)))
    return {
        "robust_level": ({**am, "level_passthrough": True}, rt),
        "robust_level_bn": ({**am, "level_passthrough": True, "level_norm": "batch"}, rt),
        "robust_level_bn_lr": ({**am, "level_passthrough": True, "level_norm": "batch"},
                               {**rt, "optimizer": {**rt["optimizer"], "lr_initial": 3e-4}}),
        "robust_bn": ({**am, "level_norm": "batch"}, rt),
        "base_bn": ({"level_norm": "batch"}, {}),
        "robust_lr": nt(lambda r: {**r, "optimizer": {**r["optimizer"], "lr_initial": 3e-4}}),
        "robust_level_lr": ({**am, "level_passthrough": True}, {**rt, "optimizer": {**rt["optimizer"], "lr_initial": 3e-4}}),
        "robust_no_small": minus(d_model=128, num_layers=4, head_hidden=128, class_head_hidden=64),
        "robust_no_inputreg": minus(input_noise_std=0.0, feature_dropout=0.0),
        "robust_no_dropout": minus(dropout=0.1),
        "robust_no_ls": nt(lambda r: {**r, "_class": {}}),
        "robust_no_wd": nt(lambda r: {**r, "optimizer": {"weight_decay": 1e-4}}),
        "robust_no_ema": nt(lambda r: {**r, "callbacks": {}}),
        "robust_flood": nt(lambda r: {**r, "_class": {**r["_class"], "flood_level": 0.69}}),
        "base_small": ({"d_model": 64, "num_layers": 2, "head_hidden": 64, "class_head_hidden": 32}, {}),
        "base_level": ({"level_passthrough": True}, {}),
    }


def target_configs(extra_class):
    cfg = {}
    for t in TARGETS:
        cfg[f"{t}_reg"] = {"true_key": f"y_{t}_reg", "task_type": "evidential",
                           "output_keys": {"mu": f"y_{t}", "nu": f"y_{t}_nu", "alpha": f"y_{t}_alpha",
                                           "beta": f"y_{t}_beta", "confidence": f"y_{t}_confidence"},
                           "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib"}
        cfg[f"{t}_class"] = {"true_key": f"y_{t}_class", "task_type": "classification", "binary": True,
                             "output_keys": {"logits": f"y_{t}_class_logits"}, **(extra_class or {})}
    return cfg


# ───────────────────────────── train ─────────────────────────────
def train_eval(name, S, model_cfg, trainer_cfg, epochs, seed=0, shuffle=False, batch_size=256, run_root="/tmp/am_runs"):
    ns = project()
    import tensorflow as tf
    tr, va, te = S["train"], S["val"], S["test"]
    scale = {t: float(np.std(tr["y"][t])) for t in TARGETS}
    to_y = lambda p: {**{f"y_{t}_reg": (p["y"][t] / scale[t]).astype("float32") for t in TARGETS},
                      **{f"y_{t}_class": (p["y"][t] > 0).astype("float32") for t in TARGETS}}
    ytr = to_y(tr)
    tr_eval = tr
    if shuffle:
        perm = np.random.default_rng(seed + 123).permutation(len(tr["X"]))
        ytr = {k: v[perm] for k, v in ytr.items()}
        tr_eval = {**tr, "y": {t: tr["y"][t][perm] for t in TARGETS}}
    mcfg = {"price_targets": TARGETS, "enforce_order": False,
            "head_types": {t: ["nig_regression", "binary_classification"] for t in TARGETS}, **(model_cfg or {})}
    trainer_cfg = dict(trainer_cfg or {})
    extra_class = trainer_cfg.pop("_class", {})
    run_dir = os.path.join(run_root, name)
    shutil.rmtree(run_dir, ignore_errors=True)
    cfg = ns["build_config"](ns["deep_update"]({
        "run": {"run_dir": run_dir, "epochs": epochs, "batch_size": batch_size, "train_mode": "new", "verbose": 0,
                "seed": seed},
        "targets": target_configs(extra_class),
        "loss": {"use_uncertainty_weighting": True, "schedules": {
            "lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
            "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5, "schedule": "cosine"}}},
        "callbacks": {"early_stopping": {"monitor": "val_raw_loss", "mode": "min"}, "metrics_log_every": 10 ** 6},
        "checkpoint": {"save_every": 10 ** 6, "save_best_weights": False},
    }, trainer_cfg))
    ns["TRAINER_REGISTRY"].pop(run_dir, None)
    tf.keras.utils.set_random_seed(seed)
    ds = lambda X, y, sh: (tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=seed) if sh
                           else tf.data.Dataset.from_tensor_slices((X, y))).batch(batch_size, drop_remainder=sh).prefetch(2)
    train_ds, val_ds = ds(tr["X"], ytr, True), ds(va["X"], to_y(va), False)
    builder = lambda: ns["build_model_fn"](tr["X"].shape[1], tr["X"].shape[2], config=mcfg)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer, callbacks, ie = ns["build_training_system"](builder, cfg, next(iter(train_ds)))
    last, traj = {}, []
    every = max(epochs // 6, 1)

    class Probe(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            last["w"] = [w.copy() for w in trainer.model.get_weights()]
            if (epoch + 1) % every == 0 or epoch + 1 == epochs:
                r = {"epoch": epoch + 1}
                r.update({f"train_{k}": v for k, v in evaluate(trainer.model, tr_eval, 8000).items()})
                r.update({f"test_{k}": v for k, v in evaluate(trainer.model, te, 12000).items()})
                traj.append(r)

    cbs = [c for c in callbacks if type(c).__name__ != "EpochCheckpointCallback"]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        hist = trainer.fit(train_ds, validation_data=val_ds, initial_epoch=ie, epochs=epochs,
                           callbacks=[Probe()] + cbs, verbose=0)
    best = next(c for c in cbs if type(c).__name__ == "BestModelTracker")
    res = {"name": name, "seconds": time.time() - t0, "epochs_run": len(hist.history.get("loss", [])),
           "best_epoch": best.best_epoch, "n_params": int(trainer.model.count_params()), "n_train": len(tr["X"])}
    for tag in ("best", "last"):
        if tag == "last":
            trainer.model.set_weights(last["w"])
        for pn, part in (("train", tr_eval), ("val", va), ("test", te)):
            for k, v in evaluate(trainer.model, part, 20000 if pn == "train" else None).items():
                res[f"{tag}_{pn}_{k}"] = v
    res["trajectory"] = traj
    shutil.rmtree(run_dir, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--history-dir")
    g.add_argument("--coinmetrics-dir")
    ap.add_argument("--run", nargs="+", required=True, help="kind:cfg:epochs[:shuf][:seed=N]")
    ap.add_argument("--out", default="anti_memorization_results.jsonl")
    a = ap.parse_args()
    px, vol = load_history_csv(a.history_dir) if a.history_dir else load_coinmetrics(a.coinmetrics_dir)
    print(f"panel: {px.shape[1]} assets, {px.index.min().date()} → {px.index.max().date()}", flush=True)
    cache, cfgs = {}, configs()
    for item in a.run:
        kind, cfg, ep, *opts = item.split(":")
        seed = next((int(o[5:]) for o in opts if o.startswith("seed=")), 0)
        shuf = "shuf" in opts
        if kind not in cache:
            cache[kind] = scenario(px, vol, kind)
        name = f"{kind}{'_shuf' if shuf else ''}_{cfg}_s{seed}"
        r = train_eval(name, cache[kind], *cfgs[cfg], epochs=int(ep), seed=seed, shuffle=shuf)
        r.update({"kind": kind, "cfg": cfg, "shuffle": shuf, "seed": seed, "epochs": int(ep)})
        with open(a.out, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        print(f"[{r['seconds']:5.0f}s] {name}: best@{r['best_epoch']} | "
              + " ".join(f"{p}_{s}_{t}_auc={r[f'{p}_{s}_{t}_auc']:.3f}" for p in ("best", "last")
                         for s in ("train", "test") for t in TARGETS), flush=True)


if __name__ == "__main__":
    main()
