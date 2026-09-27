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
  <kind>_xrank : + per-date cross-sectional rank of every feature across coins (e.g. full_xrank, little_xrank)
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


def build_features(px, vol, bad_norm=False, xrank=False):
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
    if xrank:
        # رتبة مقطعية لكل ميزة بين العملات في نفس التاريخ (سببية: قيم نفس اليوم متاحة وقت t)، في [-0.5, 0.5] —
        # البديل الرخيص عن الانتباه بين الأصول (nyanp: ranks within the same time-id؛ sugghi: الفرق عن متوسط العملات)
        names = [c for c in next(iter(feats.values())).columns if c not in ("MKT_RET_1", "BTC_RET_1")]
        for c in names:
            panel = pd.DataFrame({a: feats[a][c] for a in feats})
            ranked = panel.rank(axis=1, pct=True) - 0.5
            for a in feats:
                feats[a][f"{c}_XR"] = ranked[a]
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
    parts = kind.split("_")
    feats, tg = build_features(px, vol, bad_norm="badnorm" in parts, xrank="xrank" in parts)
    X, y, days, aid, names = make_windows(feats, tg)
    S = split(X, y, days, aid)
    if "little" in parts:
        coins = np.random.default_rng(7).choice(np.unique(S["train"]["aid"]), 8, replace=False)
        S["train"] = subset(S["train"], np.isin(S["train"]["aid"], coins))
    S["feature_names"] = names
    S["n_coins"] = int(aid.max()) + 1
    return S


def linear_reference(S, shuffle=False, seed=0, C=0.1, tune=False, grid=(0.001, 0.01, 0.1, 1.0, 10.0)):
    """Logistic regression on [last step, window mean] — the same reference main.ipynb reports (§7-ز).

    tune=True: C يُختار لكل هدف على val (AUC) من grid — نفس امتياز النماذج العميقة التي تختار حقبتها على val؛
    test لا يُلمس في الاختيار. يُضاف {t}_C للنتيجة."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    f = lambda X: np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1).astype("float64")
    tr, va, te = S["train"], S["val"], S["test"]
    sc = StandardScaler().fit(np.nan_to_num(f(tr["X"])))
    Ftr, Fva, Fte = (sc.transform(np.nan_to_num(f(p["X"]))) for p in (tr, va, te))
    perm = np.random.default_rng(seed + 123).permutation(len(Ftr)) if shuffle else np.arange(len(Ftr))
    out = {}
    for t in TARGETS:
        ytr = tr["y"][t][perm] > 0
        if tune:
            fits = [(auc(va["y"][t], m.predict_proba(Fva)[:, 1]), c, m) for c in grid
                    for m in [LogisticRegression(C=c, max_iter=3000).fit(Ftr, ytr)]]
            _v, c_best, m = max(fits, key=lambda x: x[0])
            out[f"{t}_C"] = c_best
        else:
            m = LogisticRegression(C=C, max_iter=3000).fit(Ftr, ytr)
        out[f"{t}_train_auc"] = auc(ytr, m.predict_proba(Ftr)[:, 1])
        out[f"{t}_val_auc"] = auc(va["y"][t], m.predict_proba(Fva)[:, 1])
        out[f"{t}_test_auc"] = auc(te["y"][t], m.predict_proba(Fte)[:, 1])
    return out


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


def model_x(part, idx=None, coins=False):
    X = part["X"] if idx is None else part["X"][idx]
    if not coins:
        return X
    return {"input_sequence": X, "coin_id": (part["aid"] if idx is None else part["aid"][idx]).astype("int32")}


def evaluate(model, part, max_n=None, seed=0, probs_out=None):
    idx = np.arange(len(part["X"]))
    if max_n and len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(len(idx), max_n, replace=False))
    coins = isinstance(model.input, (list, tuple)) or (isinstance(model.inputs, list) and len(model.inputs) > 1)
    out = model.predict(model_x(part, idx, coins), batch_size=2048, verbose=0)
    res = {}
    for t in TARGETS:
        yt = part["y"][t][idx]
        p = np.asarray(out[f"y_{t}_class_logits"]).ravel()
        if probs_out is not None:
            probs_out[t] = p
        res[f"{t}_auc"] = auc(yt, p)
        res[f"{t}_acc"] = float(np.mean((p >= 0.5) == (yt > 0)))
        res[f"{t}_ic"], res[f"{t}_ic_t"] = daily_ic(yt, np.asarray(out[f"y_{t}"]).ravel(), part["days"][idx])
        res[f"{t}_p_std"] = float(np.std(p))
    return res


# ───────────────────────────── configs ─────────────────────────────
# الإعداد «robust» كما قيس في PR #7 قبل اختيار المُرمِّز — مُثبَّت هنا (لا يُقرأ من الدفتر) كي تبقى كل الأسماء
# التاريخية (robust*, RL*) تعني نفس النموذج بعد تغيّر الافتراضي في model_v2.
AM_V1 = dict(d_model=64, num_layers=2, head_hidden=64, class_head_hidden=32, dropout=0.25, input_clip=4.0,
             stats_mode="symlog", input_noise_std=0.1, feature_dropout=0.1, linear_path_l2=1e-3)


def configs():
    ns = project()
    robust_trainer = {"optimizer": {"weight_decay": 0.05},
                      "callbacks": {"early_stopping": {"weights_snapshot": "ema_weights"}},
                      "_class": {"label_smoothing": 0.1}}
    am = dict(AM_V1)
    cfgs = {"baseline": ({}, {}), "robust": (am, robust_trainer)}
    cfgs.update(EXTRA_CONFIGS(am, robust_trainer))
    # ما يستخدمه main.ipynb فعلاً مع ANTI_MEMORIZATION=True (يُقرأ من الدفتر: يتبع أي تغيير مستقبلي فيه)
    cfgs["am_default"] = (dict(ns["ANTI_MEMORIZATION_CONFIG"]),
                          {**robust_trainer, "optimizer": {"lr_initial": 3e-4, "weight_decay": 0.05,
                                                           "ema_warmup": True, "ema_window_epochs": 1.0}})
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
        **arch_variants(am, rt),
        "base_small": ({"d_model": 64, "num_layers": 2, "head_hidden": 64, "class_head_hidden": 32}, {}),
        "base_level": ({"level_passthrough": True}, {}),
    }


def arch_variants(am, rt):
    """§7 of the doc: architecture directions, all on top of the level fix (robust_level_bn_lr = "RL")."""
    RL = {**am, "level_passthrough": True, "level_norm": "batch"}
    rt_lr = {**rt, "optimizer": {**rt["optimizer"], "lr_initial": 3e-4}}
    tiny = {"d_model": 32, "num_layers": 1, "num_heads": 2, "num_kv_heads": 1, "head_hidden": 32, "class_head_hidden": 16}
    v = {
        "RL": {}, "RL_film": {"level_film": True}, "RL_coin": {"n_coins": -1},
        "RL_film_coin": {"level_film": True, "n_coins": -1},
        "RL_2tower": {"architecture": "two_tower"}, "RL_2tower_gate": {"architecture": "two_tower", "fusion": "gate"},
        "RL_tiny": tiny, "RL_tcn": {"encoder": "tcn"}, "RL_gru": {"encoder": "gru"},
        "RL_tiny_film": {**tiny, "level_film": True}, "RL_tcn_film": {"encoder": "tcn", "level_film": True},
        "RL_gru_film": {"encoder": "gru", "level_film": True},
        "RL_2tower_tiny": {**tiny, "architecture": "two_tower"},
        "RL_tcn_tiny": {**tiny, "encoder": "tcn"},
        "RL_tcn_tiny_film": {**tiny, "encoder": "tcn", "level_film": True},
    }
    out = {k: ({**RL, **x}, rt_lr) for k, x in v.items()}
    # drive_v2: اختيار الحقبة الأفضل على خسارة التصنيف وحدها، عزل تنعيم التسميات، GRU بطبقة واحدة
    gru = {**RL, "encoder": "gru"}
    sel = {**rt_lr, "callbacks": {**rt_lr.get("callbacks", {}),
                                  "early_stopping": {**rt_lr.get("callbacks", {}).get("early_stopping", {}),
                                                     "monitor": "val_class_loss", "mode": "min"}}}
    out.update({
        "RL_gru_selcls": (gru, sel),
        "RL_gru_nols": (gru, {**rt_lr, "_class": {}}),
        "RL_gru_1l": ({**gru, "num_layers": 1}, rt_lr),
        "RL_gru_1l_selcls": ({**gru, "num_layers": 1}, sel),
        "RL_tiny_selcls": ({**RL, **tiny}, sel),
    })
    # EMA: زخم 0.999 بلا إحماء يُبقي 0.999^N من المتوسط على أوزان أول الخطوات (N≈960 على بيانات Drive القليلة جداً)
    emaw = {**rt_lr, "optimizer": {**rt_lr["optimizer"], "ema_warmup": True}}
    emaw1 = {**rt_lr, "optimizer": {**rt_lr["optimizer"], "ema_warmup": True, "ema_window_epochs": 1.0}}
    raw = {**rt_lr, "callbacks": {**rt_lr.get("callbacks", {}),
                                  "early_stopping": {**rt_lr.get("callbacks", {}).get("early_stopping", {}),
                                                     "weights_snapshot": "raw"}}}
    out.update({
        "RL_gru_emaw": (gru, emaw),
        "RL_gru_raw": (gru, raw),
        "RL_tiny_raw": ({**RL, **tiny}, raw),
        "RL_tiny_emaw": ({**RL, **tiny}, emaw),
        "RL_gru_emaw1": (gru, emaw1),
        "RL_tiny_emaw1": ({**RL, **tiny}, emaw1),
    })
    return out


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
def train_eval(name, S, model_cfg, trainer_cfg, epochs, seed=0, shuffle=False, batch_size=256, run_root="/tmp/am_runs",
               pred_dir=None, eval_max=None):
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
    coins = bool(mcfg.get("n_coins"))
    if coins:
        mcfg["n_coins"] = int(S["n_coins"])
    # خلط أرقام العيّنات ثم جمعها (gather) بدل خلط البيانات نفسها: ~0.1 ثانية للحقبة بدل ~1 (نفس التوزيع، ترتيب آخر)
    Xtr = tf.nest.map_structure(tf.constant, model_x(tr, coins=coins))
    Ytr = {k: tf.constant(v) for k, v in ytr.items()}
    train_ds = (tf.data.Dataset.range(len(tr["X"])).shuffle(len(tr["X"]), seed=seed)
                .batch(batch_size, drop_remainder=True)
                .map(lambda i: (tf.nest.map_structure(lambda a: tf.gather(a, i), Xtr),
                                {k: tf.gather(v, i) for k, v in Ytr.items()}), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(2))
    # التحقّق بدفعات 2048: نفس الأوزان ونفس العيّنات، أقلّ بـ8 مرّات من الخطوات (57 ألف عيّنة val مقابل 4 آلاف
    # train على little كانت تجعل التحقّق ~14 ضعف التدريب ومقيّداً بالمعالج). يغيّر val_* تغيّراً ضئيلاً فقط.
    val_ds = tf.data.Dataset.from_tensor_slices((model_x(va, coins=coins), to_y(va))).batch(max(batch_size, 2048)).prefetch(2)
    builder = lambda: ns["build_model_fn"](tr["X"].shape[1], tr["X"].shape[2], config=mcfg)
    with contextlib.redirect_stdout(io.StringIO()):
        trainer, callbacks, ie = ns["build_training_system"](builder, cfg, next(iter(train_ds)))
    last, traj = {}, []
    every = max(epochs // 6, 1) if not eval_max else 10 ** 9   # الفحص السريع: بلا مسار وسيط

    class Probe(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            last["w"] = [w.copy() for w in trainer.model.get_weights()]
            if (epoch + 1) % every == 0 or epoch + 1 == epochs:
                r = {"epoch": epoch + 1}
                r.update({f"train_{k}": v for k, v in evaluate(trainer.model, tr_eval, 8000).items()})
                r.update({f"test_{k}": v for k, v in evaluate(trainer.model, te, 12000).items()})
                traj.append(r)

    tm = {"train": 0.0, "val": 0.0, "callbacks": 0.0}

    class TimeFirst(tf.keras.callbacks.Callback):   # أين يذهب الوقت: خطوات التدريب / التحقّق / callbacks نهاية الحقبة
        def on_epoch_begin(self, epoch, logs=None):
            tm["_t"] = time.time()

        def on_test_begin(self, logs=None):
            tm["train"] += time.time() - tm["_t"]
            tm["_t"] = time.time()

        def on_test_end(self, logs=None):
            tm["val"] += time.time() - tm["_t"]

        def on_epoch_end(self, epoch, logs=None):
            tm["_t"] = time.time()

    class TimeLast(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            tm["callbacks"] += time.time() - tm["_t"]

    cbs = [c for c in callbacks if type(c).__name__ != "EpochCheckpointCallback"]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        hist = trainer.fit(train_ds, validation_data=val_ds, initial_epoch=ie, epochs=epochs,
                           callbacks=[TimeFirst(), Probe()] + cbs + [TimeLast()], verbose=0)
    t_fit = time.time() - t0
    best = next(c for c in cbs if type(c).__name__ == "BestModelTracker")
    res = {"name": name, "seconds": time.time() - t0, "epochs_run": len(hist.history.get("loss", [])),
           "best_epoch": best.best_epoch, "monitor": cfg["callbacks"]["early_stopping"]["monitor"],
           "monitor_logged": cfg["callbacks"]["early_stopping"]["monitor"] in hist.history, "n_params": int(trainer.model.count_params()), "n_train": len(tr["X"])}
    for tag in ("best", "last"):
        if tag == "last":
            trainer.model.set_weights(last["w"])
        for pn, part in (("train", tr_eval), ("val", va), ("test", te)):
            probs = {} if (pred_dir and not eval_max and tag == "best" and pn == "test") else None
            n_eval = 20000 if pn == "train" else eval_max   # eval_max: عيّنة ثابتة (نفس البذرة) لكل التشغيلات
            for k, v in evaluate(trainer.model, part, n_eval, probs_out=probs).items():
                res[f"{tag}_{pn}_{k}"] = v
            if probs:   # توقّعات test لأوزان «الأفضل» — لقياس متوسط البذور (ensemble) بلا تدريب إضافي
                os.makedirs(pred_dir, exist_ok=True)
                np.savez_compressed(os.path.join(pred_dir, f"{name}.npz"),
                                    **{t: v.astype("float16") for t, v in probs.items()})
    res["trajectory"] = traj
    res["time"] = {k: round(v, 1) for k, v in tm.items() if not k.startswith("_")}
    res["time"]["final_eval"] = round(time.time() - t0 - t_fit, 1)
    shutil.rmtree(run_dir, ignore_errors=True)
    return res


PLANS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "am_plans.json")


def summarize(results, refs, header=""):
    """ملخّص مضغوط للنسخ واللصق: سطر لكل تشغيل + المرجع الخطّي لكل سيناريو."""
    lines = [f"### AM-SUMMARY {header}".rstrip(),
             "| run | best@ | test mag best/last | test dir best/last | train mag best/last | train dir best/last | sec |",
             "|---|---|---|---|---|---|---|"]
    f = lambda r, a, b: f"{r[a]:.3f}/{r[b]:.3f}"
    for r in results:
        lines.append(f"| {r['name']} | {r['best_epoch']} | {f(r, 'best_test_mag_auc', 'last_test_mag_auc')} | "
                     f"{f(r, 'best_test_dir_auc', 'last_test_dir_auc')} | {f(r, 'best_train_mag_auc', 'last_train_mag_auc')} | "
                     f"{f(r, 'best_train_dir_auc', 'last_train_dir_auc')} | {r['seconds']:.0f} |")
    for name, ref in refs.items():
        lines.append(f"| {name} (linear) | - | {ref['mag_test_auc']:.3f} | {ref['dir_test_auc']:.3f} | "
                     f"{ref['mag_train_auc']:.3f} | {ref['dir_train_auc']:.3f} | - |")
    return "\n".join(lines)


HP_OPTS = {"lr": "optimizer.lr_initial", "wd": "optimizer.weight_decay", "do": "model.dropout",
           "pat": "early_stopping.patience", "mon": "early_stopping.monitor (cls = val_class_loss)",
           "bs": "training batch size (default 256; لا يُضبط معه lr تلقائياً — الشبكة lr تغطّيه)",
           "dm": "model.d_model (عرض النموذج — head_hidden يتبعه، class_head_hidden نصفه)", "nl": "model.num_layers"}


def parse_item(item):
    """kind:cfg:epochs[:shuf][:seed=N][:lr=..][:wd=..][:do=..][:pat=..][:mon=cls]"""
    kind, cfg, ep, *opts = item.split(":")
    p = {"item": item, "kind": kind, "cfg": cfg, "epochs": int(ep), "shuf": "shuf" in opts, "seed": 0, "hp": {}}
    for o in opts:
        if o.startswith("seed="):
            p["seed"] = int(o[5:])
        elif "=" in o:
            key, val = o.split("=", 1)
            if key not in HP_OPTS:
                raise SystemExit(f"خيار غير معروف {o!r} في {item!r} — المتاح: {sorted(HP_OPTS)}")
            p["hp"][key] = val
        elif o != "shuf":
            raise SystemExit(f"خيار غير معروف {o!r} في {item!r}")
    return p


def item_name(p):
    hp = "".join(f"_{k}{v}" for k, v in sorted(p["hp"].items()))   # بلا خيارات = نفس الأسماء التاريخية
    return f"{p['kind']}{'_shuf' if p['shuf'] else ''}_{p['cfg']}{hp}_s{p['seed']}"


def apply_hp(model_cfg, trainer_cfg, hp):
    import copy
    m, t = copy.deepcopy(model_cfg or {}), copy.deepcopy(trainer_cfg or {})
    es = lambda: t.setdefault("callbacks", {}).setdefault("early_stopping", {})
    if "lr" in hp:
        t.setdefault("optimizer", {})["lr_initial"] = float(hp["lr"])
    if "wd" in hp:
        t.setdefault("optimizer", {})["weight_decay"] = float(hp["wd"])
    if "do" in hp:
        m["dropout"] = float(hp["do"])
    if "dm" in hp:
        m.update(d_model=int(hp["dm"]), head_hidden=int(hp["dm"]), class_head_hidden=max(int(hp["dm"]) // 2, 8))
    if "nl" in hp:
        m["num_layers"] = int(hp["nl"])
    if "pat" in hp:
        es()["patience"] = int(hp["pat"])
    if hp.get("mon") == "cls":
        es().update({"monitor": "val_class_loss", "mode": "min"})
    return m, t


def with_opts(item, seed=None, shuf=False):
    parts = [o for o in item.split(":") if not (seed is not None and o.startswith("seed="))]
    return ":".join(parts + ([f"seed={seed}"] if seed is not None else []) + (["shuf"] if shuf else []))


def _run_cost(item):
    """تقدير نسبي لزمن تشغيل (للتوزيع على العمّال): الحقب × حجم السيناريو."""
    p = parse_item(item)
    return 0 if p["cfg"] == "linear" else p["epochs"] * (4 if p["kind"].split("_")[0] == "full" else 1)


def split_items(items, n):
    """توزيع LPT: الأطول أولاً على العامل الأقلّ حِملاً. داخل كل عامل: السيناريوهات بترتيب أول ظهور لها في الخطة
    (الأولوية محفوظة) وتشغيلات كل سيناريو متتالية (يُحرَّر كاش السيناريو بعد آخر استخدام)."""
    load, parts = [0] * n, [[] for _ in range(n)]
    for i, it in sorted(enumerate(items), key=lambda x: -_run_cost(x[1])):
        k = load.index(min(load))
        parts[k].append((i, it))
        load[k] += _run_cost(it) or 1
    first = {}
    for i, it in enumerate(items):
        first.setdefault(it.split(":")[0], i)
    return [[it for _i, it in sorted(p, key=lambda x: (first[x[1].split(":")[0]], x[0]))] for p in parts if p]


def pred_dir_of(a):
    return a.pred_dir or f"{a.out}.preds"


def ensemble_auc(pred_dir, kind, names):
    """AUC لمتوسط احتمالات test عبر عدّة تشغيلات (بذور) — None إن نقص ملف."""
    try:
        y = np.load(os.path.join(pred_dir, f"_labels_{kind}.npz"))
        ps = [np.load(os.path.join(pred_dir, f"{n}.npz")) for n in names]
    except (OSError, FileNotFoundError):
        return None
    return {t: auc(y[t], np.mean([p[t].astype("float32") for p in ps], axis=0)) for t in TARGETS}


def load_panel(a):
    px, vol = load_history_csv(a.history_dir) if a.history_dir else load_coinmetrics(a.coinmetrics_dir)
    if a.max_assets and px.shape[1] > a.max_assets:
        keep = px.notna().sum().sort_values(ascending=False).index[:a.max_assets]
        px, vol = px[keep], vol[keep]
    env = f"{px.shape[1]} assets, {px.index.min().date()} → {px.index.max().date()}"
    try:
        import tensorflow as tf
        env += f", GPU={len(tf.config.list_physical_devices('GPU'))}"
    except Exception:  # noqa: BLE001
        pass
    return px, vol, env


def run_serial(a, items):
    px, vol, env = load_panel(a)
    print(f"panel: {env} | runs: {len(items)}", flush=True)
    cache, cfgs, done, refs = {}, configs(), [], {}
    pred_dir = pred_dir_of(a)
    last_use = {it.split(":")[0]: i for i, it in enumerate(items)}
    for i_item, item in enumerate(items):
        p = parse_item(item)
        kind = p["kind"]
        if kind not in cache:
            cache[kind] = scenario(px, vol, kind)
            print(f"scenario {kind}: train={len(cache[kind]['train']['X'])} val={len(cache[kind]['val']['X'])} "
                  f"test={len(cache[kind]['test']['X'])}", flush=True)
            os.makedirs(pred_dir, exist_ok=True)
            np.savez_compressed(os.path.join(pred_dir, f"_labels_{kind}.npz"),
                                **{t: cache[kind]["test"]["y"][t].astype("float32") for t in TARGETS})
            for tag, kw in (("", {}), ("_shuf", {"shuffle": True}), ("_tunedC", {"tune": True})):
                lr_ref = linear_reference(cache[kind], **kw)
                refs[f"{kind}{tag}"] = lr_ref
                print(f"linear_ref {kind}{tag}: " + " ".join(f"{k}={v:.3f}" for k, v in lr_ref.items()), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps({"name": f"{kind}{tag}_linear_ref", "kind": kind, "shuffle": tag == "_shuf",
                                        "linear_ref": lr_ref}) + "\n")
        if p["cfg"] != "linear":   # cfg=linear: المرجع الخطّي وحده لهذا السيناريو
            name = item_name(p)
            r = train_eval(name, cache[kind], *apply_hp(*cfgs[p["cfg"]], p["hp"]), epochs=p["epochs"], seed=p["seed"],
                           shuffle=p["shuf"], batch_size=int(p["hp"].get("bs", 256)),
                           pred_dir=None if p["shuf"] else pred_dir, eval_max=a.eval_max)
            r.update({"item": item, "kind": kind, "cfg": p["cfg"], "hp": p["hp"], "shuffle": p["shuf"],
                      "seed": p["seed"], "epochs": p["epochs"]})
            done.append(r)
            with open(a.out, "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
            tt = r.get("time", {})
            print(f"[{r['seconds']:5.0f}s] {name}: best@{r['best_epoch']}/{r['epochs_run']} | "
                  f"time train/val/cb/eval={tt.get('train', 0):.0f}/{tt.get('val', 0):.0f}/{tt.get('callbacks', 0):.0f}/"
                  f"{tt.get('final_eval', 0):.0f}s | "
                  + " ".join(f"{q}_{s_}_{t}_auc={r[f'{q}_{s_}_{t}_auc']:.3f}" for q in ("best", "last")
                             for s_ in ("train", "test") for t in TARGETS), flush=True)
        if last_use[kind] == i_item:
            cache.pop(kind, None)   # ذاكرة: لا يُحتاج هذا السيناريو بعد الآن
    import resource
    print(f"peak_ram_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}", flush=True)
    return done, refs, {"panel": env}


def run_parallel(a, items, n):
    """يشغّل n عمليات مستقلّة على نفس GPU (كلٌّ يحجز ما يحتاجه فقط: TF_FORCE_GPU_ALLOW_GROWTH)، يبثّ مخرجاتها
    ببادئة [wK]، ثم يدمج نتائجها في --out."""
    import subprocess
    import threading
    parts = split_items(items, n)
    procs, outs, info, peaks = [], [], {}, []
    threads_each = str(max(1, (os.cpu_count() or 2) // len(parts)))   # حصّة عادلة من الأنوية لكل عامل
    env = {**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true", "PYTHONUNBUFFERED": "1",
           "TF_NUM_INTRAOP_THREADS": threads_each, "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": threads_each}
    data = ["--history-dir", a.history_dir] if a.history_dir else ["--coinmetrics-dir", a.coinmetrics_dir]
    if a.max_assets:
        data += ["--max-assets", str(a.max_assets)]
    print(f"⚡ تشغيل متوازٍ: {len(parts)} عمّال لـ{len(items)} تشغيلاً على نفس GPU", flush=True)
    for k, part in enumerate(parts, 1):
        out = f"{a.out}.w{k}"
        if os.path.exists(out):
            os.remove(out)
        outs.append(out)
        print(f"   [w{k}] {' '.join(part)}", flush=True)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), *data, "--out", out,
                                       "--pred-dir", pred_dir_of(a),
                                       *(["--eval-max", str(a.eval_max)] if a.eval_max else []), "--run", *part],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env))

    def pump(k, pr):
        for line in pr.stdout:
            if line.startswith("panel:"):
                info.setdefault("panel", line.split("|")[0][6:].strip())
            if line.startswith("peak_ram_mb="):
                peaks.append(float(line.split("=")[1]))
            if line.startswith(("panel:", "scenario", "linear_ref", "[", "Traceback", "peak_ram")) or "Error" in line:
                print(f"[w{k}] {line}", end="", flush=True)

    threads = [threading.Thread(target=pump, args=(k, pr), daemon=True) for k, pr in enumerate(procs, 1)]
    for th in threads:
        th.start()
    codes = [pr.wait() for pr in procs]
    for th in threads:
        th.join()
    done, refs = {}, {}
    with open(a.out, "a") as fo:
        for out in outs:
            if not os.path.exists(out):
                continue
            for line in open(out):
                fo.write(line)
                r = json.loads(line)
                if "linear_ref" in r:
                    refs[r["name"][:-len("_linear_ref")]] = r["linear_ref"]
                else:
                    done[r["name"]] = r
            os.remove(out)
    order = {item_name(parse_item(it)): i for i, it in enumerate(items) if parse_item(it)["cfg"] != "linear"}
    bad = [f"w{k}={c}" for k, c in enumerate(codes, 1) if c != 0]
    if bad:
        oom = " (‎-9 = قتله النظام، غالباً نفاد الذاكرة RAM — قلّل --parallel)" if -9 in codes else ""
        print(f"⚠️ عمّال انتهوا بخطأ: {', '.join(bad)}{oom} — النتائج المكتملة محفوظة في {a.out}", flush=True)
    missing = [nm for nm in order if nm not in done]
    if missing:
        print(f"⚠️ لم تكتمل: {missing}", flush=True)
    if peaks:
        info["ram"] = f"RAM peak {sum(peaks) / 1024:.1f}GB total, {max(peaks) / 1024:.1f}GB/worker"
    info["parallel"] = len(parts)
    return sorted(done.values(), key=lambda r: order.get(r["name"], 1e9)), dict(sorted(refs.items())), info


RAM_PER_WORKER_GB = 4.5


def _mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def execute(a, items, parallel, forced=False):
    """parallel مُقيَّد بنصف أنوية المعالج: drive_v2 على Colab العادي (نواتان) بأربعة عمّال أعطى نفس الإنتاجية
    تقريباً كالتسلسلي (~900 ثانية للتجربة بدل ~210) — العنق هو المعالج (خطوات TF الصغيرة) لا ذاكرة GPU."""
    want = max(1, min(int(parallel or 1), len(items)))
    cap_cpu = max(1, (os.cpu_count() or 2) // 2)
    avail = _mem_available_gb()
    # drive_v2 على Drive: ~3.4GB ذروة للعامل (full_xrank أكثر)؛ 4 عمّال على Colab (12.7GB) ⇒ قُتل اثنان (-9)
    cap_ram = max(1, int(avail // RAM_PER_WORKER_GB)) if avail else want
    # --parallel صريح من سطر الأوامر يتجاوز حدّ المعالج (للقياس) لكن لا يتجاوز حدّ الذاكرة (نفادها يقتل العمّال)
    cap = cap_ram if forced else min(cap_cpu, cap_ram)
    if want > cap:
        print(f"ℹ️ parallel={want} خُفِّض إلى {cap}: {os.cpu_count()} أنوية معالج، ~{avail or 0:.1f}GB ذاكرة متاحة "
              f"(~{RAM_PER_WORKER_GB}GB لكل عامل) — عمّال أكثر يتزاحمون على المعالج أو تنفد الذاكرة", flush=True)
        want = cap
    return run_parallel(a, items, want) if want > 1 else run_serial(a, items)


# ───────────────────────────── fair tuning ─────────────────────────────
def val_score(r):
    """معيار الاختيار: متوسط AUC على val للرأسين (أوزان «الأفضل») — test لا يدخل في أي اختيار."""
    return float(np.mean([r[f"best_val_{t}_auc"] for t in TARGETS]))


def select_winners(results):
    """لكل (سيناريو، نوع): التشغيل الأعلى val_score بين إعدادات جولة الضبط (بذرة 0، تسميات حقيقية)."""
    best = {}
    for r in results:
        if r["shuffle"] or r["seed"] != 0:
            continue
        key = (r["kind"], r["cfg"])
        if key not in best or val_score(r) > val_score(best[key]):
            best[key] = r
    return best


def fair_table(results, refs, winners, seeds, pred_dir=None):
    """جدول المقارنة العادلة: لكل نوع أفضل إعداد له (على val)، ثم test بمتوسط ± انحراف عبر البذور."""
    lines = ["### FAIR-SUMMARY (كل نوع بأفضل إعداداته على val؛ test = متوسط ± انحراف عبر "
             f"{len(seeds)} بذور؛ الحفظ = AUC تدريب «الأفضل» على تسميات مخلوطة)",
             "| kind | type | best hp (val) | val | test mag | test dir | ensemble mag/dir | train-test gap mag/dir | "
             "shuffled train mag/dir | epochs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda xs: f"{np.mean(xs):.3f}±{np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}"
    for (kind, cfg), w in sorted(winners.items(), key=lambda x: (x[0][0], -val_score(x[1]))):
        same = lambda r: r["kind"] == kind and r["cfg"] == cfg and r["hp"] == w["hp"]
        runs = [r for r in results if same(r) and not r["shuffle"] and r["seed"] in seeds]
        sh = [r for r in results if same(r) and r["shuffle"]]
        hp = " ".join(f"{k}={v}" for k, v in sorted(w["hp"].items())) or "-"
        gap = lambda t: [r[f"best_train_{t}_auc"] - r[f"best_test_{t}_auc"] for r in runs]
        shs = f"{sh[0]['best_train_mag_auc']:.3f}/{sh[0]['best_train_dir_auc']:.3f}" if sh else "-"
        ens = ensemble_auc(pred_dir, kind, [r["name"] for r in runs]) if pred_dir and len(runs) > 1 else None
        ens_s = f"{ens['mag']:.3f}/{ens['dir']:.3f}" if ens else "-"
        lines.append(f"| {kind} | {cfg} | {hp} | {val_score(w):.3f} | {ms([r['best_test_mag_auc'] for r in runs])} | "
                     f"{ms([r['best_test_dir_auc'] for r in runs])} | {ens_s} | {np.mean(gap('mag')):.3f}/{np.mean(gap('dir')):.3f} | "
                     f"{shs} | {'/'.join(str(r['best_epoch']) for r in runs)} |")
    for name, ref in refs.items():
        if name.endswith("_tunedC"):
            c = "/".join(f"{ref.get(f'{t}_C', '-')}" for t in TARGETS)
            lines.append(f"| {name[:-7]} | linear (C={c} on val) | - | "
                         f"{np.mean([ref[f'{t}_val_auc'] for t in TARGETS]):.3f} | {ref['mag_test_auc']:.3f} | "
                         f"{ref['dir_test_auc']:.3f} | - | {ref['mag_train_auc'] - ref['mag_test_auc']:.3f}/"
                         f"{ref['dir_train_auc'] - ref['dir_test_auc']:.3f} | - | - |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--history-dir")
    g.add_argument("--coinmetrics-dir")
    runs = ap.add_mutually_exclusive_group(required=True)
    runs.add_argument("--run", nargs="+", help="kind:cfg:epochs[:shuf][:seed=N][:lr=..][:wd=..][:do=..][:pat=..][:mon=cls]")
    runs.add_argument("--plan", help=f"اسم خطة من {os.path.basename(PLANS_FILE)}")
    ap.add_argument("--out", default="anti_memorization_results.jsonl")
    ap.add_argument("--max-assets", type=int, default=None,
                    help="أطول N عملات تاريخاً فقط (للسرعة مع مئات العملات)")
    ap.add_argument("--summary", action="store_true", help="اطبع في النهاية ملخّصاً مضغوطاً للنسخ")
    ap.add_argument("--pred-dir", default=None, help="مجلد توقّعات test لكل تشغيل (افتراضياً: <out>.preds)")
    ap.add_argument("--eval-max", type=int, default=None,
                    help="فحص سريع: تقييم val/test على عيّنة ثابتة بهذا الحجم وبلا مسار وسيط (أو حقل eval_max في الخطة)")
    ap.add_argument("--parallel", type=int, default=None,
                    help="عدد التشغيلات المتزامنة على نفس GPU (افتراضياً: حقل parallel في الخطة، وإلا 1)")
    a = ap.parse_args()
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # قبل أي استيراد لـ TF: لا يحجز كل ذاكرة GPU
    items, parallel, tune = a.run, a.parallel, None
    if a.plan:
        plans = json.load(open(PLANS_FILE, encoding="utf-8"))
        if a.plan not in plans:
            raise SystemExit(f"خطة غير موجودة: {a.plan!r} — المتاح: {sorted(k for k in plans if not k.startswith('_'))}")
        plan = plans[a.plan]
        items = plan["runs"] if isinstance(plan, dict) else plan
        if isinstance(plan, dict):
            parallel = plan.get("parallel") if parallel is None else parallel
            tune = plan.get("tune")
            a.eval_max = a.eval_max or plan.get("eval_max")
    for it in items:
        parse_item(it)   # خطأ صريح قبل أي تدريب
    t0 = time.time()
    forced = a.parallel is not None
    results, refs, info = execute(a, items, parallel, forced)
    winners, seeds = {}, [0]
    if tune:
        # الجولة ٢: بذور إضافية + تسميات مخلوطة لفائز كل نوع فقط — الفائز اختير على val وحده
        winners = select_winners(results)
        seeds = [0] + list(tune.get("extra_seeds", [1, 2]))
        round2 = [with_opts(w["item"], seed=s) for w in winners.values() for s in seeds[1:]]
        if tune.get("shuffle_winner", True):
            round2 += [with_opts(w["item"], shuf=True) for w in winners.values()]
        print(f"🏁 جولة الضبط انتهت — الفائزون على val: "
              + ", ".join(f"{k[0]}:{k[1]}→{w['hp'] or '-'}" for k, w in winners.items())
              + f" | الجولة ٢: {len(round2)} تشغيلاً", flush=True)
        r2, refs2, _ = execute(a, round2, parallel, forced)
        results += r2
        refs.update(refs2)
    if a.summary:
        hdr = " | ".join(x for x in (f"plan={a.plan or 'custom'}", info.get("panel", ""),
                                      f"parallel={info['parallel']}" if "parallel" in info else "", info.get("ram", ""),
                                      f"{time.time() - t0:.0f}s") if x)
        print("\n" + summarize(results, refs, header=hdr))
        if tune:
            print("\n" + fair_table(results, refs, winners, seeds, pred_dir=pred_dir_of(a)))


if __name__ == "__main__":
    main()
