"""
PURPOSE:  Synthetic pipeline datasets (single 1h timeframe, and 1h + 4h) built by the real data pipeline from seeded OHLCV, plus
          fingerprints of splits/configs; the shared fixture of the main-wiring equivalence tests (RunSettings / workflow.run).
TAGS:     run fixture, synthetic dataset, golden, fingerprint, RunSettings, main wiring, tests
PITFALLS: Fingerprints are (shape, dtype, sum, sum of squares) compared with a relative tolerance, never byte hashes: the pipeline's
          float64 exp/log differ in the last bits between CPUs (PHILOSOPHY 2.1), while any change of logic moves a sum by far more.
          The recorded values in tests/golden_run_settings.json were taken from the code before RunSettings existed (main.ipynb
          cells executed in order, commit 1b1b14f).
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ohlcv(days, seed, start):
    """Seeded 1h candles (the generator of tests/test_multi_tf.py)."""
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
                         "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003), "close": close,
                         "volume": rng.random(n) * 1000 + 50}, index=idx)


def build_dataset(ns, two_tf=False, coins=6, days=70, holdout=None):
    """A real pipeline dataset on synthetic candles (~1,200 windows, 6 coins, split dates inside the data range). ``ns`` is a pipeline
    namespace (``_nbload.load_pipeline()``); its CONFIG is reset to the pipeline defaults afterwards. ``holdout``: the sealed
    holdout_start the dataset carries (None = a dataset file without the key, like the old files)."""
    ns["reset_config"]()
    over = ns["HOURLY_4H_OVERRIDES"] if two_tf else ns["HOURLY_W32_S8_OVERRIDES"]
    _quiet(ns["apply_hourly_preset"], over)
    tf_order = ["1h", "4h"] if two_tf else ["1h"]
    windows = {"1h": 16, "4h": 8} if two_tf else {"1h": 16}
    ns["update_config"]({"tf_order": tf_order, "window_sizes": windows,
                         "phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
                         "funding_rate": {"enabled": False}, "open_interest": {"enabled": False}})
    _quiet(ns["refresh_features"])
    starts = ["2025-01-01", "2025-01-01 03:00", "2025-01-01 05:00", "2025-01-01 11:00", "2025-01-02", "2025-01-01"]

    def loader(fid, name):
        i = int(name[1])
        return ohlcv(days, 40 + i, starts[i])

    ds = _quiet(ns["build_dataset"], [{"name": f"C{i}USDT"} for i in range(coins)], load_asset_fn=loader,
                resample_fn=ns["make_resample_fn"](ns["CONFIG"]), max_workers=1, config=ns["CONFIG"])
    ns["reset_config"]()
    ds["split_dates"] = {"train_end": "2025-02-08", "val_end": "2025-02-22"}
    if holdout:
        ds["holdout_start"] = holdout
    else:
        ds.pop("holdout_start", None)
    return ds


def run_namespace():
    """A fresh shared namespace like main.ipynb's after its %run cells and the workflow load: pipeline, model, trainer, chicks
    evaluation (self-tests / smoke tests excluded) and all workflow modules."""
    ns = _nbload.load_pipeline()
    _nbload.load_model(ns)
    _nbload.load_trainer(ns)
    _nbload.load_evaluation(ns)
    _quiet(_nbload.workflow_package().load_into, ns)
    return ns


def arr_fp(a):
    a = np.asarray(a)
    f = a.astype("float64") if a.dtype.kind in "fiub" else None
    return {"shape": list(a.shape), "dtype": str(a.dtype),
            "sum": None if f is None else float(np.nansum(f)), "sq": None if f is None else float(np.nansum(f * f))}


def split_fp(split):
    """Fingerprint of one split (train/val) or an asset dict (test): array digests of X/y/last_candles/base_params + stamps."""
    if "y" not in split:                                            # test: {asset: split}
        return {k: split_fp(v) for k, v in sorted(split.items())}
    out = {"y": {k: arr_fp(v) for k, v in sorted(split["y"].items())},
           "last_candles": arr_fp(split["last_candles"]), "base_params": arr_fp(split["base_params"]),
           "stamps": {k: split.get(k) for k in ("target_mode", "reg_target_scale", "reg_target_scales", "entry_close_reg")}}
    out["X"] = {k: arr_fp(v) for k, v in sorted(split.items()) if k.startswith("X_")}
    return out


def close(a, b, path="", rtol=1e-6):
    """Recursive comparison of two fingerprint trees: numbers with a relative tolerance, everything else exactly. Returns a list of
    human-readable differences (empty = equal)."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: only in {'first' if k in a else 'second'}")
            else:
                diffs += close(a[k], b[k], f"{path}/{k}", rtol)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += close(x, y, f"{path}[{i}]", rtol)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or not np.isclose(a, b, rtol=rtol, atol=1e-9):
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def jsonable(x):
    """Plain-JSON image of a config/spec tree: tuples -> lists, numpy scalars -> python, dataclasses -> dicts, anything else -> repr."""
    import dataclasses
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


def model_fp(model):
    return {"layers": [[l.name, type(l).__name__, [list(w.shape) for w in l.weights]] for l in model.layers],
            "inputs": [i.name for i in model.inputs],
            "outputs": list(model.output.keys()) if isinstance(model.output, dict) else None,
            "params": int(model.count_params())}


def capture_state(*, config, model_tf, model_tfs, reg_target_scale, train, val, test, price_targets, suspended_targets,
                  model_overrides, model_seq_len, model_n_features, model, main_config, val_batch, chicks_targets,
                  chicks_market_neutral, test_dict, eval_target_specs, history=None):
    """The fingerprint of a main run, from named parts (the same dict whether the parts came from main.ipynb's old cells or from the
    workflow.run step functions). Every value is JSON-able; see the module PITFALLS for how arrays are digested."""
    keys = ("tf_order", "base_tf", "window_sizes", "forecast_horizon", "stride", "reg_target_scale", "price_norm_mode",
            "price_pct_clip", "split_dates", "holdout_start", "project_name")
    xb, yb = val_batch
    out = {"config": {k: jsonable(config.get(k)) for k in keys}, "model_tf": model_tf, "model_tfs": list(model_tfs),
           "reg_target_scale": float(reg_target_scale),
           "splits": {"train": split_fp(train), "val": split_fp(val), "test": split_fp(test)},
           "price_targets": list(price_targets), "suspended_targets": list(suspended_targets),
           "model_overrides": jsonable(model_overrides), "model_seq_len": jsonable(model_seq_len),
           "model_n_features": jsonable(model_n_features), "model": model_fp(model), "main_config": jsonable(main_config),
           "val_batch": {"x": ({k: arr_fp(v) for k, v in sorted(xb.items())} if isinstance(xb, dict) else arr_fp(xb)),
                         "y": {k: arr_fp(v) for k, v in sorted(yb.items())}},
           "chicks_targets": None if chicks_targets is None else list(chicks_targets),
           "chicks_market_neutral": bool(chicks_market_neutral),
           "test_dict": None if test_dict is None else {
               a: {"X": {k: arr_fp(v) for k, v in sorted(d.items()) if k.startswith("X_")},
                   "base_params": arr_fp(d["base_params"]), "last_candles": arr_fp(d["last_candles"]),
                   "y": {k: arr_fp(v) for k, v in sorted(d["y"].items())}}
               for a, d in sorted(test_dict.items())},
           "eval_target_specs": jsonable(eval_target_specs)}
    if history is not None:
        out["history"] = {k: [float(v) for v in vs] for k, vs in sorted(history.items())}
    return out
