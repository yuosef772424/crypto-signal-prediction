"""
PURPOSE:  Fingerprints of the splits `data.split_data` / `split_holdout` / `rolling_splits` produce on the synthetic pipeline datasets (tests/run_fixture.py), and the recorder of tests/golden_split_arrays.json: the oracle that every split refactor (label_end, SplitPlan, calibration carve-out) must reproduce.
TAGS:     split golden, split fingerprint, split_data, rolling_splits, split_holdout, byte identical splits, SplitPlan oracle, tests
PITFALLS: A split is a row SELECTION of the dataset, so it is digested by what decides the selection and is CPU independent: the key set (minus the additive stamps), every array's shape and dtype, the sha256 of the int64 timestamp column, and the sum of last_close with a relative tolerance. golden_split_arrays.json was recorded from the code of f112394 (before label_end / SplitPlan existed): never re-record it to make a test pass; re-record only with `python tests/split_golden.py --record` when a split rule changes ON PURPOSE (and say so in the commit). The additive keys `split_role` and `label_end` are excluded from the key set on purpose: they are the only things a split change may add.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import _nbload  # noqa: E402
from core.schema import LAST_COLUMN_INDEX, TS_COL  # noqa: E402
from run_fixture import _quiet, build_dataset  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_split_arrays.json")
#: keys a split change may ADD (additive stamps); they are not part of the pinned key set
ADDITIVE_KEYS = ("split_role", "label_end")
SPLIT_DATES = {"train_end": "2025-02-08", "val_end": "2025-02-22"}
HOLDOUT = "2025-02-25"


def _digest(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype)}


def split_fp(split):
    """Digest of one split or of an asset dict (test)."""
    if "y" not in split:
        return {k: split_fp(v) for k, v in sorted(split.items())}
    lc = np.asarray(split["last_candles"])
    out = {"keys": sorted(k for k in split if k not in ADDITIVE_KEYS)}
    out["arrays"] = {k: _digest(v) for k, v in sorted(split.items()) if isinstance(v, np.ndarray) and k not in ADDITIVE_KEYS}
    out["y"] = {k: _digest(v) for k, v in sorted(split["y"].items())}
    ts = np.ascontiguousarray(lc[:, TS_COL].astype("int64"))
    out["ts_sha256"] = hashlib.sha256(ts.tobytes()).hexdigest()
    out["n"] = int(len(lc))
    out["last_close_sum"] = float(np.sum(lc[:, LAST_COLUMN_INDEX["last_close"]].astype("float64")))
    return out


def triple_fp(triple):
    return {"train": split_fp(triple[0]), "val": split_fp(triple[1]), "test": split_fp(triple[2])}


def scenarios(ns):
    """name -> fingerprint, over the synthetic datasets; every scenario goes through the PUBLIC split functions."""
    out = {}
    cfg = lambda **o: dict(ns["CONFIG"], split_dates=dict(SPLIT_DATES), **o)  # noqa: E731
    ds1 = build_dataset(ns)
    for name, kw in (("g_sep", dict(keep_asset_test_separate=True)), ("g_merged", dict(keep_asset_test_separate=False))):
        out[f"global_time/1tf/{name}"] = triple_fp(_quiet(ns["split_data"], ds1, config=cfg(), **kw))
    out["global_time/1tf/auto_dates"] = triple_fp(_quiet(ns["split_data"], ds1, config=dict(ns["CONFIG"]), keep_asset_test_separate=False))
    out["global_time/1tf/leak_free_raw_quantile"] = triple_fp(_quiet(ns["build_leak_free_split"], ds1, config=dict(ns["CONFIG"]),
                                                                    keep_asset_test_separate=False))
    for mode_kw in (dict(keep_asset_test_separate=True), dict(keep_asset_test_separate=False)):
        out[f"per_asset/1tf/sep={mode_kw['keep_asset_test_separate']}"] = triple_fp(
            _quiet(ns["split_data"], ds1, mode="per_asset", config=dict(ns["CONFIG"]), **mode_kw))
    ds2 = build_dataset(ns, two_tf=True)
    out["global_time/2tf/g_sep"] = triple_fp(_quiet(ns["split_data"], ds2, config=cfg(), keep_asset_test_separate=True))
    dsh = build_dataset(ns, holdout=HOLDOUT)
    hcfg = cfg(holdout_start=HOLDOUT, open_holdout=True, min_split_samples=8)
    out["holdout/train_val_test"] = triple_fp(_quiet(ns["split_data"], dsh, config=hcfg, keep_asset_test_separate=False))
    out["holdout/sealed"] = split_fp(_quiet(ns["split_holdout"], dsh, config=hcfg, keep_asset_test_separate=False))
    wins = _quiet(ns["rolling_splits"], ds1, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=3,
                  keep_asset_test_separate=False, config=dict(ns["CONFIG"]), verbose=False)
    out["rolling/1tf"] = [triple_fp(w) for w in wins]
    return out


def close(a, b, path=""):
    """Differences (list of strings) between two fingerprint trees; floats at rtol 1e-9, everything else exact."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: only in {'new' if k in a else 'golden'}")
            else:
                diffs += close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if not np.isclose(a, b, rtol=1e-9, atol=0.0):
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if "--record" not in sys.argv:
        raise SystemExit("usage: python tests/split_golden.py --record   (re-records tests/golden_split_arrays.json; see PITFALLS)")
    fps = scenarios(_nbload.load_pipeline())
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(json.loads(json.dumps(fps)), f, indent=1, sort_keys=True)
    print("recorded", len(fps), "scenarios ->", GOLDEN_PATH)
