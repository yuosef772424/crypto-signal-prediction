"""إصلاحات تدقيق الجولة ٢ (docs/research/audit/r2_01..07) — كل اختبار يفشل على الكود قبل الإصلاح.

تُنفَّذ خلايا الدفاتر نفسها (لا نسخاً منها) ببيانات تركيبية صغيرة بلا Drive:
    python -m unittest tests.test_audit_round2 -v
(r2_06، مقياس القسم غير المختوم في cross_asset، في tests/test_cross_asset.py::AuditRound2Tests.)
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402

_NS = None


def _ns():
    """نطاق خط الأنابيب (بلا خلية الاختبارات الذاتية) — مرّة واحدة لكل الاختبارات؛ كل اختبار يعمل على نسخة من CONFIG."""
    global _NS
    if _NS is None:
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            _NS = _nbload.load_pipeline()
        finally:
            os.chdir(cwd)
    return _NS


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _cfg(**over):
    """إعداد 1h صغير: نافذة 32، stride 8، أفق 1، بلا ميزات عابرة للأصول ولا مرحلة ٢ (سريع وحتمي)."""
    ns = _ns()
    cfg = deepcopy(ns["CONFIG"])
    cfg.update(tf_order=["1h"], base_tf="1h", model_tf="1h", window_sizes={"1h": 32}, stride=8, forecast_horizon=1,
               feature_order=None, split_dates=None, holdout_start=None, open_holdout=False, excluded_coins=[],
               min_split_samples=8, split_mode="global_time", keep_asset_test_separate=True, embargo_candles=None,
               market_context={"enabled": False}, funding_rate={"enabled": False}, open_interest={"enabled": False},
               momentum_orth_natr={"enabled": False}, market_breadth={"enabled": False},
               phase2_data={"use_intraday_15m": False, "use_futures_metrics": False})
    cfg.update(over)
    return cfg


def _ohlcv(start, end, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
                         "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003),
                         "close": close, "volume": rng.random(n) * 1000 + 50}, index=idx)


def _build(cfg, frames, checkpoint_dir=None):
    ns = _ns()
    resample = lambda df, tfs: ns["resample_timeframes"](df, tfs, config=cfg)  # noqa: E731
    return _quiet(ns["build_dataset_from_loader"], [{"name": n} for n in frames], lambda f, n: frames[n].copy(),
                  resample, config=cfg, max_workers=1, checkpoint_dir=checkpoint_dir)


def _grid_dataset(names, start, end, freq="8h"):
    """مجموعة بيانات هيكلية على شبكة 8h (طوابع فقط تهمّ التقسيم)."""
    grid = pd.date_range(start, end, freq=freq, tz="UTC")
    lcs, bounds, off = [], [], 0
    for nm in names:
        lc = np.full((len(grid), 7), 100.0)
        lc[:, 3] = grid.as_unit("ns").asi8
        lcs.append(lc)
        bounds.append({"name": nm, "start": off, "end": off + len(grid)})
        off += len(grid)
    lc = np.concatenate(lcs)
    n = len(lc)
    return {"base_params": np.zeros((n, 2), "float32"), "last_candles": lc, "X_1h": np.zeros((n, 4, 2), "float32"),
            "y_high_class": np.zeros(n, "float32"), "y_high_reg": np.zeros(n, "float32"), "timeframes": ["1h"],
            "targets": ["high_class", "high_reg"], "asset_bounds": bounds, "base_tf": "1h",
            "window_sizes": {"1h": 32}, "forecast_horizon": 1, "stride": 8}


def _ts(split_or_dict):
    parts = [split_or_dict] if "last_candles" in split_or_dict else list(split_or_dict.values())
    return pd.to_datetime(np.concatenate([p["last_candles"][:, 3] for p in parts]).astype("int64"), utc=True)


class StaleCheckpointTests(unittest.TestCase):
    """r2_01: بصمة نقاط الاستئناف تشمل محتوى البيانات الخام، لا الإعدادات وحدها."""

    def test_checkpoint_recomputed_when_raw_data_grows(self):
        cfg = _cfg()
        full = {c: _ohlcv("2025-01-01", "2025-03-01", s) for s, c in enumerate(("AAAUSDT", "BBBUSDT"))}
        old = {c: f[f.index < "2025-02-01"] for c, f in full.items()}
        ck = tempfile.mkdtemp()
        a = _build(cfg, old, ck)
        b = _build(cfg, full, ck)                          # نفس checkpoint_dir، بيانات أطول
        ref = _build(cfg, full)
        self.assertGreater(len(ref["base_params"]), len(a["base_params"]))
        np.testing.assert_array_equal(b["last_candles"], ref["last_candles"])

    def test_checkpoint_reused_for_identical_data(self):
        cfg = _cfg()
        frames = {"AAAUSDT": _ohlcv("2025-01-01", "2025-02-01", 5)}
        ck = tempfile.mkdtemp()
        a = _build(cfg, frames, ck)
        ns = _ns()
        calls = {"n": 0}

        def resample(df, tfs):
            calls["n"] += 1
            return ns["resample_timeframes"](df, tfs, config=cfg)
        b = _quiet(ns["build_dataset_from_loader"], [{"name": "AAAUSDT"}], lambda f, n: frames[n].copy(), resample,
                   config=cfg, max_workers=1, checkpoint_dir=ck)
        self.assertEqual(calls["n"], 0)                    # استئناف بلا إعادة معالجة
        np.testing.assert_array_equal(a["X_1h"], b["X_1h"])


class WindowContiguityTests(unittest.TestCase):
    """r2_02: نافذة الإدخال 32 شمعة متتالية بالضبط؛ ما يعبر فجوة يُحذف كعيّنات فجوة الهدف."""

    def test_no_window_spans_a_hole(self):
        ns, cfg = _ns(), _cfg()
        h = _ohlcv("2025-01-01", "2025-02-15", 3)
        h = h.drop(pd.date_range("2025-02-01 09:00", "2025-02-01 11:00", freq="1h", tz="UTC"))
        dfs = _quiet(ns["resample_timeframes"], h, ["1h"], config=cfg)
        X, y, b, last = _quiet(ns["prepare_single_asset"], dfs, config=cfg)
        idx = dfs["1h"].index
        ts = pd.to_datetime(last[:, 3].astype("int64"), utc=True)
        self.assertGreater(len(ts), 100)
        spans = np.array([(idx[idx.get_loc(t)] - idx[idx.get_loc(t) - 31]) / pd.Timedelta("1h") for t in ts])
        self.assertTrue((spans == 31).all(), spans[spans != 31])


class NormalizationKindTests(unittest.TestCase):
    """r2_03 وr2_04: أعلام *_available ثنائية لا تُصفَّر، وITD_TRADES_LOG يحفظ حجم الدرجة."""

    def _pw(self, w, names):
        ns = _ns()
        n = w.shape[0]
        return ns["process_windows"](w.astype("float32"), names, np.zeros(n, "float32"), np.ones(n, "float32"),
                                     "robust", ns["CONFIG"])

    def test_availability_flags_separate_covered_from_uncovered(self):
        ns = _ns()
        for f in ("EFF_RATIO_available", "VWAP_DEVIATION_available", "VOL_CONC_HHI_available",
                  "FUND_available", "OI_available", "ITD_available", "MET_available"):
            self.assertEqual(ns["classify_feature"](f), ns["BINARY_FLAG"], f)
            w = np.stack([np.ones((32, 1)), np.zeros((32, 1))])      # نافذة مغطّاة كلها / غير مغطّاة كلها
            out = self._pw(w, [f])
            self.assertEqual((float(out[0].mean()), float(out[1].mean())), (1.0, -1.0), f)

    def test_step_feature_keeps_step_size(self):
        for n_rows in (9, 5):                                          # أقلية ≥ 25% و< 25% من الصفوف
            z = {}
            for step in (1e-3, 1.0):
                w = np.full((1, 32, 1), 7.0)
                w[0, :n_rows, 0] += step
                z[step] = float(np.abs(self._pw(w, ["ITD_TRADES_LOG"])).max())
            self.assertLess(z[1e-3], 0.01 * z[1.0], (n_rows, z))
            self.assertLess(z[1.0], 4.99, (n_rows, z))                  # لا قصّ


class SealedHoldoutTests(unittest.TestCase):
    """r2_05: holdout مختوم ثابت التاريخ خارج train/val/test، لا يُفتح إلا صراحةً."""

    def _split(self, cfg, ds=None):
        ns = _ns()
        ds = ds or _grid_dataset(["A", "B", "C"], "2025-03-01", "2026-09-20")
        return ds, _quiet(ns["split_data"], ds, config=cfg)

    def test_preset_seals_the_tail(self):
        ns = _ns()
        ov = ns["HOURLY_W32_S8_OVERRIDES"]
        self.assertEqual(ov["split_dates"], {"train_end": "2025-06-24", "val_end": "2025-11-21"})
        hs = pd.Timestamp(ov["holdout_start"], tz="UTC")
        cfg = _cfg(split_dates=dict(ov["split_dates"]), holdout_start=ov["holdout_start"])
        ds, (tr, va, te) = self._split(cfg)
        gap = pd.Timedelta("33h")
        for part in (tr, va, te):
            self.assertLess(_ts(part).max(), hs - gap)
        self.assertGreaterEqual(_ts(te).min(), pd.Timestamp("2025-11-21", tz="UTC") + gap)
        with self.assertRaises(PermissionError):
            ns["split_holdout"](ds, config=cfg)
        cfg["open_holdout"] = True
        ho = _quiet(ns["split_holdout"], ds, config=cfg)
        self.assertGreaterEqual(_ts(ho).min(), hs)
        self.assertEqual(sorted(ho), ["A", "B", "C"])

    def test_no_holdout_keeps_old_behaviour(self):
        _, (tr, va, te) = self._split(_cfg(split_dates={"train_end": "2025-06-24", "val_end": "2025-11-21"}))
        self.assertEqual(_ts(te).max(), pd.Timestamp("2026-09-20", tz="UTC"))

    def test_holdout_overlapping_val_is_refused(self):
        with self.assertRaises(ValueError):
            self._split(_cfg(split_dates={"train_end": "2025-06-24", "val_end": "2025-11-21"},
                             holdout_start="2025-11-01"))

    def test_auto_dates_and_rolling_windows_stay_before_holdout(self):
        ns = _ns()
        cfg = _cfg(holdout_start="2026-07-15")
        ds, (tr, va, te) = self._split(cfg)
        self.assertLess(_ts(te).max(), pd.Timestamp("2026-07-15", tz="UTC"))
        sched = ns["rolling_split_schedule"](ds, test_span="30D", val_span="30D", train_span=None,
                                             initial_train_span="120D", config=cfg)
        self.assertLess(max(w["test_end"] for w in sched), pd.Timestamp("2026-07-15", tz="UTC"))

    def test_dataset_records_split_settings_and_main_reads_them(self):
        """خط الأنابيب يحفظ split_dates/holdout_start في البيانات، وخلية main (القسم ٣) تنقلها إلى CONFIG."""
        ov = _ns()["HOURLY_W32_S8_OVERRIDES"]
        cfg = _cfg(split_dates=dict(ov["split_dates"]), holdout_start=ov["holdout_start"])
        ds = _build(cfg, {"AAAUSDT": _ohlcv("2025-01-01", "2025-01-20", 1)})
        self.assertEqual((ds["split_dates"], ds["holdout_start"]), (ov["split_dates"], ov["holdout_start"]))

        ns = dict(_nbload.load_pipeline())                           # CONFIG مستقلّ عن بقية الاختبارات
        ns["load_data_from_drive"] = lambda **k: ds
        src = "".join(json.loads(_read("main.ipynb"))["cells"][7]["source"])
        _quiet(exec, compile(src, "main#cell7", "exec"), ns)
        self.assertEqual(ns["CONFIG"]["split_dates"], ov["split_dates"])
        self.assertEqual(ns["CONFIG"]["holdout_start"], ov["holdout_start"])


class Phase2RequiredTests(unittest.TestCase):
    """S6: إعداد 1h_s8 يرفض البناء بلا مجلدات المرحلة ٢ بدل الرجوع الصامت إلى 23 ميزة."""

    def test_missing_folders_fail_loudly(self):
        ns = _ns()
        root = Path(tempfile.mkdtemp())
        cfg = _cfg(phase2_data={"use_intraday_15m": "auto", "use_futures_metrics": "auto", "data_root": str(root),
                                "module_dirs": [ROOT]})
        with self.assertRaises(RuntimeError):
            ns["require_phase2_data"](config=cfg)
        for d in ("history_15m", "funding_rate", "open_interest", "futures_metrics"):
            (root / d).mkdir()
            (root / d / "X.csv.gz").write_bytes(b"")
        self.assertEqual(ns["require_phase2_data"](config=cfg), {"use_intraday_15m": True, "use_futures_metrics": True})
        self.assertEqual((cfg["phase2_data"]["use_intraday_15m"], cfg["phase2_data"]["use_futures_metrics"]),
                         (True, True))

    def test_hourly_preset_builder_requires_phase2(self):
        ns = _ns()
        saved = deepcopy(ns["CONFIG"])
        try:
            ns["CONFIG"]["phase2_data"]["data_root"] = tempfile.mkdtemp()      # جذر فارغ
            ns["CONFIG"]["phase2_data"]["module_dirs"] = [ROOT]
            with self.assertRaisesRegex(RuntimeError, "المرحلة ٢"):
                _quiet(ns["build_hourly_w32_s8_dataset"], save=False, estimate=False)
        finally:
            ns["CONFIG"].clear()
            ns["CONFIG"].update(saved)


class DocsPanelDirTests(unittest.TestCase):
    """r2_07: مسار مجلد اللوحة في تعليمات Colab = ما يحسبه main للتشغيل الموثَّق."""

    def test_docs_zip_path_matches_notebook(self):
        cells = json.loads(_read("main.ipynb"))["cells"]
        cell = next("".join(c["source"]) for c in cells
                    if c["cell_type"] == "code" and '"run_dir": "/content/drive' in "".join(c["source"]))
        expr = re.search(r'"run_dir": (.*?),\n\s*"epochs"', cell, re.S).group(1)
        env = {"TARGET_MODE": None, "REG_TARGET_SCALE": 100.0, "_am": True}
        env["globals"] = lambda: env
        panel_dir = eval("(" + expr + ")", env) + "_panel"
        docs = _read("docs", "research", "hourly_1h.md")
        for d in re.findall(r"cd (/content/drive/MyDrive/training_runs/\S+?) &&", docs):
            self.assertEqual(d, panel_dir)


if __name__ == "__main__":
    unittest.main()
