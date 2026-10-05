"""فريمان (1h أساسي + سياق 4h) عبر خط الأنابيب والنموذج واللوحة:  python -m unittest tests.test_multi_tf -v

ما يُثبَت (بيانات تركيبية بلا Drive):
  خط الأنابيب  — عدم النظر للمستقبل بالإعداد الحقيقي HOURLY_4H_OVERRIDES: تغيير كل ما بعد t لا يغيّر نافذة 1h ولا 4h لأي
                 عيّنة قبله (مع اختبار «أسنان»)، وشمعة 4h الأخيرة مغلقة عند t، وأشكال البيانات ومحاذاتها عبر العملات، وأن الإعدادات
                 القديمة (1h_s8 أحادي الفريم و2-فريم legacy) تُنتج بايتات مطابقة لما قبل التعديل.
  النموذج      — بناء متعدّد الفريمات وتمرير أمامي (قاموس/قائمة)، وخطوة تدريب بالمدرّب الفعلي، وأن الفريم الواحد يُبنى
                 كما كان (أسماء الطبقات وأشكال الأوزان وعدد المعاملات، وتحميل نقطة حفظ قديمة).
  اللوحة       — تمرير أمامي بفريمين، وتدريب حقبة، وبصمات الفريم الواحد لم تتغيّر.
المراجع «كما كانت» في tests/golden_pre_multi_tf.json (مأخوذة من الالتزام 4cb2d7b قبل التعديل).
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402

GOLDEN = json.load(open(os.path.join(ROOT, "tests", "golden_pre_multi_tf.json"), encoding="utf-8"))
H_NS = 3600 * 10 ** 9


GOLDEN_COMMIT = "4cb2d7b"   # the pre-change commit every recorded digest was taken from


def _data_golden(section):
    """Recorded byte-exact digests for this machine, or None when none were recorded for it.
    They depend on numpy's SIMD path (float64 exp/log differ in the last bits between X86_V4/AVX512 and X86_V3/AVX2)
    and on the library versions, so they exist for exactly two cases: the pinned versions on X86_V4 or on X86_V3.
    Anything else (other CPUs such as ARM, other versions, or CSP_GOLDEN_MODE=differential) returns None, and the
    caller compares against the old commit itself built on this machine — still byte-exact, never a tolerance."""
    if os.environ.get("CSP_GOLDEN_MODE") == "differential" or not _versions_match():
        return None
    from numpy._core._multiarray_umath import __cpu_features__ as cpu
    if cpu.get("X86_V4"):
        return GOLDEN[section]
    if cpu.get("X86_V3"):
        return {**GOLDEN[section], **GOLDEN["x86_v3"][section]}
    return None


def _old_commit_tree():
    """A temp dir holding the files of GOLDEN_COMMIT (via git archive), or None when git or the commit is missing."""
    d = tempfile.mkdtemp()
    try:
        tar = subprocess.run(["git", "archive", GOLDEN_COMMIT, "crypto_data_pipeline_v6.ipynb", "cross_asset",
                              "docs/research/audit"], cwd=ROOT, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=tar, check=True)
        return d
    except (OSError, subprocess.CalledProcessError):
        shutil.rmtree(d, ignore_errors=True)
        return None


def _versions_match():
    import importlib.metadata as md
    have = {k: md.version(k.replace("_", "-")) for k in GOLDEN["versions"]}
    return have == GOLDEN["versions"]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ohlcv(days, seed, start):
    """شموع 1h تركيبية (نفس المولّد الذي التُقطت به البصمات المرجعية)."""
    rng = np.random.default_rng(seed)
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
                         "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003), "close": close,
                         "volume": rng.random(n) * 1000 + 50}, index=idx)


def _digest(ds):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(ds):
        v = ds[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).tobytes())
            h.update(str((v.dtype, v.shape)).encode())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# خط الأنابيب
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class PipelineMultiTFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_pipeline()          # نطاق خاص بهذا الصنف: CONFIG المُعدَّل هنا لا يتسرّب لاختبارات أخرى

    def _build(self, overrides, tf_order, windows, mode=None, perturb_at=None, offset=None, coins=7, days=60,
               start="2025-01-01"):
        ns = self.ns
        ns["reset_config"]()
        _quiet(ns["apply_hourly_preset"], overrides)
        upd = {"tf_order": tf_order, "window_sizes": windows,
               "phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
               "funding_rate": {"enabled": False}, "open_interest": {"enabled": False}}
        if mode:
            upd.update(mode)
        if offset is not None:
            upd["higher_tf_offset"] = offset
        ns["update_config"](upd)
        _quiet(ns["refresh_features"])
        starts = [start, start + " 03:00", start + " 05:00", start + " 11:00", "2025-01-02", start, "2025-01-03 07:00"]

        def loader(fid, name):
            i = int(name[1])
            df = ohlcv(days, 40 + i, starts[i])
            if perturb_at is not None:                        # كل شمعة فتحها ≥ perturb_at تُستبدل بقيم عشوائية
                m = df.index >= perturb_at
                rng, k = np.random.default_rng(999 + i), int(m.sum())
                close = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, k)))
                op = close * (1 + rng.normal(0, 0.01, k))
                df.loc[m, "close"], df.loc[m, "open"] = close, op
                df.loc[m, "high"], df.loc[m, "low"] = np.maximum(op, close) * 1.02, np.minimum(op, close) * 0.98
                df.loc[m, "volume"] = rng.random(k) * 9000 + 1
            return df

        return _quiet(ns["build_dataset"], [{"name": f"C{i}USDT"} for i in range(coins)], load_asset_fn=loader,
                      resample_fn=ns["make_resample_fn"](ns["CONFIG"]), max_workers=1, config=ns["CONFIG"])

    # ── الإعدادات القديمة: بايتات مطابقة لما قبل التعديل ──
    def _old_preset_digests(self):
        ns = self.ns
        single = self._build(ns["HOURLY_W32_S8_OVERRIDES"], ["1h"], {"1h": 32}, days=45)
        legacy = self._build(ns["HOURLY_W32_S8_OVERRIDES"], ["1h", "4h"], {"1h": 32, "4h": 6}, days=45)
        return {"single_1h": _digest(single), "legacy_1h_4h": _digest(legacy)}

    def test_old_presets_byte_identical(self):
        have = self._old_preset_digests()
        want = _data_golden("pipeline_digests")
        if want is None:                           # no recorded digests for this machine: build the old commit here
            tree = _old_commit_tree()
            if tree is None:
                self.skipTest(f"no recorded digests for this CPU/versions and commit {GOLDEN_COMMIT} is not in the "
                              "clone (fetch full history) — cannot compare")
            new_ns = self.ns
            try:
                self.ns = _nbload.load_notebook(os.path.join(tree, "crypto_data_pipeline_v6.ipynb"),
                                                skip_contains=("اختبارات ذاتية لتعديلات هذا الدفتر",),
                                                ns={"RUN_HOURLY_4H_SELFTESTS": False})
                want = self._old_preset_digests()
            finally:
                self.ns = new_ns
                shutil.rmtree(tree, ignore_errors=True)
        single, legacy = have["single_1h"], have["legacy_1h_4h"]
        self.assertEqual(single, want["single_1h"], "1h_s8 أحادي الفريم لم يعد مطابقاً لما قبل التعديل")
        self.assertEqual(legacy, want["legacy_1h_4h"], "مسار 2-فريم legacy لم يعد مطابقاً لما قبل التعديل")

    # ── الإعداد الجديد ──
    def test_preset_shapes_alignment_and_dtype(self):
        ns = self.ns
        ds = self._build(ns["HOURLY_4H_OVERRIDES"], ns["HOURLY_4H_OVERRIDES"]["tf_order"],
                         ns["HOURLY_4H_OVERRIDES"]["window_sizes"])
        n, f = len(ds["base_params"]), len(ds["feature_order"])
        self.assertGreater(n, 500)
        self.assertEqual(ds["X_1h"].shape, (n, 32, f))
        self.assertEqual(ds["X_4h"].shape, (n, 32, f))
        self.assertEqual((ds["X_1h"].dtype, ds["X_4h"].dtype), (np.float16, np.float16))
        self.assertEqual((ds["timeframes"], ds["base_tf"], ds["higher_tf_mode"]), (["1h", "4h"], "1h", "closed"))
        self.assertEqual((ds["stride"], ds["forecast_horizon"], ds["reg_target_scale"], ds["holdout_start"]),
                         (8, 1, 100.0, "2026-07-15"))
        ts = ds["last_candles"][:, ns["TS_COL"]].astype("int64")
        self.assertTrue((ts % (8 * H_NS) == 0).all(), "نهايات النوافذ خارج شبكة 8h")
        per = {b["name"]: ts[b["start"]:b["end"]] for b in ds["asset_bounds"]}
        lo, hi = max(v.min() for v in per.values()), min(v.max() for v in per.values())
        common = [set(v[(v >= lo) & (v <= hi)].tolist()) for v in per.values()]
        self.assertTrue(all(c == common[0] for c in common) and len(common[0]) > 50, "الطوابع لا تتطابق بين العملات")
        self.assertTrue(np.isfinite(ds["X_4h"].astype("float32")).all())

    def test_preset_last_4h_bar_closed_at_t(self):
        """نوافذ 4h بإعداد الشبكة الحقيقي (stride 8، نافذة 32/32): كل صفوفها متتالية وآخرها شمعة إغلاقها ≤ t، وعلى الشبكة بالضبط
        t − 4h (فتحها) لأن t ≡ 0 (mod 8h)."""
        ns = self.ns
        ns["reset_config"]()
        _quiet(ns["apply_hourly_preset"], ns["HOURLY_4H_OVERRIDES"])
        ns["update_config"]({"phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False}})
        cfg = ns["CONFIG"]
        raw = ohlcv(40, 3, "2025-03-01 03:00")
        dfs = _quiet(ns["resample_timeframes"], raw, ["1h", "4h"], with_features=False, config=cfg)
        hours = lambda ix: pd.DataFrame({"h": ix.as_unit("ns").asi8 / H_NS}, index=ix)
        w, ends = ns["align_multi_timeframes_time_based"](
            {tf: hours(df.index) for tf, df in dfs.items()}, config=cfg)
        self.assertGreater(len(ends), 50)
        t = w["1h"][:, -1, 0].astype("float64")
        last4 = w["4h"][:, -1, 0].astype("float64")
        self.assertTrue(((last4 + 4) <= t).all(), "شمعة 4h إغلاقها بعد t")
        self.assertTrue((last4 == t - 4).all(), "على شبكة 8h آخر شمعة مغلقة هي التي فتحها t − 4h")
        self.assertTrue((np.diff(w["4h"][:, :, 0].astype("float64"), axis=1) == 4).all())
        self.assertTrue((t % 8 == 0).all())

    def _changed_before(self, a, b, cut):
        ts_col = self.ns["TS_COL"]

        def keyed(ds):
            ts = ds["last_candles"][:, ts_col].astype("int64")
            return {(bd["name"], int(ts[i])): i for bd in ds["asset_bounds"] for i in range(bd["start"], bd["end"])}
        ka, kb = keyed(a), keyed(b)
        n = c1 = c4 = 0
        for k, i in ka.items():
            if k[1] >= cut.value:
                continue
            n += 1
            j = kb.get(k)
            c1 += j is None or not np.array_equal(a["X_1h"][i], b["X_1h"][j])
            c4 += j is None or not np.array_equal(a["X_4h"][i], b["X_4h"][j])
        return n, c1, c4

    def test_no_lookahead_after_t_real_preset(self):
        """الدليل الحاسم: كل شمعة 1h فتحها ≥ القطع تُستبدل بقيم عشوائية (لكل العملات، ومنها مرجع BTC والميزات العابرة للأصول)
        ← نافذتا 1h و4h لكل عيّنة قبل القطع بلا أي فرق بايت. القطع 10:00 (منتصف شمعة 4h [08:00، 12:00) قيد التكوّن) وعيّنة
        t=08:00 قبله. ومعه اختبار «أسنان»: قاعدة تسرّب عمداً (legacy بإزاحة 1 = الشمعة الجارية) يجب أن يكشفها الاختبار نفسه."""
        ns = self.ns
        ov = ns["HOURLY_4H_OVERRIDES"]
        cut = pd.Timestamp("2025-02-05 10:00", tz="UTC")
        a = self._build(ov, ov["tf_order"], ov["window_sizes"])
        b = self._build(ov, ov["tf_order"], ov["window_sizes"], perturb_at=cut)
        n, c1, c4 = self._changed_before(a, b, cut)
        self.assertGreater(n, 400)
        self.assertEqual((c1, c4), (0, 0), f"نوافذ تغيّرت بتغيّر ما بعد القطع: 1h={c1}، 4h={c4} من {n}")
        self.assertFalse(np.array_equal(a["X_1h"][-30:], b["X_1h"][-30:]), "التغيير لم يصل إلى العيّنات المتأخّرة")
        leaky = dict(mode={"higher_tf_mode": "legacy", "x_storage_dtype": None}, offset=1)
        la = self._build(ov, ov["tf_order"], ov["window_sizes"], **leaky)
        lb = self._build(ov, ov["tf_order"], ov["window_sizes"], perturb_at=cut, **leaky)
        _, l1, l4 = self._changed_before(la, lb, cut)
        self.assertEqual(l1, 0)
        self.assertGreater(l4, 0, "الاختبار لم يكشف تسرّباً متعمَّداً — لا قيمة له")

    def test_embargo_covers_4h_context(self):
        ns = self.ns
        ov = ns["HOURLY_4H_OVERRIDES"]
        cfg = deepcopy(ns["CONFIG"])
        ns["_deep_update"](cfg, deepcopy(ns["HOURLY_PRESET"]))
        ns["_deep_update"](cfg, deepcopy(ov))
        ds = {"base_tf": "1h", "timeframes": ["1h", "4h"], "window_sizes": cfg["window_sizes"], "forecast_horizon": 1,
              "higher_tf_mode": "closed"}
        self.assertEqual(ns["embargo_candles"](ds, cfg), 32 * 4 + 1)                 # 129 ساعة لا 33
        self.assertEqual(ns["embargo_candles"]({**ds, "timeframes": ["1h"]}, cfg), 33)   # فريم واحد كما كان

    def test_embargo_widens_off_the_4h_phase_and_for_legacy(self):
        """R3-01/R3-02: الشبكة المشتركة (stride مضاعف لـ4h) تُبقي 129؛ أي طور آخر أو legacy يُضيف ratio-1 ولا يبقى 33."""
        ns = self.ns
        cfg = deepcopy(ns["CONFIG"])
        cfg.update(tf_order=["1h", "4h"], base_tf="1h", window_sizes={"1h": 32, "4h": 32}, forecast_horizon=1,
                   embargo_candles=None, align_windows_to_grid=True, higher_tf_mode="closed", stride=8)
        ds = {"base_tf": "1h", "timeframes": ["1h", "4h"], "window_sizes": cfg["window_sizes"], "forecast_horizon": 1}
        emb = lambda **kw: ns["embargo_candles"]({**ds, **{k: v for k, v in kw.items() if k in ("stride", "higher_tf_mode")}},
                                                 {**cfg, **{k: v for k, v in kw.items() if k == "align_windows_to_grid"}})
        self.assertEqual(emb(stride=8), 129)                                   # الإعداد المشحون: بلا تغيير
        self.assertEqual(emb(stride=4), 129)
        self.assertEqual(emb(stride=1), 132)                                   # الطور غير ثابت: 128 + 3 + 1
        self.assertEqual(emb(stride=5), 132)
        self.assertEqual(emb(stride=8, align_windows_to_grid=False), 132)
        self.assertEqual(emb(stride=8, higher_tf_mode="legacy"), 129)          # legacy: نافذة 4h تمتدّ 128h أيضاً (كانت 33)
        self.assertEqual(emb(stride=1, higher_tf_mode="legacy"), 132)

    def test_audit_normalization_float16_matches_float32(self):
        """R3-03: X مخزّن float16 يعطي std/verdict نفسها كـfloat32 (كان std=inf عند تجاوز مجموع المربّعات 65504)."""
        ns = self.ns
        x = np.random.default_rng(0).normal(0, 1, (4000, 32, 2)).astype("float16")     # 128k قيمة/ميزة: مربّعاتها > 65504
        t16 = ns["audit_normalization"](x, ["close", "volume"], verbose=False)
        t32 = ns["audit_normalization"](x.astype("float32"), ["close", "volume"], verbose=False)
        self.assertTrue(np.isfinite(t16["std"]).all())
        np.testing.assert_allclose(t16["std"], t32["std"], rtol=1e-6)
        self.assertEqual(list(t16["verdict"]), list(t32["verdict"]))


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# النموذج
# ══════════════════════════════════════════════════════════════════════════════════════════════════
_MODEL_NS = {}


def _model_ns():
    """model_v2 (خلاياه كلها عدا الاختبار الذاتي الذي يعمل عند الاستيراد) + المدرّب وbuild_target_configs من main."""
    if not _MODEL_NS:
        import tensorflow as tf  # noqa: F401
        mv = {"__name__": "audit_nb"}
        for cell in json.load(open(os.path.join(ROOT, "model_v2 (1).ipynb")))["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))
                     and ln.strip() != "run_model_selftests()"]
            _quiet(exec, compile("\n".join(lines), "model_v2", "exec"), mv)
        _MODEL_NS.update(mv)
    return _MODEL_NS


_TRAINER_NS = {}


def _trainer_ns():
    if not _TRAINER_NS:
        ns = {"__name__": "audit_nb"}
        for cell in json.load(open(os.path.join(ROOT, "trainer_framework_v2.ipynb")))["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if any(s in src for s in ("Smoke Test", "10.3) مثال", "12) K-Fold")):
                continue
            lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
            _quiet(exec, compile("\n".join(lines), "trainer", "exec"), ns)
        _nbload.workflow_package().load_into(ns, only=("training_config",))                 # build_target_configs
        _TRAINER_NS.update(ns)
    return _TRAINER_NS


ANTI = None


def _cfg(mv, **over):
    """إعداد مقاومة الحفظ الحقيقي برأسَي high/low (كدفتر main مع close معلَّق)، صغيراً للسرعة."""
    heads = {t: ["nig_regression", "binary_classification"] for t in ("high", "low")}
    return dict(mv["ANTI_MEMORIZATION_CONFIG"], price_targets=("high", "low"), head_types=heads, **over)


class MultiTFModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mv = _model_ns()
        cls.build = staticmethod(cls.mv["build_model_fn"])

    def test_single_tf_build_unchanged(self):
        """فريم واحد يُبنى كما كان تماماً: نفس أسماء الطبقات وأنواعها وأشكال أوزانها وأسماء المدخلات وعدد المعاملات — ما يجعل
        نقاط الحفظ القديمة تُحمَّل. المرجع من الالتزام 4cb2d7b."""
        mv = self.mv
        tiny = dict(d_model=16, num_layers=2, num_heads=2, num_kv_heads=1, head_hidden=16, class_head_hidden=8)
        heads = {t: ["nig_regression", "binary_classification"] for t in ("high", "low")}
        cases = {
            "default": dict(),
            "anti_mem": dict(mv["ANTI_MEMORIZATION_CONFIG"], price_targets=("high", "low"), head_types=heads),
            "transformer_tiny": dict(tiny, price_targets=("high", "low"), head_types=heads),
            "tcn_film_coin": dict(tiny, encoder="tcn", level_film=True, n_coins=5, level_passthrough=True,
                                  level_norm="batch"),
            "two_tower_gate": dict(tiny, encoder="gru", architecture="two_tower", fusion="gate",
                                   level_passthrough=True),
        }
        for name, cfg in cases.items():
            m = self.build(32, 43, config=cfg)
            want = GOLDEN["model"][name]
            got = [[l.name, type(l).__name__, [list(w.shape) for w in l.weights]] for l in m.layers]
            self.assertEqual(got, want["layers"], name)
            self.assertEqual([i.name for i in m.inputs], want["inputs"], name)
            self.assertEqual(int(m.count_params()), want["params"], name)
            self.assertEqual(list(m.output.keys()), want["outputs"], name)
        # قاموس بفريم واحد = فريم واحد (يبني النموذج نفسه)
        one = self.build({"1h": 32}, {"1h": 43}, config=cases["anti_mem"])
        self.assertEqual([l.name for l in one.layers], [l[0] for l in GOLDEN["model"]["anti_mem"]["layers"]])

    def test_single_tf_old_checkpoint_layout_loads(self):
        """أوزان نموذج أحادي الفريم تُحمَّل في بناء جديد بلا أي تعديل (نفس أسماء الطبقات = نفس مسارات ملف h5)."""
        m1 = self.build(32, 43, config=_cfg(self.mv))
        m2 = self.build({"1h": 32}, 43, config=_cfg(self.mv))
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "w.weights.h5")
            m1.save_weights(path)
            m2.load_weights(path)
            x = np.random.default_rng(0).normal(size=(4, 32, 43)).astype("float32")
            o1, o2 = m1(x, training=False), m2(x, training=False)
            for k in o1:
                np.testing.assert_allclose(o1[k].numpy(), o2[k].numpy(), atol=1e-6, err_msg=k)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_multi_tf_build_and_forward(self):
        cfg = _cfg(self.mv)
        one = self.build(32, 43, config=cfg)
        two = self.build({"1h": 32, "4h": 24}, 43, config=cfg)
        self.assertEqual([i.name for i in two.inputs], ["1h", "4h"])
        self.assertEqual([tuple(i.shape[1:]) for i in two.inputs], [(32, 43), (24, 43)])
        names = [l.name for l in two.layers]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("branch_concat", names)
        # فرع لكل فريم بأوزان منفصلة وبكل خيارات مقاومة الحفظ (إسقاط قنوات، ضجيج، تمرير المستوى، BatchNorm، GRU)
        for tf in ("1h", "4h"):
            for base in ("instance_norm", "feature_dropout", "input_noise", "level_symlog", "level_bn", "gru_1",
                         "readout_fc", "linear_path", "stats_bn"):
                self.assertIn(f"{base}_{tf}", names, f"{base}_{tf}")
        self.assertNotIn("instance_norm", names)
        self.assertGreater(two.count_params(), 1.7 * one.count_params())
        # تغذية بقاموس {فريم: X} = بقائمة بالترتيب نفسه = tuple (كما تفعل أدوات التقييم)
        rng = np.random.default_rng(1)
        x1, x4 = rng.normal(size=(6, 32, 43)).astype("float32"), rng.normal(size=(6, 24, 43)).astype("float32")
        od = two({"1h": x1, "4h": x4}, training=False)
        ol = two([x1, x4], training=False)
        ot = two((x1, x4), training=False)
        self.assertEqual(set(od), set(one(x1, training=False)))
        for k in od:
            self.assertEqual(tuple(od[k].shape), (6, 1), k)
            np.testing.assert_allclose(od[k].numpy(), ol[k].numpy(), atol=1e-6)
            np.testing.assert_allclose(od[k].numpy(), ot[k].numpy(), atol=1e-6)
        # X مخزَّنة float16 تُرفع إلى float32 عند النداء
        o16 = two({"1h": x1.astype("float16"), "4h": x4.astype("float16")}, training=False)
        for k in od:
            np.testing.assert_allclose(od[k].numpy(), o16[k].numpy(), atol=5e-2, err_msg=k)
        # الفرع الثاني يؤثّر فعلاً في المخرج
        x4b = x4 + 1.0
        ob = two({"1h": x1, "4h": x4b}, training=False)
        self.assertFalse(np.allclose(od["y_high"].numpy(), ob["y_high"].numpy()))

    def test_multi_tf_shared_coin_embedding_and_validation(self):
        cfg = dict(_cfg(self.mv), n_coins=5)
        m = self.build({"1h": 32, "4h": 16}, {"1h": 43, "4h": 43}, config=cfg)
        self.assertEqual([i.name for i in m.inputs], ["1h", "4h", "coin_id"])
        self.assertEqual([l.name for l in m.layers].count("coin_embedding"), 1)
        rng = np.random.default_rng(2)
        o = m({"1h": rng.normal(size=(3, 32, 43)).astype("float32"), "4h": rng.normal(size=(3, 16, 43)).astype("float32"),
               "coin_id": np.array([0, 1, 2])}, training=False)
        self.assertEqual(tuple(o["y_high"].shape), (3, 1))
        with self.assertRaises(ValueError):
            self.build({"1h": 32, "4h": 16}, {"1h": 43}, config=_cfg(self.mv))

    def test_one_batch_train_step_with_real_trainer(self):
        """GenericTrainer (main القسم ٥) بنموذج بفريمين وبيانات tf.data بقاموس {فريم: X}: خسارة منتهية، وتتحدّث أوزان الفرعين."""
        import tensorflow as tf
        tn, mv = _trainer_ns(), self.mv
        cfg = tn["build_config"]({
            "run": {"run_dir": tempfile.mkdtemp(), "epochs": 1, "batch_size": 8, "verbose": 0, "train_mode": "new"},
            "targets": tn["build_target_configs"](("high", "low"), label_smoothing=0.1),
        })
        model_cfg = _cfg(mv, d_model=16, num_layers=1, head_hidden=16, class_head_hidden=8)
        builder = lambda: self.build({"1h": 16, "4h": 8}, 10, config=model_cfg)  # noqa: E731
        rng = np.random.default_rng(3)
        n = 16
        x = {"1h": rng.normal(size=(n, 16, 10)).astype("float32"), "4h": rng.normal(size=(n, 8, 10)).astype("float32")}
        y = {f"y_{t}_reg": rng.normal(scale=0.01, size=n).astype("float32") for t in ("high", "low")}
        y.update({f"y_{t}_class": rng.integers(0, 2, size=n).astype("float32") for t in ("high", "low")})
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8, drop_remainder=True)
        trainer, callbacks, initial_epoch = _quiet(tn["build_training_system"], builder, cfg, next(iter(ds)))
        before = {v.path: v.numpy().copy() for v in trainer.model.trainable_variables}
        _quiet(trainer.fit, ds, epochs=1, steps_per_epoch=1, verbose=0)
        after = {v.path: v.numpy() for v in trainer.model.trainable_variables}
        moved = {tf_: any(not np.allclose(before[p], after[p]) for p in before if p.split("/")[0].endswith(f"_{tf_}"))
                 for tf_ in ("1h", "4h")}
        self.assertEqual(moved, {"1h": True, "4h": True}, moved)
        out = trainer.model({k: v[:4] for k, v in x.items()}, training=False)
        self.assertTrue(all(np.isfinite(v.numpy()).all() for v in out.values()))


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# اللوحة
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class PanelMultiTFTests(unittest.TestCase):
    @staticmethod
    def _two_tf_split(seed, name, start_day=18000, n_days=30, x_dtype="float16"):
        from cross_asset import selftest as st
        from cross_asset.data import PanelSplit
        base = st.synthetic_split(20, n_days, 8, 5, start_day=start_day, seed=seed, signal=2.0, name=name)
        rng = np.random.default_rng(seed + 100)
        X4 = rng.normal(size=(base.n, 6, 5)).astype("float32")
        y = {}
        for i, t in enumerate(base.targets):
            y[f"y_{t}_class"] = base.ycls[:, i]
            y[f"y_{t}_reg"] = base.yreg[:, i]
        X = {"1h": base.X.astype(x_dtype), "4h": X4.astype(x_dtype)}
        return PanelSplit(X, y, base.lc, base.assets, name), base

    def test_two_tf_batches_gather_same_rows_as_float32(self):
        ps, base = self._two_tf_split(1, "train")
        self.assertEqual(ps.tfs, ["1h", "4h"])
        self.assertEqual(ps.n, base.n)
        self.assertTrue(ps.check()["ok"])
        b = ps.make_batch([0, 1])
        self.assertEqual(set(b["x"]), {"1h", "4h"})
        m = len(b["idx"])
        self.assertEqual((b["x"]["1h"].shape, b["x"]["4h"].shape), ((m, 8, 5), (m, 6, 5)))
        self.assertTrue(all(a.dtype == np.float32 for a in b["x"].values()))         # float16 مخزَّنة ← float32 عند الدفعة
        np.testing.assert_allclose(b["x"]["1h"], base.X[b["idx"]], atol=2.5e-3 * 5)
        sub = ps.take(b["idx"])
        self.assertEqual(sub.tfs, ["1h", "4h"])
        self.assertEqual(len(sub.X["4h"]), m)
        self.assertNotEqual(ps.content_hash(), base.content_hash())                   # الفريم الثاني داخل البصمة

    def test_panel_split_from_list_of_tfs(self):
        from cross_asset.data import panel_split_from
        ps, _ = self._two_tf_split(2, "x")
        n = ps.n
        y = {k: np.zeros(n, "float32") for t in ("high", "low", "close") for k in (f"y_{t}_class", f"y_{t}_reg")}
        split = {"X_1h": ps.X["1h"], "X_4h": ps.X["4h"], "y": y, "last_candles": ps.lc, "reg_target_scale": 1.0}
        one = panel_split_from(split, "1h")
        self.assertIsNone(one.tfs)                                                   # فريم واحد: مصفوفة كما كان
        self.assertIsNone(panel_split_from(split, ["1h"]).tfs)
        two = panel_split_from(split, ["1h", "4h"])
        self.assertEqual(two.tfs, ["1h", "4h"])
        cut = n // 2
        parts = {a: {**split, "X_1h": ps.X["1h"][s], "X_4h": ps.X["4h"][s], "last_candles": ps.lc[s],
                     "y": {k: v[s] for k, v in y.items()}} for a, s in (("A", slice(0, cut)), ("B", slice(cut, n)))}
        tdict = panel_split_from(parts, ["1h", "4h"])
        self.assertEqual((tdict.n, len(tdict.X["4h"])), (n, n))

    def test_panel_forward_with_two_tfs(self):
        import tensorflow as tf
        from cross_asset.model import build_panel_model, tiny_encoder
        ps, _ = self._two_tf_split(3, "train")
        enc = tiny_encoder({"1h": 8, "4h": 6}, 5)
        self.assertEqual([i.name for i in enc.inputs], ["1h", "4h"])
        m = build_panel_model(enc, {"1h": 8, "4h": 6}, 5, d_model=16, num_heads=4, dropout=0.0)
        b = ps.make_batch([0, 1, 2])
        batch = {"x": {k: tf.constant(v) for k, v in b["x"].items()}, "day": tf.constant(b["day"]),
                 "pos": tf.constant(b["pos"])}
        out = m(batch, training=False)
        self.assertEqual(tuple(out["logit"].shape), (len(b["idx"]), 3))
        self.assertEqual(tuple(out["mu"].shape), (len(b["idx"]), 3))
        # المخرج يعتمد على سياق 4h
        batch4 = dict(batch, x={"1h": batch["x"]["1h"], "4h": batch["x"]["4h"] + 2.0})
        self.assertFalse(np.allclose(out["logit"].numpy(), m(batch4, training=False)["logit"].numpy()))
        # مع الانتباه عبر العملات: عملة أخرى في اليوم نفسه تغيّر المخرج (الفريمان يمرّان بالمسار نفسه)
        d = out["logit"].numpy()
        x1 = batch["x"]["1h"].numpy().copy()
        x1[1] += 3.0
        alt = m(dict(batch, x={"1h": tf.constant(x1), "4h": batch["x"]["4h"]}), training=False)["logit"].numpy()
        same_day = batch["day"].numpy() == batch["day"].numpy()[0]
        if same_day.sum() > 1:
            self.assertFalse(np.allclose(d[0], alt[0]))

    def test_panel_trains_one_epoch_with_two_tfs_and_single_tf_fingerprint_unchanged(self):
        from cross_asset import selftest as st
        from cross_asset.experiment import run_panel_variant
        from cross_asset.model import tiny_encoder
        tcfg = dict(epochs=1, patience=3, batch_samples=128, max_days=8, max_steps_per_epoch=3, ic_min_coins=5)

        class TwoTfBase:
            """يحاكي نموذج build_nig_timenet_v2 متعدّد الفريمات: trunk_drop فوق فرعين مُدخلهما باسم الفريم."""
            def __new__(cls):
                import tensorflow as tf
                L = tf.keras.layers
                ins = {"1h": L.Input((8, 5), name="1h"), "4h": L.Input((6, 5), name="4h")}
                h = L.Concatenate()([L.GRU(8)(L.BatchNormalization()(i)) for i in ins.values()])
                h = L.Dropout(0.1, name="trunk_drop")(h)
                return tf.keras.Model(list(ins.values()), {"y_close": L.Dense(1)(h)})

        tr, _ = self._two_tf_split(11, "train")
        va, _ = self._two_tf_split(12, "val", start_day=18100)
        te, _ = self._two_tf_split(13, "test", start_day=18200)
        d = tempfile.mkdtemp()
        try:
            v, t, state, trainer = run_panel_variant(tr, va, te, TwoTfBase, {"1h": 8, "4h": 6}, 5, d, "A_ic", 0, None,
                                                     tcfg, verbose=False)
            self.assertEqual(state["epoch"], 1)
            self.assertTrue(np.isfinite(state["history"][0]["train_loss"]))
            self.assertEqual((len(v), len(t)), (va.n, te.n))
            self.assertTrue(np.isfinite(v["p_up_close"]).all())
            # البصمة تحوي نوافذ كل فريم (فريم واحد: البصمة القديمة نفسها — أدناه)
            self.assertEqual(trainer.fingerprint["tfs"], {"1h": [8, 5], "4h": [6, 5]})
        finally:
            shutil.rmtree(d, ignore_errors=True)

        s_tr = st.synthetic_split(20, 40, 8, 5, seed=1, signal=2.0, name="train")
        s_va = st.synthetic_split(20, 20, 8, 5, start_day=18200, seed=2, signal=2.0, name="val")
        s_te = st.synthetic_split(20, 20, 8, 5, start_day=18300, seed=3, signal=2.0, name="test")
        d = tempfile.mkdtemp()
        try:
            _, _, state, _ = run_panel_variant(s_tr, s_va, s_te, lambda: st._TinyBase(8, 5), 8, 5, d, "A_ic", 0, None,
                                               tcfg | {"batch_samples": 256}, verbose=False)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(state["fingerprint"], GOLDEN["panel"]["fingerprint"])   # config hash: CPU-independent
        g = _data_golden("panel")
        if g is None:                              # no recorded digests for this machine: hash the old commit's data here
            tree = _old_commit_tree()
            if tree is None:
                self.skipTest(f"no recorded digests for this CPU/versions and commit {GOLDEN_COMMIT} is not in the "
                              "clone (fetch full history) — cannot compare")
            code = ("import sys; sys.path.insert(0, sys.argv[1]); from cross_asset import selftest as st\n"
                    "s = [st.synthetic_split(20, 40, 8, 5, seed=1, signal=2.0, name='train'),\n"
                    "     st.synthetic_split(20, 20, 8, 5, start_day=18200, seed=2, signal=2.0, name='val'),\n"
                    "     st.synthetic_split(20, 20, 8, 5, start_day=18300, seed=3, signal=2.0, name='test')]\n"
                    "print('|'.join(x.content_hash() for x in s))")
            try:
                fp = subprocess.run([sys.executable, "-c", code, tree], capture_output=True, text=True, check=True,
                                    env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}).stdout.strip().splitlines()[-1]
            finally:
                shutil.rmtree(tree, ignore_errors=True)
            g = {"data_fp": fp, "train_content_hash": fp.split("|")[0]}
        self.assertEqual(state["data_fp"], g["data_fp"])
        self.assertEqual(s_tr.content_hash(), g["train_content_hash"])


if __name__ == "__main__":
    unittest.main()
