"""اختبارات نموذج اللوحة عبر العملات (سريعة، بيانات تركيبية، بلا Drive):  python -m unittest tests.test_cross_asset -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cross_asset import selftest as st  # noqa: E402


class PanelDataTests(unittest.TestCase):
    def test_grouping(self):
        self.assertTrue(st.test_grouping())

    def test_batching(self):
        self.assertTrue(st.test_batching())

    def test_subset_keeps_same_coins_across_splits(self):
        a = st.synthetic_split(seed=0).subset(last_days=10, coins=5, seed=7)
        b = st.synthetic_split(seed=0, start_day=18100).subset(last_days=10, coins=5, seed=7)
        self.assertLessEqual(len(set(a.assets)), 5)
        self.assertEqual(set(a.assets), set(b.assets))
        self.assertTrue(a.check()["ok"])
        self.assertEqual(a.n_days, 10)

    def test_class_label_encoding_agnostic(self):
        """خط الأنابيب يُصدر _class بترميز 1/0، وretarget_splits (main) وبيانات أقدم بـ ±1 — كلاهما يعطي نفس ycls."""
        import numpy as np
        from cross_asset.data import PanelSplit
        ps = st.synthetic_split(seed=3)                      # يبني تسميات ±1
        for enc in ("pm1", "unit"):
            y = {}
            for i, t in enumerate(ps.targets):
                up = ps.ycls[:, i]
                y[f"y_{t}_class"] = up * 2.0 - 1.0 if enc == "pm1" else up.copy()
                y[f"y_{t}_reg"] = ps.yreg[:, i]
            other = PanelSplit(ps.X, y, ps.lc, ps.assets, enc, ps.targets)
            np.testing.assert_array_equal(other.ycls, ps.ycls)
            self.assertEqual(set(np.unique(other.ycls).tolist()), {0.0, 1.0})
        sub = ps.take(np.arange(0, ps.n, 3))                 # take يمرّ بترميز 1/0 ولا يفقد التسميات
        np.testing.assert_array_equal(sub.ycls, ps.ycls[np.arange(0, ps.n, 3)])


class PanelModelTests(unittest.TestCase):
    def test_masking(self):
        self.assertTrue(st.test_masking())

    def test_ic_loss(self):
        self.assertTrue(st.test_ic_loss())

    def test_training_export_resume(self):
        st.test_training()

    def test_attention_uses_same_day_coins(self):
        """ضابط إيجابي: المعلومة في انحراف العملة عن يومها (إزاحة يومية مشتركة كبيرة على الميزة). A يلتقطها في
        AUC المُجمَّع عبر الأيام، وB (كل عملة وحدها) لا يستطيع معرفة الإزاحة."""
        import tensorflow as tf
        from cross_asset.model import build_panel_model, tiny_encoder
        from cross_asset.train import PanelTrainer, robust_scales
        from sklearn.metrics import roc_auc_score
        import tempfile
        kw = dict(n_assets=40, seq_len=8, n_features=5, signal=3.0, day_shift=3.0)
        tr = st.synthetic_split(n_days=150, seed=1, **kw)
        va = st.synthetic_split(n_days=40, start_day=18300, seed=2, **kw)
        auc = {}
        for att in (True, False):
            tf.keras.utils.set_random_seed(0)
            m = build_panel_model(tiny_encoder(8, 5), 8, 5, d_model=16, num_heads=4, cross_attention=att, dropout=0.0)
            cfg = dict(epochs=8, patience=20, batch_samples=200, max_days=8, lr_warmup_epochs=0,
                       lr_schedule={"type": "constant"}, lr_initial=3e-3, ema_window_epochs=0.5, label_smoothing=0.0)
            t = PanelTrainer(m, cfg, tempfile.mkdtemp(), robust_scales(tr.yreg), 8, 5, verbose=False)
            t.fit(tr, va)
            t.load_best()
            logit, _ = t.predict(va)
            auc[att] = roc_auc_score(va.ycls[:, 2], logit[:, 2])
        print(f"\n   AUC close (val): A {auc[True]:.3f} | B {auc[False]:.3f}")
        self.assertGreater(auc[True], auc[False] + 0.05, auc)


if __name__ == "__main__":
    unittest.main()
