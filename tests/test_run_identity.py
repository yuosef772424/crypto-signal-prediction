"""A run has a persisted identity (audit 2026-10-31, class C3, PR 4): identity.json is written when a run starts and compared when it
resumes; the seed is a setting and fixes the batch order.

Guards:
1. RESUME CHECK: a run dir trained with loss_weight 1.0 and resumed with 5.0 raises, naming the changed key (it used to continue
   silently under the new objective); each of the objective keys below does the same; only `epochs` (and the other keys the
   fingerprint ignores) may change; a moved run folder (run_dir / mirror_dir) still resumes.
2. LEGACY DIRS: a run dir without identity.json resumes UNCHANGED (same epoch, no error), prints ONE warning and adopts the identity
   (the file now exists and the next resume compares against it). A corrupt identity.json raises.
3. SEED: TrainSettings.seed is a strict field whose default (42) is the seed model init already used (trainer default run.seed) and does
   not change the trainer config fingerprint. Same seed => identical first batches (workflow.run.make_datasets) and identical
   epoch-1 loss; different seed => different batches.
4. INTENDED BEHAVIOR CHANGE: before this PR the training dataset was built BEFORE any seeding (shuffle seed=None), so two builds with
   the same run.seed produced different batch orders; `test_unseeded_pre_pr_order_is_not_reproducible` pins the old behavior as the
   reason for the change (batch order of seeded runs changes on purpose, model init does not).
    python -m pytest tests/test_run_identity.py -q
"""
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from core.seeds import seed_everything  # noqa: E402
import run_fixture as rf  # noqa: E402
import test_run_steps as trs  # noqa: E402

_NS = {}


def _ns():
    if not _NS:
        _NS["ns"] = rf.run_namespace()
    return _NS["ns"]


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def _builder():
    inp = tf.keras.Input(shape=(8,), name="features")
    h = tf.keras.layers.Dense(16, activation="relu")(inp)
    outputs = {"y_b_logits": tf.keras.layers.Dense(1, activation="sigmoid", name="y_b_logits", dtype="float32")(h)}
    return tf.keras.Model(inputs=inp, outputs=outputs)


def _config(run_dir, loss_weight=1.0, **over):
    ns = _ns()
    cfg = {
        "run": {"run_dir": run_dir, "epochs": 2, "batch_size": 32, "verbose": 0},
        "targets": {"b": {"true_key": "y_b", "task_type": "classification", "output_keys": {"logits": "y_b_logits"},
                          "binary": True, "loss_weight": loss_weight}},
        "loss": {"use_uncertainty_weighting": False, "schedules": {}},
        "optimizer": {"lr_initial": 1e-3, "lr_warmup_epochs": 0, "lr_schedule": {"type": "constant"}},
        "callbacks": {"early_stopping": {"patience": 100, "monitor": "val_loss"}, "metrics_log_every": 1},
    }
    return ns["build_config"](ns["deep_update"](cfg, over)) if over else ns["build_config"](cfg)


def _data(n=256):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 8)).astype("float32")
    y = {"y_b": rng.integers(0, 2, size=(n, 1)).astype("float32")}
    return X, y


def _train(cfg, epochs, seed_batches=True):
    """Build (new / resume per cfg) and fit up to ``epochs``; returns (trainer, history, initial_epoch)."""
    ns = _ns()
    X, y = _data()
    if seed_batches:
        seed_everything(cfg["run"]["seed"])      # what workflow.run.make_datasets does before building the dataset
    ds = ns["make_shuffled_dataset"](X, y, 32, seed=cfg["run"]["seed"] if seed_batches else None)
    tf.keras.backend.clear_session()
    ns["TRAINER_REGISTRY"].pop(cfg["run"]["run_dir"], None)
    (trainer, cbs, ie), _ = _quiet(ns["build_training_system"], _builder, cfg, next(iter(ds)))
    hist, _ = _quiet(trainer.fit, ds, validation_data=ds, initial_epoch=ie, epochs=epochs, callbacks=cbs, verbose=0)
    return trainer, hist, ie


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="run_identity_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_dir(self, name="run"):
        return os.path.join(self.tmp, name)


class ResumeChecksTheObjective(_Tmp):
    def test_identity_json_is_written_at_start(self):
        rd = self.run_dir()
        _train(_config(rd), 1)
        d = json.load(open(os.path.join(rd, "identity.json"), encoding="utf-8"))
        ns = _ns()
        self.assertEqual(d["config_fingerprint"], ns["config_fingerprint"](_config(rd), ns["IDENTITY_IGNORED_RUN_KEYS"]))
        self.assertEqual((d["seed"], d["parent_hashes"], d["data_fingerprint"], d["schema_version"]), (42, [], None, 1))
        self.assertTrue(d["code_version"] is None or len(d["code_version"]) == 40)   # git sha when run from a checkout
        self.assertEqual(d["config_payload"]["targets"]["b"]["loss_weight"], 1.0)

    def test_resume_with_changed_loss_weight_raises_and_names_the_key(self):
        rd = self.run_dir()
        _train(_config(rd, loss_weight=1.0), 2)
        with self.assertRaises(_ns()["RunIdentityMismatch"]) as cm:
            _train(_config(rd, loss_weight=5.0), 3)
        msg = str(cm.exception)
        self.assertIn("targets.b.loss_weight: 1.0 -> 5.0", msg)
        self.assertIn("warm_start", msg)

    def test_every_objective_change_raises(self):
        changes = {
            "label_smoothing": {"targets": {"b": {"label_smoothing": 0.1}}},
            "lr": {"optimizer": {"lr_initial": 5e-4}},
            "weighting": {"loss": {"use_uncertainty_weighting": True}},
            "schedule": {"loss": {"schedules": {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 2, "schedule": "linear"}}}},
            "monitor": {"callbacks": {"early_stopping": {"patience": 7}}},
        }
        rd = self.run_dir()
        _train(_config(rd), 1)
        for name, over in changes.items():
            with self.subTest(name):
                with self.assertRaises(_ns()["RunIdentityMismatch"]):
                    _train(_config(rd, **over), 2)

    def test_changing_only_ignored_keys_resumes(self):
        rd = self.run_dir()
        _train(_config(rd), 1)
        cfg = _config(rd, **{"run": {"epochs": 5, "batch_size": 64, "verbose": 1, "seed": 7}})
        trainer, _, ie = _train(cfg, 2)
        self.assertEqual(ie, 1)   # resumed from epoch 1, no error

    def test_moved_run_folder_still_resumes(self):
        a, b = self.run_dir("a"), self.run_dir("b")
        _train(_config(a), 1)
        shutil.copytree(a, b)
        _, _, ie = _train(_config(b, **{"run": {"mirror_dir": os.path.join(self.tmp, "mirror")}}), 2)
        self.assertEqual(ie, 1)

    def test_new_start_archives_the_old_identity_and_writes_a_new_one(self):
        rd = self.run_dir()
        _train(_config(rd, loss_weight=1.0), 1)
        _train(_config(rd, loss_weight=5.0, **{"run": {"train_mode": "new"}}), 1)
        d = json.load(open(os.path.join(rd, "identity.json"), encoding="utf-8"))
        self.assertIn("5.0", json.dumps(d["config_payload"]))
        archived = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join(rd, "_archive")) for f in fs if f == "identity.json"]
        self.assertEqual(len(archived), 1)
        self.assertIn("1.0", json.dumps(json.load(open(archived[0], encoding="utf-8"))["config_payload"]))


class LegacyRunDirs(_Tmp):
    def test_dir_without_identity_resumes_unchanged_with_one_warning_and_adopts(self):
        rd = self.run_dir()
        _train(_config(rd), 2)
        os.remove(os.path.join(rd, "identity.json"))            # a run dir written before this PR
        ns = _ns()
        X, y = _data()
        ds = ns["make_shuffled_dataset"](X, y, 32, seed=42)
        cfg = _config(rd, loss_weight=5.0)                      # not verifiable: nothing stored to compare against
        (res, out) = _quiet(ns["build_training_system"], _builder, cfg, next(iter(ds)))
        _, _, initial_epoch = res
        self.assertEqual(initial_epoch, 2)                      # resumed exactly where it stopped
        self.assertEqual(out.count("has no identity.json"), 1)
        self.assertTrue(os.path.exists(os.path.join(rd, "identity.json")))
        # the adopted identity is now enforced
        with self.assertRaises(ns["RunIdentityMismatch"]):
            _train(_config(rd, loss_weight=1.0), 3)

    def test_corrupt_identity_raises(self):
        rd = self.run_dir()
        _train(_config(rd), 1)
        with open(os.path.join(rd, "identity.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(ValueError) as cm:
            _train(_config(rd), 2)
        self.assertIn("not a valid run identity", str(cm.exception))


class IdentityUnit(unittest.TestCase):
    def test_diff_and_allow_changes(self):
        ns = _ns()
        tmp = tempfile.mkdtemp(prefix="run_identity_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        a = ns["collect_run_identity"](_config(tmp, loss_weight=1.0))
        b = ns["collect_run_identity"](_config(tmp, loss_weight=5.0))
        ns["write_run_identity"](tmp, a)
        with self.assertRaises(ns["RunIdentityMismatch"]):
            ns["check_run_identity"](tmp, b)
        self.assertEqual(ns["check_run_identity"](tmp, b, allow_changes=("targets.b.loss_weight",)), "match")
        self.assertEqual(ns["check_run_identity"](tmp, a), "match")
        self.assertEqual(ns["RunIdentity"].from_dict(a.to_dict()), a)
        with self.assertRaises(ValueError):
            ns["RunIdentity"].from_dict({**a.to_dict(), "typo": 1})

    def test_data_fingerprint_mismatch_raises_only_when_both_sides_have_one(self):
        ns = _ns()
        tmp = tempfile.mkdtemp(prefix="run_identity_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg = _config(tmp)
        ns["write_run_identity"](tmp, ns["collect_run_identity"](cfg, data_fingerprint="aaa"))
        with self.assertRaises(ns["RunIdentityMismatch"]):
            ns["check_run_identity"](tmp, ns["collect_run_identity"](cfg, data_fingerprint="bbb"))
        self.assertEqual(ns["check_run_identity"](tmp, ns["collect_run_identity"](cfg)), "match")

    def test_fingerprint_without_the_new_kwarg_is_the_old_one(self):
        ns = _ns()
        cfg = _config("/x/run")
        self.assertEqual(ns["config_fingerprint"](cfg), ns["config_fingerprint"](cfg, ()))
        self.assertNotEqual(ns["config_fingerprint"](cfg), ns["config_fingerprint"](cfg, ("run_dir",)))   # run_dir is part of the plain one


class SeedIsASetting(_Tmp):
    def test_default_is_the_trainer_default_seed_and_validated(self):
        ns = _ns()
        S = ns["RunSettings"]
        self.assertIsNone(S().train.seed)      # None = the trainer's default seed (what model init already used), not passed on
        kit = ns["Toolkit"].from_namespace(ns)
        res = ns["run_main"](S().updated({"train": {"run_main_training": False}}), kit, dataset=trs._dataset(False, None), summary=False)
        self.assertEqual(res.main_config["run"]["seed"], ns["TRAINER_DEFAULT_CONFIG"]["run"]["seed"])
        for bad in (-1, 1.5, "1", True, 2 ** 32):
            with self.assertRaises(ValueError, msg=repr(bad)):
                S().updated({"train": {"seed": bad}})
        with self.assertRaises(ValueError):
            S().updated({"train": {"sed": 1}})

    def test_seed_reaches_the_trainer_config_and_does_not_change_its_fingerprint(self):
        ns = _ns()
        kit = ns["Toolkit"].from_namespace(ns)
        s = trs.settings_for(ns, {"holdout": None, "two_tf": False}, "/tmp/x")
        cfgs = {}
        for seed in (None, 7):
            st = s.updated({"train": {"seed": seed}})
            res = ns["run_main"](st.updated({"train": {"run_main_training": False}}), kit, dataset=trs._dataset(False, None), summary=False)
            cfgs[seed] = res.main_config
        self.assertEqual([cfgs[None]["run"]["seed"], cfgs[7]["run"]["seed"]], [42, 7])
        self.assertEqual(ns["config_fingerprint"](cfgs[None]), ns["config_fingerprint"](cfgs[7]))

    def test_same_seed_same_first_batches_different_seed_different(self):
        ns = _ns()
        kit = ns["Toolkit"].from_namespace(ns)
        s = trs.settings_for(ns, {"holdout": None, "two_tf": False}, "/tmp/x").updated({"train": {"run_main_training": False}})

        def first_batches(seed):
            res = ns["run_main"](s.updated({"train": {"seed": seed}}), kit, dataset=trs._dataset(False, None), summary=False)
            out = []
            for xb, yb in res.train_ds.take(3):
                x = next(iter(xb.values())) if isinstance(xb, dict) else xb
                out.append(np.concatenate([np.asarray(x).reshape(len(x), -1)[:, :4]]
                                          + [np.asarray(v).reshape(len(v), -1) for v in yb.values()], axis=1))
            return np.stack(out)

        a1, a2, b = first_batches(7), first_batches(7), first_batches(8)
        np.testing.assert_array_equal(a1, a2)
        self.assertFalse(np.array_equal(a1, b))

    def test_same_seed_same_epoch1_loss_different_seed_different(self):
        def loss(seed):
            cfg = _config(os.path.join(self.tmp, f"r{seed}_{id(object())}"), **{"run": {"seed": seed}})
            _, hist, _ = _train(cfg, 1)
            return hist.history["loss"][0]
        self.assertEqual(loss(3), loss(3))
        self.assertNotEqual(loss(3), loss(4))

    def test_unseeded_pre_pr_order_is_not_reproducible(self):
        """The behavior this PR changes ON PURPOSE: a shuffle built with seed=None before seed_everything (the old make_datasets order)
        gives a different batch order on every build, even when the run seed is the same."""
        ns = _ns()
        X = np.arange(512, dtype="float32").reshape(512, 1)
        y = {"y": X.copy()}

        def old_order():
            ds = ns["make_shuffled_dataset"](X, y, 32)           # built first, seed=None (old make_datasets)
            tf.keras.utils.set_random_seed(42)                   # build_training_system seeded AFTER the dataset
            return np.concatenate([b[1]["y"].numpy().ravel() for b in ds.take(4)])

        def new_order():
            seed_everything(42)
            ds = ns["make_shuffled_dataset"](X, y, 32, seed=42)
            return np.concatenate([b[1]["y"].numpy().ravel() for b in ds.take(4)])

        self.assertFalse(np.array_equal(old_order(), old_order()))
        np.testing.assert_array_equal(new_order(), new_order())


if __name__ == "__main__":
    unittest.main()
