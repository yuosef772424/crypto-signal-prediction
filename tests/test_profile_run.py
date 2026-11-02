"""tools/profile_run.py (hardening S03): the run profiler measures the real workflow.run steps without changing what they compute.
    python -m pytest tests/test_profile_run.py -q
Unit tests (no data): epoch summary, Profiler (time, RSS growth, nesting, errors), config strictness, input probe, markdown, the
training-system wrapper, the Keras timing callback. Integration tests on the synthetic fixture (tests/run_fixture.py): a profiled one-epoch
training equals the recorded golden fingerprint of the same scenario (tests/golden_run_settings.json), the deadline stops cleanly, and the CLI
writes a bundle that passes the Colab run-contract validation (tools/colab_runner.validate_outputs).
"""
import copy
import gzip
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "docs", "research", "audit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import colab_runner as cr  # noqa: E402
import profile_run as pr  # noqa: E402
import run_fixture as rf  # noqa: E402
import test_run_steps as trs  # noqa: E402

STEP_ORDER = ["apply_project_config", "apply_dataset_config", "make_splits", "retarget", "plan_model", "build_model",
              "make_training_config", "make_datasets", "train_model", "prepare_chicks"]


class Pure(unittest.TestCase):
    def test_epoch_summary_skips_the_first_step_of_epoch_zero_only(self):
        e0 = pr.summarize_epoch(0, 12.0, [9.0, 0.1, 0.1, 0.1], 1.0)
        self.assertAlmostEqual(e0["step_ms_mean"], 100.0)                       # tracing step excluded
        self.assertAlmostEqual(e0["steps_per_s"], 10.0)
        self.assertEqual((e0["steps"], e0["first_step_s"]), (4, 9.0))
        self.assertAlmostEqual(e0["other_s"], 12.0 - 9.3 - 1.0)
        e1 = pr.summarize_epoch(1, 1.0, [0.2, 0.1], 0.1)
        self.assertAlmostEqual(e1["step_ms_mean"], 150.0)                       # later epochs keep every step
        self.assertIsNone(pr.summarize_epoch(0, 1.0, [0.5], 0.0)["step_ms_mean"])    # one step in epoch 0: nothing steady to average

    def test_profiler_records_time_rss_nesting_and_errors(self):
        lines = []
        prof = pr.Profiler(sample_seconds=0.01, echo=lines.append)
        with prof.step("outer") as info:
            info["k"] = 1
            with prof.step("inner", kind="detail"):
                time.sleep(0.05)
            block = np.ones(40 * 2 ** 20, dtype="float64")           # 320 MB touched
            time.sleep(0.15)
            del block
        outer = prof.steps[-1]
        self.assertEqual([s["name"] for s in prof.steps], ["inner", "outer"])
        self.assertGreaterEqual(outer["seconds"], 0.19)
        self.assertGreater(outer["rss_peak_mb"] - outer["rss_start_mb"], 250)
        self.assertEqual(outer["info"], {"k": 1})
        self.assertGreaterEqual(outer["process_peak_mb"], outer["rss_peak_mb"] - 1)
        self.assertEqual(len(lines), 2)
        with self.assertRaises(ZeroDivisionError):
            with prof.step("boom"):
                1 / 0
        self.assertIn("ZeroDivisionError", prof.steps[-1]["error"])
        with self.assertRaises(ValueError):
            with prof.step("x", kind="other"):
                pass

    def test_input_probe(self):
        r = pr.input_probe([1, 2, 3, 4], 10)                                     # fewer batches than asked: stops at the end
        self.assertEqual(r["batches"], 4)
        self.assertIsNotNone(r["ms_per_batch"])
        self.assertIsNone(pr.input_probe([1], 5)["ms_per_batch"])                # one batch: no steady figure
        with self.assertRaises(ValueError):
            pr.input_probe([], 5)

    def test_config_is_strict(self):
        def cfg(**over):
            c = {"settings": {}, "reports": [], "input_batches": 5, "require_gpu": False}
            c.update(over)
            path = os.path.join(tempfile.mkdtemp(), "c.json")
            json.dump(c, open(path, "w"))
            return path
        self.assertEqual(pr.load_config(cfg())["input_batches"], 5)
        for bad in ({"extra": 1}, {"settings": {"train": {"run_dir": "/x"}}}, {"reports": "gap"}, {"input_batches": 1},
                    {"input_batches": True}, {"settings": []}, {"require_gpu": "yes"}):
            with self.assertRaises(SystemExit, msg=str(bad)):
                pr.load_config(cfg(**bad))
        c = json.load(open(cfg()))
        del c["reports"]
        p = os.path.join(tempfile.mkdtemp(), "m.json")
        json.dump(c, open(p, "w"))
        with self.assertRaises(SystemExit):
            pr.load_config(p)

    def test_require_gpu_exits_before_loading_when_tensorflow_sees_none(self):
        if pr.gpu_devices():
            self.skipTest("a GPU is visible here")
        self.assertEqual(pr.check_gpu(False), [])
        with self.assertRaises(SystemExit) as cm:
            pr.check_gpu(True)
        self.assertIn("no GPU", str(cm.exception))

    def test_training_system_wrapper_appends_the_callback_without_touching_the_cached_list(self):
        prof = pr.Profiler(echo=lambda s: None)
        cached = ["a", "b"]
        wrapped = pr.timed_training_system(lambda *a, **k: ("trainer", cached, 3), prof, "TIMING")
        trainer, cbs, epoch = wrapped(1, x=2)
        self.assertEqual((trainer, cbs, epoch), ("trainer", ["a", "b", "TIMING"], 3))
        self.assertEqual(cached, ["a", "b"])
        self.assertEqual([(s["name"], s["kind"]) for s in prof.steps], [("build_training_system", "detail")])

    def test_markdown_is_sorted_by_seconds_and_lists_status(self):
        def step(name, sec, kind="run"):
            return {"name": name, "kind": kind, "seconds": sec, "share_of_run": 0.5 if kind == "run" else None, "rss_start_mb": 1.0,
                    "rss_peak_mb": 2.0, "rss_end_mb": 1.5, "gpu_peak_mb": None, "info": {"n": 3}, "error": None}
        p = {"status": "partial", "stopped_before": "report:gap", "error": None, "wall_seconds": 3.0, "run_total_seconds": 2.0,
             "process_peak_rss_mb": 9.0, "top_by_seconds": ["b"], "top_by_rss_peak_mb": ["a"],
             "environment": {"python": "3", "tensorflow": "2", "tf_gpus": [], "cpu_count": 1, "mem_total_gb": 1.0},
             "steps": [step("a", 1.0), step("b", 5.0), step("c", 0.5, "probe")],
             "training": {"epochs": [], "steady_step_ms": None, "steady_steps_per_s": None, "input_ms_per_batch": None, "input_bound_ratio": None}}
        md = pr.render_markdown(p)
        self.assertIn("partial", md.splitlines()[0])
        self.assertLess(md.index("| b |"), md.index("| a |"))
        self.assertLess(md.index("| a |"), md.index("| c |"))
        self.assertIn("report:gap", md)

    def test_timing_callback_records_epochs_with_validation(self):
        import tensorflow as tf
        x = np.random.default_rng(0).random((64, 4), dtype="float32")
        y = x.sum(1, keepdims=True)
        m = tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(1)])
        m.compile("sgd", "mse")
        cb = pr.make_timing_callback()
        ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        m.fit(ds, validation_data=ds, epochs=2, verbose=0, callbacks=[cb])
        self.assertEqual([e["epoch"] for e in cb.epochs], [0, 1])
        for e in cb.epochs:
            self.assertEqual(e["steps"], 4)
            self.assertGreater(e["val_s"], 0)
            self.assertGreater(e["epoch_s"], e["step_s_sum"])
            self.assertGreater(e["other_s"], -1e-3)
        m.evaluate(ds, verbose=0)                                               # outside fit: no active epoch, must not raise
        self.assertEqual(len(cb.epochs), 2)


_DATASET = {}


def _dataset():
    if "ds" not in _DATASET:
        _DATASET["ds"] = rf.build_dataset(trs._nbload.load_pipeline(), two_tf=False, holdout="2025-03-06")
    return copy.deepcopy(_DATASET["ds"])


class ProfilesTheRealSteps(unittest.TestCase):
    def test_a_profiled_training_equals_the_golden_fingerprint(self):
        sc = trs.GOLDEN["scenarios"]["return_trained_holdout"]
        gold = copy.deepcopy(sc)
        scenario = gold.pop("scenario")
        work = tempfile.mkdtemp(prefix="profile_test_")
        ns = rf.run_namespace()
        settings = trs.settings_for(ns, scenario, os.path.join(work, "run"))
        cwd = os.getcwd()
        lines = []
        profile, res = pr.profile_run(settings, ns, dataset=_dataset(), reports=["signals"], input_batches=4, work_dir=work, echo=lines.append)
        self.assertEqual(os.getcwd(), cwd)
        got = trs._norm(trs.capture(ns, res), work)
        g_hist, n_hist = gold.pop("history"), got.pop("history")
        self.assertEqual(sorted(g_hist), sorted(n_hist))
        for k in g_hist:
            np.testing.assert_allclose(n_hist[k], g_hist[k], rtol=0.15, atol=0.05, err_msg=k)
        diffs = rf.close(got, gold)
        self.assertEqual(diffs, [], "\n".join(diffs[:25]))
        # the profile itself
        self.assertEqual(profile["status"], "ok")
        names = [s["name"] for s in profile["steps"]]
        for want in STEP_ORDER + ["build_training_system", "input_pipeline_only", "predict_val", "report:signals"]:
            self.assertIn(want, names)
        self.assertEqual([n for n in names if n in STEP_ORDER], STEP_ORDER)       # run_main's order
        self.assertNotIn("load_dataset", names)                                   # the dataset was passed in
        self.assertEqual(len(lines), len(profile["steps"]))
        run_steps = [s for s in profile["steps"] if s["kind"] == "run"]
        self.assertAlmostEqual(sum(s["share_of_run"] for s in run_steps), 1.0, places=6)
        self.assertEqual(len(profile["training"]["epochs"]), 1)
        self.assertEqual(profile["training"]["epochs"][0]["steps"], profile["steps"][names.index("make_datasets")]["info"]["train_rows"] // 32)
        self.assertIsNotNone(profile["input_pipeline"]["first_batch_s"])
        self.assertTrue(profile["steps"][names.index("report:signals")]["info"]["produced_output"])
        self.assertEqual(profile["settings"]["train"]["run_dir"], os.path.join(work, "run"))      # never the Drive default
        self.assertTrue([d for d in os.listdir(work) if d.startswith("run")])                      # checkpoints went to the temp work dir
        self.assertTrue(os.path.exists(os.path.join(work, "reports", "signals_val.csv.gz")))        # report files stay out of the cwd
        json.dumps(profile, default=str)

    def test_deadline_stops_before_the_next_step(self):
        ns = rf.run_namespace()
        work = tempfile.mkdtemp(prefix="profile_test_")
        profile, res = pr.profile_run(ns["RunSettings"](), ns, dataset=_dataset(), work_dir=work, deadline=time.time() - 1,
                                      echo=lambda s: None)
        self.assertEqual((profile["status"], profile["stopped_before"], profile["steps"], res),
                         ("partial", "apply_project_config", [], None))

    def test_unknown_report_raises_before_anything_runs(self):
        ns = rf.run_namespace()
        with self.assertRaises(ValueError):
            pr.profile_run(ns["RunSettings"](), ns, dataset=_dataset(), reports=["nope"], work_dir=tempfile.mkdtemp(), echo=lambda s: None)

    def test_a_failing_step_is_recorded_and_the_partial_profile_is_attached(self):
        ns = rf.run_namespace()
        ns["plan_model"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("planned failure"))
        with self.assertRaises(RuntimeError) as cm:
            pr.profile_run(ns["RunSettings"](), ns, dataset=_dataset(), work_dir=tempfile.mkdtemp(), echo=lambda s: None)
        prof = cm.exception.profile
        self.assertEqual(prof["status"], "failed")
        self.assertIn("planned failure", prof["error"])
        self.assertIn("planned failure", prof["steps"][-1]["error"])
        self.assertEqual(prof["steps"][-1]["name"], "plan_model")

    def test_cli_bundle_passes_the_colab_run_contract(self):
        work = tempfile.mkdtemp(prefix="profile_cli_")
        data = os.path.join(work, "fixture.pkl.gz")
        with gzip.open(data, "wb") as f:
            pickle.dump(_dataset(), f)
        cfg = {"settings": {"data": {"path": data}, "train": {"epochs": 1, "fit_verbose": 0}}, "reports": ["signals"], "input_batches": 4,
               "require_gpu": False}
        cfg_path, out = os.path.join(work, "cfg.json"), os.path.join(work, "out")
        json.dump(cfg, open(cfg_path, "w"))
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "profile_run.py"), "--config", cfg_path, "--out", out,
                            "--max-minutes", "10"], cwd=work, capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertEqual(sorted(os.listdir(out)), ["manifest.json", "profile.json", "profile.md"])
        profile = json.load(open(os.path.join(out, "profile.json")))
        self.assertEqual(profile["status"], "ok")
        self.assertIn("load_dataset", [s["name"] for s in profile["steps"]])          # loaded from settings.data.path
        self.assertEqual(os.listdir(work).count("verification"), 0)                  # nothing leaked into the cwd
        job = {"job_id": "profile-test", "repo_ref": cr.git_head(ROOT), "entry": "tools/profile_run.py", "args": [], "config": cfg,
               "max_minutes": 10, "needs_drive": False, "expected_outputs": ["profile.json", "profile.md"], "note": "test"}
        cr.validate_job(job)
        self.assertEqual(cr.validate_outputs(out, job, job["repo_ref"]), [])


if __name__ == "__main__":
    unittest.main()
