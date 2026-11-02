#!/usr/bin/env python3
"""
PURPOSE:  Run profiler of a main run (hardening S03): executes the workflow/run.py step functions in main.ipynb order on given RunSettings and records, per step, wall time, RSS (start / sampled peak / end), GPU peak memory when a GPU is visible, plus the training epochs (step time, steps/s, first-step latency, validation time, other time between steps) through a Keras timing callback, the input pipeline alone (first-batch latency, ms/batch) and one timing per evaluation report. Output: profile.json + profile.md (steps sorted by time). Also a Colab run-contract entry (job `profile-001`).
TAGS:     profiler, profile, bottleneck, wall time, peak RSS, memory, GPU memory, tf.data, input pipeline, steps per second, timing callback, workflow.run steps, run_main, Colab job, run contract, S03
PITFALLS: `require_gpu` true makes the job exit before loading anything when TensorFlow sees no GPU (Colab's GPU runtime is not enough: requirements-ci.txt installs tensorflow-cpu, whose files win over a later tensorflow). `install_cuda_tensorflow` = V repairs that in the job (uninstall tensorflow + tensorflow-cpu, install tensorflow[and-cuda]==V, re-exec once; ~2 min; modifies the runtime's Python environment, so only on a disposable runtime). The steps are the real workflow.run functions called in run_main's order with the same arguments (profile_run only wraps them), so the run is the same one; the only change is an extra last callback on trainer.fit (kit.build_training_system is wrapped) and train.run_dir, which is ALWAYS redirected to a temp dir (never resume or write the real training folder or Drive; checkpoints and report files live outside --out, which must stay <= 300 KB per file). The training callback times batch begin -> end, and Keras 3 pulls the batch INSIDE that call, so input wait cannot be separated by a callback (measured on a deliberately slow tf.data pipeline: wait ~ 0, compute = fetch + step); the input pipeline is therefore also run ALONE after training (`input_pipeline_only`, so it cannot move the seeded batch order of the real run; page cache is warm by then) and `input_bound_ratio` = its ms/batch over the steady step ms (near 1 = input-bound). RSS peak is sampled every 50 ms from /proc (a spike shorter than that is missed; `process_peak_mb` is the OS high-water mark of the whole process so far). With the `signals` report, `signal_diagnostics` (tools/signal_diagnostics.py) tests on the val and test tables whether confidence / uncertainty are informative about the error (within-asset Spearman vs a permutation null); it describes the profile model (2 epochs), not a fully trained one. A report that fails (run_reports swallows it) is recorded with `produced_output` false and the tail of its output in `stdout_tail`. Step kinds: `run` = part of a main run (counted in the total and the shares), `detail` = nested inside a run step (build_training_system), `probe` = extra work the profiler adds (input_pipeline_only, predict_val). The deadline (85% of --max-minutes) is checked BEFORE each step only; a long step overshoots and the runner kills the job at max_minutes + 2, so each finished step is also echoed to the log as one line. Status `partial` (deadline) still exits 0 but is written in profile.json / profile.md. input_fingerprint in the manifest is the hash of an empty input list: the dataset is described in profile.json (`run_facts`). On Colab the runner installs requirements-ci.txt (tensorflow-cpu): check `environment.tf_gpus` in profile.json before reading the numbers as GPU numbers. Linux/macOS only (resource, /proc).

Usage:  python tools/profile_run.py --config cfg.json --out DIR [--max-minutes N]
cfg = {"settings": {"train": {"epochs": 2, "fit_verbose": 2}},   # RunSettings().updated(...); train.run_dir is forbidden (owned by the profiler)
       "reports": ["gap", "market_neutral"],                        # subset of workflow.run.REPORTS, [] = none
       "input_batches": 50,                                         # batches of the input-pipeline-only probe
       "require_gpu": true,                                          # true = exit at once (before loading data) when TensorFlow sees no GPU
       "install_cuda_tensorflow": "2.21.0"}                          # null, or the version: with require_gpu and a GPU in nvidia-smi but none in
                                                                     # TensorFlow, replace tensorflow + tensorflow-cpu by tensorflow[and-cuda]==V and re-exec
"""
import argparse
import contextlib
import dataclasses
import importlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import time

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
for _p in (TOOLS_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = 1
CONFIG_KEYS = ("settings", "reports", "input_batches", "require_gpu", "install_cuda_tensorflow")
#: set (to "1") in the re-executed process after the CUDA TensorFlow was installed, so the fix runs at most once
FIXED_ENV = "CSP_CUDA_TF_INSTALLED"
MAX_BYTES = 300 * 1024
SAMPLE_SECONDS = 0.05
#: no new step starts after this share of --max-minutes (the runner kills the job at max_minutes + 2 and a kill loses every output)
DEADLINE_SHARE = 0.85
#: package -> modules left out of its load (the self-tests / smoke tests a profile must not pay for)
PACKAGES = (("data", ("selftests",)), ("model", ("selftests",)), ("trainer", ("smoke_test", "kfold")), ("evaluation", ()), ("workflow", ()))


# ---------------------------------------------------------------------------------------------------------------- measuring
def rss_mb():
    """Resident set size of this process now, in MB (/proc/self/statm; the OS high-water mark where /proc is missing)."""
    try:
        with open("/proc/self/statm", encoding="ascii") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except OSError:
        return peak_rss_mb()


def peak_rss_mb():
    """High-water mark of the RSS of this process so far, in MB (never goes down)."""
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 2 ** 20 if sys.platform == "darwin" else ru / 1024


def gpu_devices():
    import tensorflow as tf
    return [d.name for d in tf.config.list_physical_devices("GPU")]


def _say(text):
    print(text, flush=True)


class Profiler:
    """Collects one record per measured step: ``with prof.step("name", kind="run") as info: ...`` (``info`` = a dict the body may fill
    with facts). A step that raises is recorded with its error and the exception propagates."""

    def __init__(self, gpu=False, sample_seconds=SAMPLE_SECONDS, echo=None):
        self.steps, self.gpu, self.sample_seconds, self.echo = [], gpu, sample_seconds, echo or _say

    def _gpu(self, reset):
        import tensorflow as tf
        if reset:
            tf.config.experimental.reset_memory_stats("GPU:0")
            return None
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2 ** 20

    @contextlib.contextmanager
    def step(self, name, kind="run"):
        if kind not in ("run", "detail", "probe"):
            raise ValueError(f"unknown step kind {kind!r}")
        rec = {"name": name, "kind": kind, "seconds": None, "rss_start_mb": rss_mb(), "rss_peak_mb": None, "rss_end_mb": None,
               "process_peak_mb": None, "gpu_peak_mb": None, "info": {}, "error": None}
        peak = [rec["rss_start_mb"]]
        stop = threading.Event()

        def sample():
            while not stop.wait(self.sample_seconds):
                peak[0] = max(peak[0], rss_mb())

        if self.gpu:
            self._gpu(reset=True)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        try:
            yield rec["info"]
        except BaseException as ex:
            rec["error"] = f"{type(ex).__name__}: {ex}"
            raise
        finally:
            rec["seconds"] = time.perf_counter() - t0
            stop.set()
            sampler.join()
            rec["rss_end_mb"] = rss_mb()
            rec["rss_peak_mb"] = max(peak[0], rec["rss_end_mb"])
            rec["process_peak_mb"] = peak_rss_mb()
            if self.gpu:
                rec["gpu_peak_mb"] = self._gpu(reset=False)
            self.steps.append(rec)
            self.echo(f"[profile] {name:<28} {kind:<6} {rec['seconds']:9.2f}s  rss {rec['rss_start_mb']:8.0f} -> peak {rec['rss_peak_mb']:8.0f} MB"
                      + (f"  ERROR {rec['error']}" if rec["error"] else ""))


class Tee:
    """Writes to ``stream`` and keeps the last ``keep`` characters: run_reports swallows a failing report's exception (it only prints it), and the
    runner's log keeps just the last 200 KB, so the reason of a failed report would be lost; the profile stores this tail instead."""

    def __init__(self, stream, keep=4000):
        self.stream, self.keep, self.tail = stream, keep, ""

    def write(self, text):
        self.tail = (self.tail + text)[-self.keep:]
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


def make_timing_callback():
    """A Keras callback (built lazily: needs TensorFlow) that records, per epoch, the batch begin -> end times, the validation time and the
    rest of the epoch. Keras 3 fetches the batch inside the step call, so ``step`` times include the input pipeline (see PITFALLS)."""
    import tensorflow as tf

    class TimingCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.epochs, self._cur, self._t_batch, self._t_val = [], None, 0.0, 0.0

        def on_epoch_begin(self, epoch, logs=None):
            self._cur = {"epoch": int(epoch), "t0": time.perf_counter(), "step_s": [], "val_s": 0.0}

        def on_train_batch_begin(self, batch, logs=None):
            self._t_batch = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            self._cur["step_s"].append(time.perf_counter() - self._t_batch)

        def on_test_begin(self, logs=None):
            self._t_val = time.perf_counter()

        def on_test_end(self, logs=None):
            if self._cur is not None:                      # fit's validation only (evaluate() outside fit has no active epoch)
                self._cur["val_s"] += time.perf_counter() - self._t_val

        def on_epoch_end(self, epoch, logs=None):
            cur, self._cur = self._cur, None
            self.epochs.append(summarize_epoch(cur["epoch"], time.perf_counter() - cur["t0"], cur["step_s"], cur["val_s"]))

    return TimingCallback()


def summarize_epoch(epoch, epoch_s, step_s, val_s):
    """One epoch record from raw times (pure; unit-tested). ``steady`` statistics skip the first step of epoch 0 (graph tracing)."""
    steady = step_s[1:] if epoch == 0 else step_s
    total = sum(step_s)
    mean_ms = 1000 * sum(steady) / len(steady) if steady else None
    return {"epoch": int(epoch), "epoch_s": epoch_s, "steps": len(step_s), "step_s_sum": total, "step_ms_mean": mean_ms,
            "steps_per_s": (1000 / mean_ms) if mean_ms else None, "first_step_s": step_s[0] if step_s else None,
            "val_s": val_s, "other_s": epoch_s - total - val_s}


def timed_training_system(build, prof, timing):
    """``kit.build_training_system`` with the timing callback appended last (so the epoch time also covers the other callbacks' epoch-end
    work, e.g. checkpoint saves) and the build itself recorded as a `detail` step."""
    def wrapped(*args, **kwargs):
        with prof.step("build_training_system", kind="detail"):
            trainer, callbacks, initial_epoch = build(*args, **kwargs)
        return trainer, list(callbacks) + [timing], initial_epoch
    return wrapped


def input_probe(dataset, batches):
    """Iterate ``dataset`` alone (no model) for up to ``batches`` batches: first-batch latency and the mean ms/batch of the rest."""
    it = iter(dataset)
    times, t = [], time.perf_counter()
    for _ in range(batches):
        try:
            next(it)
        except StopIteration:
            break
        now = time.perf_counter()
        times.append(now - t)
        t = now
    if not times:
        raise ValueError("input probe: the dataset produced no batch")
    rest = times[1:]
    return {"batches": len(times), "first_batch_s": times[0], "ms_per_batch": (1000 * sum(rest) / len(rest)) if rest else None}


# ---------------------------------------------------------------------------------------------------------------- the run
class _OutOfTime(Exception):
    pass


def build_namespace(quiet=True):
    """The shared namespace of a main run (pipeline, model, trainer, chicks evaluation, workflow), loaded like main.ipynb's %run cells
    without the self-tests / smoke tests. ``CSP_DRIVE_ROOT`` (set by the Colab runner for needs_drive jobs) becomes the pipeline's Drive root."""
    ns = {"__name__": "profile_run", "display": print}
    import builtins
    builtins.display = print
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        for pkg, exclude in PACKAGES:
            importlib.import_module(pkg).load_into(ns, exclude=exclude)
    root = os.environ.get("CSP_DRIVE_ROOT")
    if root:
        from pathlib import Path
        ns["_DRIVE_ROOT"]["root"] = Path(root)
    return ns


def profile_run(settings, ns, *, dataset=None, reports=(), input_batches=50, work_dir, deadline=None, echo=None):
    """Run the main steps on ``settings`` and measure them. ``dataset`` = an already loaded dataset (the load_dataset step is then not run).
    Returns ``(profile_dict, RunResult or None)``; the result is None when the run stopped early (deadline / error, see profile["status"])."""
    reports = list(reports)
    unknown = sorted(set(reports) - set(ns["REPORTS"]))
    if unknown:
        raise ValueError(f"unknown report(s) {unknown}; known: {list(ns['REPORTS'])}")
    gpus = gpu_devices()
    prof = Profiler(gpu=bool(gpus), echo=echo)
    timing = make_timing_callback()
    kit0 = ns["Toolkit"].from_namespace(ns)
    kit = dataclasses.replace(kit0, build_training_system=timed_training_system(kit0.build_training_system, prof, timing))
    settings = settings.updated({"train": {"run_dir": os.path.join(work_dir, "run")}})
    t_start, state, result, probes = time.time(), {"status": "ok", "stopped_before": None, "error": None}, None, {}

    def due(name):
        if deadline is not None and time.time() > deadline:
            raise _OutOfTime(name)

    def run(name, fn, **kw):
        due(name)
        with prof.step(name) as info:
            out = fn()
            for k, v in kw.items():
                info[k] = v(out) if callable(v) else v
        return out

    try:
        run("apply_project_config", lambda: ns["apply_project_config"](settings, kit))
        if dataset is None:
            dataset = run("load_dataset", lambda: ns["load_dataset"](settings, kit))
        info = run("apply_dataset_config", lambda: ns["apply_dataset_config"](settings, dataset, kit))
        train, val, test = run("make_splits", lambda: ns["make_splits"](settings, dataset, info, kit))
        train, val, test = run("retarget", lambda: ns["retarget"](settings, train, val, test, info))
        plan = run("plan_model", lambda: ns["plan_model"](settings, dataset, info, kit))
        model_builder, model = run("build_model", lambda: ns["build_model"](plan, kit, summary=False), params=lambda o: int(o[1].count_params()))
        main_config = run("make_training_config", lambda: ns["make_training_config"](settings, plan, info, val, kit))
        train_ds, val_ds, sample_batch = run("make_datasets", lambda: ns["make_datasets"](main_config, train, val, info),
                                             batch_size=main_config["run"]["batch_size"], train_rows=len(train["last_candles"]))
        model, trainer, callbacks, history = run(
            "train_model", lambda: ns["train_model"](settings, kit, model=model, model_builder=model_builder, main_config=main_config,
                                                      train_ds=train_ds, val_ds=val_ds, sample_batch=sample_batch),
            epochs=main_config["run"]["epochs"])
        chicks = run("prepare_chicks", lambda: ns["prepare_chicks"](settings, plan, info, test, kit))
        result = ns["RunResult"](dataset=dataset, info=info, train=train, val=val, test=test, plan=plan, model_builder=model_builder,
                                 model=model, main_config=main_config, train_ds=train_ds, val_ds=val_ds, trainer=trainer,
                                 callbacks=callbacks, history=history, chicks=chicks)
        # probes AFTER training: they must not move the seeded batch order of the real run
        due("input_pipeline_only")
        with prof.step("input_pipeline_only", kind="probe") as i:
            probes["input"] = input_probe(train_ds, input_batches)
            i.update(probes["input"])
        due("predict_val")
        with prof.step("predict_val", kind="probe") as i:
            model.predict(val_ds, verbose=0)
            i["batches"] = int(val_ds.cardinality().numpy()) if hasattr(val_ds, "cardinality") else None
        cwd = os.getcwd()
        os.makedirs(os.path.join(work_dir, "reports"), exist_ok=True)
        os.chdir(os.path.join(work_dir, "reports"))                          # the reports write relative paths
        try:
            for key in [k for k in ns["REPORTS"] if k in reports]:
                due(f"report:{key}")
                with prof.step(f"report:{key}") as i:
                    tee = Tee(sys.stdout)
                    with contextlib.redirect_stdout(tee):
                        out = ns["run_reports"](settings, [key], kit, model=model, train=train, val=val, test=test, dataset=dataset,
                                                info=info, plan=plan, chicks=chicks)
                    i["produced_output"] = key in out
                    if key not in out:
                        i["stdout_tail"] = tee.tail[-3000:]                       # why it failed (run_reports only prints it)
            if "signals" in reports:                                             # are confidence / uncertainty informative or noise?
                due("signal_diagnostics")
                with prof.step("signal_diagnostics", kind="probe") as i:
                    import pandas as pd
                    import signal_diagnostics
                    probes["signal_diagnostics"] = {}
                    for split in ("val", "test"):
                        path = os.path.join(work_dir, "reports", f"signals_{split}.csv.gz")
                        if os.path.exists(path):
                            probes["signal_diagnostics"][split] = signal_diagnostics.check_all(pd.read_csv(path), n_perm=50)
                    i["splits"] = ",".join(probes["signal_diagnostics"])
        finally:
            os.chdir(cwd)
    except _OutOfTime as ex:
        state.update(status="partial", stopped_before=str(ex))
    except BaseException as ex:
        state.update(status="failed", error=f"{type(ex).__name__}: {ex}")
        profile = build_profile(prof, timing, probes, settings, state, time.time() - t_start, result, gpus, reports, input_batches)
        ex.profile = profile                                                  # the caller can still write the partial profile
        raise
    return build_profile(prof, timing, probes, settings, state, time.time() - t_start, result, gpus, reports, input_batches), result


def environment(gpus):
    import numpy
    import pandas
    import tensorflow as tf
    mem = None
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            mem = round(int(fh.readline().split()[1]) / 2 ** 20, 2)
    except (OSError, ValueError):
        pass
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(), "mem_total_gb": mem,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "tensorflow": tf.__version__, "tf_gpus": gpus}


def build_profile(prof, timing, probes, settings, state, wall_s, result, gpus, reports, input_batches):
    steps = [dict(s) for s in prof.steps]
    run_total = sum(s["seconds"] for s in steps if s["kind"] == "run")
    for s in steps:
        s["share_of_run"] = (s["seconds"] / run_total) if s["kind"] == "run" and run_total else None
    runs = [s for s in steps if s["kind"] == "run"]
    epochs = list(timing.epochs)
    steady = [e["step_ms_mean"] for e in epochs if e["step_ms_mean"]]
    steady_ms = sum(steady) / len(steady) if steady else None
    inp = probes.get("input")
    training = {"epochs": epochs, "steady_step_ms": steady_ms, "steady_steps_per_s": (1000 / steady_ms) if steady_ms else None,
                "input_ms_per_batch": inp["ms_per_batch"] if inp else None,
                "input_bound_ratio": (inp["ms_per_batch"] / steady_ms) if inp and inp["ms_per_batch"] and steady_ms else None}
    facts = None
    if result is not None:
        facts = {"model_tfs": list(result.info.model_tfs), "seq_len": result.plan.model_seq_len, "n_features": result.plan.model_n_features,
                 "train_rows": len(result.train["last_candles"]), "val_rows": len(result.val["last_candles"]),
                 "test_rows": int(sum(len(v["last_candles"]) for v in result.test.values())), "test_assets": len(result.test),
                 "price_targets": list(result.plan.price_targets)}
    return {"schema": SCHEMA, "status": state["status"], "stopped_before": state["stopped_before"], "error": state["error"],
            "wall_seconds": wall_s, "run_total_seconds": run_total, "process_peak_rss_mb": peak_rss_mb(),
            "top_by_seconds": [s["name"] for s in sorted(runs, key=lambda s: -s["seconds"])[:3]],
            "top_by_rss_peak_mb": [s["name"] for s in sorted(runs, key=lambda s: -s["rss_peak_mb"])[:3]],
            "environment": environment(gpus), "settings": settings.to_dict(), "reports": reports, "input_batches": input_batches,
            "run_facts": facts, "steps": steps, "training": training, "input_pipeline": inp,
            "signal_diagnostics": probes.get("signal_diagnostics")}


# ---------------------------------------------------------------------------------------------------------------- output
def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def _cell(v):
    """A table-safe, short image of a note value (stdout tails have newlines and pipes)."""
    t = str(v).replace("|", "/").replace("\n", " / ")
    return t if len(t) <= 400 else "..." + t[-400:]


def render_markdown(p):
    """Sorted markdown table of a profile dict (steps by seconds, descending) + the training epochs."""
    out = [f"# Run profile ({p['status']})", ""]
    if p["status"] != "ok":
        out += [f"**status {p['status']}**: stopped before `{p['stopped_before']}`; error: {p['error']}", ""]
    env = p["environment"]
    out += [f"wall {p['wall_seconds']:.1f}s, run steps total {p['run_total_seconds']:.1f}s, process peak RSS {p['process_peak_rss_mb']:.0f} MB; "
            f"python {env['python']}, tensorflow {env['tensorflow']}, GPUs {env['tf_gpus'] or 'none'}, CPUs {env['cpu_count']}, RAM {env['mem_total_gb']} GB",
            f"top by seconds: {', '.join(p['top_by_seconds'])}; top by RSS peak: {', '.join(p['top_by_rss_peak_mb'])}", "",
            "## Steps (sorted by seconds; kind run = counted in the share, detail = inside a run step, probe = added by the profiler)", "",
            "| step | kind | seconds | share of run | RSS start MB | RSS peak MB | RSS end MB | GPU peak MB | notes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for s in sorted(p["steps"], key=lambda s: -s["seconds"]):
        notes = "; ".join(f"{k}={_f(v, 3) if isinstance(v, float) else _cell(v)}" for k, v in s["info"].items())
        if s["error"]:
            notes = f"ERROR {s['error']}; {notes}"
        share = "-" if s["share_of_run"] is None else f"{100 * s['share_of_run']:.1f}%"
        out.append(f"| {s['name']} | {s['kind']} | {s['seconds']:.2f} | {share} | {s['rss_start_mb']:.0f} | {s['rss_peak_mb']:.0f} | "
                   f"{s['rss_end_mb']:.0f} | {_f(s['gpu_peak_mb'], 0)} | {notes} |")
    t = p["training"]
    out += ["", "## Training", "", "| epoch | epoch s | steps | step ms (steady) | steps/s | first step s | validation s | other s |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for e in t["epochs"]:
        out.append(f"| {e['epoch']} | {e['epoch_s']:.2f} | {e['steps']} | {_f(e['step_ms_mean'])} | {_f(e['steps_per_s'])} | {_f(e['first_step_s'], 3)} | "
                   f"{e['val_s']:.2f} | {e['other_s']:.2f} |")
    diag = p.get("signal_diagnostics")
    if diag:
        import signal_diagnostics
        out += ["", "## Confidence / uncertainty vs the error (within-asset Spearman, permutation null; see tools/signal_diagnostics.py)", ""]
        for split, r in diag.items():
            out += [f"- **{split}**"] + [f"  - {line}" for line in signal_diagnostics.verdict(r)]
    out += ["", f"steady step {_f(t['steady_step_ms'])} ms ({_f(t['steady_steps_per_s'])} steps/s); input pipeline alone {_f(t['input_ms_per_batch'])} ms/batch; "
            f"input_bound_ratio {_f(t['input_bound_ratio'], 3)} (near 1 = input-bound). `other s` = gaps between steps + epoch-end callbacks (checkpoint saves...).", ""]
    return "\n".join(out)


def write_outputs(profile, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "profile.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(profile, fh, indent=1, sort_keys=True, default=str)
        fh.write("\n")
    with open(os.path.join(out_dir, "profile.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(profile))
    for f in os.listdir(out_dir):
        if os.path.getsize(os.path.join(out_dir, f)) > MAX_BYTES:
            raise SystemExit(f"output {f} exceeds 300 KB")


# ---------------------------------------------------------------------------------------------------------------- CLI / job
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    missing, unknown = sorted(set(CONFIG_KEYS) - set(cfg)), sorted(set(cfg) - set(CONFIG_KEYS))
    if missing or unknown:
        raise SystemExit(f"config keys: missing {missing}, unknown {unknown}; allowed {list(CONFIG_KEYS)}")
    if not isinstance(cfg["settings"], dict):
        raise SystemExit("config 'settings' must be a {section: {setting: value}} dict (RunSettings.updated)")
    if "run_dir" in cfg["settings"].get("train", {}):
        raise SystemExit("config settings.train.run_dir is not allowed: the profiler always trains in a temp dir (never resume / write Drive)")
    if not isinstance(cfg["reports"], list) or not all(isinstance(r, str) for r in cfg["reports"]):
        raise SystemExit("config 'reports' must be a list of report names ([] = none)")
    if not isinstance(cfg["input_batches"], int) or isinstance(cfg["input_batches"], bool) or cfg["input_batches"] < 2:
        raise SystemExit("config 'input_batches' must be an integer >= 2")
    if not isinstance(cfg["require_gpu"], bool):
        raise SystemExit("config 'require_gpu' must be true or false")
    v = cfg["install_cuda_tensorflow"]
    if v is not None and not (isinstance(v, str) and v.replace(".", "").isdigit()):
        raise SystemExit("config 'install_cuda_tensorflow' must be null or a version like '2.21.0'")
    if v is not None and not cfg["require_gpu"]:
        raise SystemExit("config 'install_cuda_tensorflow' needs require_gpu true")
    return cfg


def ensure_gpu(required, fix_version, *, probe=None, run=None, smi=None, execv=os.execv, environ=os.environ):
    """Before TensorFlow is imported in this process: with ``required``, probe it in a subprocess; if it sees no GPU, nvidia-smi names one and
    ``fix_version`` is set, replace tensorflow + tensorflow-cpu (requirements-ci.txt installs the CPU build, whose files win) by one
    ``tensorflow[and-cuda]==fix_version`` and re-exec this command (at most once: FIXED_ENV). Measured on a Colab T4 (job gpu-diag-001):
    that sequence turns is_cuda_build False -> True and the GPU becomes visible (~30 s uninstall + ~75 s install). Otherwise returns and
    ``check_gpu`` reports / exits."""
    if not required:
        return
    import gpu_diag
    probe, run, smi = probe or gpu_diag.probe, run or gpu_diag.run, smi or _nvidia_smi
    if probe("start")["sees_gpu"] or fix_version is None:
        return
    if environ.get(FIXED_ENV):
        print("[profile] the CUDA TensorFlow was installed but still sees no GPU", flush=True)
        return
    named = smi()
    if named == "none":
        raise SystemExit("require_gpu: nvidia-smi sees no GPU: this runtime has none (Runtime > Change runtime type > GPU)")
    print(f"[profile] nvidia-smi: {named}; TensorFlow sees no GPU: installing tensorflow[and-cuda]=={fix_version}", flush=True)
    base = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    for cmd, timeout in ((base + ["uninstall", "-y", "tensorflow", "tensorflow-cpu"], 600),
                         (base + ["install", "-q", f"tensorflow[and-cuda]=={fix_version}"], 1800)):
        r = run(cmd, timeout)
        print(f"[profile] {r['cmd']} -> exit {r['exit']} in {r['seconds']}s", flush=True)
        if r["exit"] != 0:
            raise SystemExit(f"require_gpu: '{r['cmd']}' failed:\n{r['output'][-2000:]}")
    environ[FIXED_ENV] = "1"
    execv(sys.executable, [sys.executable] + sys.argv)


def check_gpu(required):
    """Print what TensorFlow sees; with ``required`` exit at once when it sees no GPU (a CPU-only run of a long job is wasted time)."""
    gpus = gpu_devices()
    print(f"[profile] TensorFlow GPUs: {gpus or 'none'}; nvidia-smi: {_nvidia_smi()}", flush=True)
    if required and not gpus:
        raise SystemExit("require_gpu is true but TensorFlow sees no GPU (see the nvidia-smi line above: if it names a GPU, the installed "
                         "tensorflow build has no CUDA support, e.g. tensorflow-cpu from requirements-ci.txt)")
    return gpus


def _nvidia_smi():
    import colab_runner
    return colab_runner.gpu_info()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run profiler of a main run (S03)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    ns_args = ap.parse_args(argv)
    cfg = load_config(ns_args.config)
    ensure_gpu(cfg["require_gpu"], cfg["install_cuda_tensorflow"])
    check_gpu(cfg["require_gpu"])
    out_dir = os.path.abspath(ns_args.out)
    t0 = time.time()
    import matplotlib
    matplotlib.use("Agg")
    ns = build_namespace()
    settings = ns["RunSettings"]().updated(cfg["settings"])
    work_dir = tempfile.mkdtemp(prefix="profile_run_")
    exit_code = 0
    try:
        profile, _ = profile_run(settings, ns, reports=cfg["reports"], input_batches=cfg["input_batches"], work_dir=work_dir,
                                 deadline=t0 + ns_args.max_minutes * 60 * DEADLINE_SHARE)
    except BaseException as ex:
        profile = getattr(ex, "profile", None)
        if profile is None:
            raise
        exit_code = 1
        print(f"profile run FAILED: {profile['error']}", file=sys.stderr)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    write_outputs(profile, out_dir)
    import colab_runner
    colab_runner.write_manifest(out_dir, config=cfg, row_counts={"profile.json": len(profile["steps"])},
                                runtime_seconds=time.time() - t0, n_configurations=1)
    print(f"profile {profile['status']}: {len(profile['steps'])} steps, run total {profile['run_total_seconds']:.1f}s, "
          f"top by seconds {profile['top_by_seconds']}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
