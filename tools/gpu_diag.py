#!/usr/bin/env python3
"""
PURPOSE:  Colab run-contract entry that explains why TensorFlow sees no GPU on a GPU runtime (nvidia-smi names a GPU, tf.config.list_physical_devices('GPU') is empty): lists the tensorflow / keras / nvidia-* distributions pip knows, nvidia-smi, the CUDA-related environment, and probes TensorFlow in a subprocess with C++ logging ON (the real dlopen / driver error). With try_fix it then uninstalls tensorflow + tensorflow-cpu, installs one clean `tensorflow[and-cuda]==<tf_version>` and probes again, so the next job knows which pip recipe works.
TAGS:     gpu, tensorflow-cpu, tensorflow[and-cuda], colab, diagnosis, cuda, nvidia-smi, pip, run contract, S03
PITFALLS: try_fix MODIFIES the Python environment of the machine it runs on (uninstalls tensorflow): use it only on a disposable Colab runtime. Every probe runs in a fresh subprocess (an imported TensorFlow cannot be unloaded). Outputs are truncated to 20 KB per command. Meant to be queued with the same pip list as the job it explains (the runner installs requirements-ci.txt first), so the first probe sees the same environment.

Usage:  python tools/gpu_diag.py --config cfg.json --out DIR [--max-minutes N]
cfg = {"try_fix": true, "tf_version": "2.21.0"}
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner  # noqa: E402

CONFIG_KEYS = ("try_fix", "tf_version")
MAX_CHARS = 20000
PROBE = ("import tensorflow as tf; print('tf', tf.__version__); "
         "print('build', {k: v for k, v in tf.sysconfig.get_build_info().items() if 'cuda' in k or 'cudnn' in k}); "
         "print('GPUS', tf.config.list_physical_devices('GPU'))")


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    missing, unknown = sorted(set(CONFIG_KEYS) - set(cfg)), sorted(set(cfg) - set(CONFIG_KEYS))
    if missing or unknown:
        raise SystemExit(f"config keys: missing {missing}, unknown {unknown}; allowed {list(CONFIG_KEYS)}")
    if not isinstance(cfg["try_fix"], bool):
        raise SystemExit("config 'try_fix' must be true or false")
    if not isinstance(cfg["tf_version"], str) or not cfg["tf_version"].replace(".", "").isdigit():
        raise SystemExit("config 'tf_version' must look like '2.21.0'")
    return cfg


def run(cmd, timeout, env=None):
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        text, code = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as ex:
        text, code = f"TIMEOUT after {timeout}s\n{ex.stdout or ''}{ex.stderr or ''}", 124
    except OSError as ex:
        text, code = f"{type(ex).__name__}: {ex}", 127
    return {"cmd": " ".join(cmd), "exit": code, "seconds": round(time.time() - t0, 1),
            "output": text if len(text) <= MAX_CHARS else text[:MAX_CHARS // 2] + "\n...[cut]...\n" + text[-MAX_CHARS // 2:]}


def probe(label):
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="0")
    r = run([sys.executable, "-c", PROBE], 600, env)
    r["label"] = label
    r["sees_gpu"] = "GPUS [PhysicalDevice" in r["output"]
    return r


def pip_view():
    r = run([sys.executable, "-m", "pip", "list", "--format=freeze", "--disable-pip-version-check"], 300)
    keep = [ln for ln in r["output"].splitlines() if ln.lower().startswith(("tensorflow", "keras", "nvidia-", "torch", "jax", "tf-"))]
    r["output"] = "\n".join(keep)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=30)
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    t0 = time.time()
    os.makedirs(ns.out, exist_ok=True)
    report = {"python": sys.version, "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDA", "LD_", "TF_", "NVIDIA", "XLA"))},
              "nvidia_smi": run(["nvidia-smi"], 60), "pip_before": pip_view(),
              "pip_show": run([sys.executable, "-m", "pip", "show", "tensorflow", "tensorflow-cpu", "--disable-pip-version-check"], 120),
              "probe_before": probe("as installed by the job")}
    if cfg["try_fix"]:
        base = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
        report["fix"] = [run(base + ["uninstall", "-y", "tensorflow", "tensorflow-cpu"], 600),
                         run(base + ["install", "-q", f"tensorflow[and-cuda]=={cfg['tf_version']}"], 1800)]
        report["pip_after"] = pip_view()
        report["probe_after"] = probe("after clean tensorflow[and-cuda]")
    report["verdict"] = {"gpu_before": report["probe_before"]["sees_gpu"],
                         "gpu_after_fix": report["probe_after"]["sees_gpu"] if cfg["try_fix"] else None}
    with open(os.path.join(ns.out, "gpu_diag.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for f in os.listdir(ns.out):
        if os.path.getsize(os.path.join(ns.out, f)) > 300 * 1024:
            raise SystemExit(f"output {f} exceeds 300 KB")
    colab_runner.write_manifest(ns.out, config=cfg, row_counts={"gpu_diag.json": 1}, runtime_seconds=time.time() - t0, n_configurations=1)
    print(f"gpu diag: {report['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
