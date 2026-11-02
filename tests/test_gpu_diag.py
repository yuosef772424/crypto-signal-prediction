"""tools/gpu_diag.py (S03): the GPU diagnosis entry writes a run-contract bundle; try_fix=false never touches the environment.
    python -m pytest tests/test_gpu_diag.py -q
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import colab_runner as cr  # noqa: E402
import gpu_diag as gd  # noqa: E402


def _cfg(**over):
    c = {"try_fix": False, "tf_version": "2.21.0"}
    c.update(over)
    p = os.path.join(tempfile.mkdtemp(), "c.json")
    json.dump(c, open(p, "w"))
    return p


def test_config_is_strict():
    assert gd.load_config(_cfg())["tf_version"] == "2.21.0"
    for bad in ({"try_fix": "yes"}, {"tf_version": "latest"}, {"extra": 1}):
        with pytest.raises(SystemExit):
            gd.load_config(_cfg(**bad))


def test_bundle_without_fix_passes_the_run_contract():
    out = os.path.join(tempfile.mkdtemp(), "out")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gpu_diag.py"), "--config", _cfg(), "--out", out],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(os.path.join(out, "gpu_diag.json")))
    assert rep["verdict"]["gpu_after_fix"] is None and "fix" not in rep
    assert "tf 2." in rep["probe_before"]["output"]
    job = {"job_id": "gpu-diag-test", "repo_ref": cr.git_head(ROOT), "entry": "tools/gpu_diag.py", "args": [], "config": {"try_fix": False, "tf_version": "2.21.0"},
           "max_minutes": 30, "needs_drive": False, "expected_outputs": ["gpu_diag.json"], "note": "test"}
    cr.validate_job(job)
    assert cr.validate_outputs(out, job, job["repo_ref"]) == []
