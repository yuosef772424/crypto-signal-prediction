"""colab_runner.ipynb stays thin and works: structure checks, and its code cells executed offline (fake google.colab, a fake
GitHub API served from local files, a local bare repo as the remote). No Drive, no network.
    python -m pytest tests/test_colab_runner_notebook.py -q
"""
import io
import json
import os
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "colab_runner.ipynb")
TOKEN = "gh" + "p_" + "Qw7" * 12


def cells():
    with open(NB, encoding="utf-8") as fh:
        nb = json.load(fh)
    return nb["cells"]


def code_cells():
    return ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]


def test_notebook_is_small_and_constants_live_in_the_first_code_cell_only():
    cs = code_cells()
    assert len(cells()) <= 12 and len(cs) == 5
    assert os.path.getsize(NB) < 20000
    for name in ("REPO", "TOKEN_PATH", "QUEUE_BRANCH", "RESULTS_BRANCH", "WORKDIR"):
        assert cs[0].count(f"{name} =") == 1 and all(f"{name} =" not in c for c in cs[1:])
    assert 'RESULTS_BRANCH = "colab-results"' in cs[0] and 'QUEUE_BRANCH = "colab-queue"' in cs[0]
    assert "/content/drive/MyDrive/secrets/github_token.txt" in cs[0]
    for c in cs:
        compile(c, "<cell>", "exec")
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in cells() if c["cell_type"] == "code")


def test_notebook_never_prints_or_embeds_the_token():
    text = "\n".join(code_cells())
    assert "print(_tok" not in text and "print(token" not in text and "@github.com" not in text
    assert "x-access-token" not in text and "?token=" not in text and "access_token" not in text
    assert 'Authorization": "Bearer " + _tok' in text                       # header only


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_cells_run_end_to_end_against_a_fake_github(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import colab_runner as cr_real

    src = tmp_path / "src"
    src.mkdir()
    files = {"tools/colab_runner.py": open(os.path.join(ROOT, "tools", "colab_runner.py"), encoding="utf-8").read(),
             "tools/colab_smoke.py": open(os.path.join(ROOT, "tools", "colab_smoke.py"), encoding="utf-8").read(),
             "requirements-ci.txt": "# none\n"}
    for rel, content in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(content, encoding="utf-8")
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run([*g, "init", "-q", "-b", "master"], cwd=src, check=True)
    subprocess.run([*g, "add", "-A"], cwd=src, check=True)
    subprocess.run([*g, "commit", "-q", "-m", "x"], cwd=src, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True).stdout.strip()
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)], check=True)

    job = {"job_id": "nb-smoke-001", "repo_ref": sha, "entry": "tools/colab_smoke.py", "args": ["--sleep", "0"],
           "config": {"rows": 3, "seed": 1}, "max_minutes": 2, "needs_drive": False, "expected_outputs": ["smoke.csv"],
           "note": "notebook test"}
    token_file = tmp_path / "token.txt"
    token_file.write_text(TOKEN + "\n")
    seen = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        seen.append((url, dict(req.header_items())))
        assert TOKEN not in url
        if "/contents/job.json" in url:
            assert "ref=colab-queue" in url
            return FakeResponse(json.dumps(job).encode())
        if "/commits/colab-queue" in url:
            return FakeResponse(b"q" * 40 + b"\n")
        if "/contents/tools/colab_runner.py" in url:
            assert f"ref={sha}" in url
            return FakeResponse(files["tools/colab_runner.py"].encode())
        raise AssertionError("unexpected request " + url)

    drive_mod = types.SimpleNamespace(mount=lambda path: None)
    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(drive=drive_mod))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "colab_runner", raising=False)

    cs = code_cells()
    ns = {"__name__": "nb"}
    exec(cs[0], ns)
    ns.update(TOKEN_PATH=str(token_file), WORKDIR=str(tmp_path / "work"))
    exec(cs[1], ns)
    R = ns["R"]
    assert "_tok" not in ns and "_gh" not in ns                                   # the raw token does not linger as a global
    assert seen and all(h.get("Authorization") == "Bearer " + TOKEN for _, h in seen)
    real_runner = R.Runner
    monkeypatch.setattr(R, "github_url", lambda repo: str(bare))
    monkeypatch.setattr(R, "Runner", lambda *a, **k: real_runner(*a, pip=lambda cmd, **kw: 0, echo=lambda *x: None,
                                                                  drive_runs=str(tmp_path / "drive"), **k))
    exec(cs[2], ns)
    exec(cs[3], ns)
    exec(cs[4], ns)
    assert ns["result"]["status"] == "OK", ns["result"]
    out = subprocess.run(["git", "show", "colab-results:runs/nb-smoke-001/smoke.csv"], cwd=bare, capture_output=True, text=True)
    assert out.stdout.startswith("i,value")
    assert (tmp_path / "drive" / "nb-smoke-001" / "manifest.json").is_file()
    assert cr_real.sha256_file(os.path.join(ROOT, "tools", "colab_runner.py")) == cr_real.sha256_file(
        str(tmp_path / "work" / "bootstrap" / "colab_runner.py"))


@pytest.mark.parametrize("bad_ref", ["master", "abc", "z" * 40])
def test_bootstrap_cell_refuses_a_bad_repo_ref_before_downloading_code(tmp_path, monkeypatch, bad_ref):
    token_file = tmp_path / "token.txt"
    token_file.write_text(TOKEN + "\n")
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "/contents/job.json" in req.full_url:
            return FakeResponse(json.dumps({"repo_ref": bad_ref}).encode())
        if "/commits/" in req.full_url:
            return FakeResponse(b"q" * 40)
        raise AssertionError("runner code must not be requested for a bad ref")

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(drive=types.SimpleNamespace(mount=lambda p: None)))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cs = code_cells()
    ns = {"__name__": "nb"}
    exec(cs[0], ns)
    ns.update(TOKEN_PATH=str(token_file), WORKDIR=str(tmp_path / "work"))
    with pytest.raises(RuntimeError, match="40-hex"):
        exec(cs[1], ns)
    assert not any("colab_runner.py" in c for c in calls)
