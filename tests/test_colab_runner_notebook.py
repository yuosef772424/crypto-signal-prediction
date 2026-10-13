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
    assert 'TOKEN_PATH = "/content/drive/MyDrive/crypto/secrets/github_token.txt"' in cs[0]
    for c in cs:
        compile(c, "<cell>", "exec")
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in cells() if c["cell_type"] == "code")


def test_notebook_never_prints_or_embeds_the_token():
    text = "\n".join(code_cells())
    assert "print(_tok" not in text and "print(token" not in text and "@github.com" not in text
    assert "x-access-token" not in text and "?token=" not in text and "access_token" not in text
    assert 'Authorization": "Bearer " + ' in text                           # header only
    assert "getpass" in text and "print(_new" not in text and "print(_tok" not in text


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
        if url.endswith("/repos/yuosef772424/crypto-signal-prediction"):          # the one validation GET of the token
            return FakeResponse(b"{}")
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
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not ask: the file is valid")))
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
        if req.full_url.endswith("/repos/yuosef772424/crypto-signal-prediction"):
            return FakeResponse(b"{}")                                              # the token validation GET
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


# ------------------------------------------------------------------------- token file: first run, valid file, bad token
GOOD, OLD = TOKEN, "gh" + "p_" + "Old9" * 9
REPO_URL_END = "/repos/yuosef772424/crypto-signal-prediction"


class Api:
    """Fake GitHub API for the bootstrap cell: a token in `valid` is accepted, a token in `forbidden` gets 404, others 401;
    `down` makes every call fail with a server error."""

    def __init__(self, valid, forbidden=(), down=False):
        self.valid, self.forbidden, self.down, self.calls = set(valid), set(forbidden), down, []

    def __call__(self, req, timeout=None):
        import urllib.error
        url = req.full_url
        auth = dict(req.header_items()).get("Authorization", "")
        tok = auth.replace("Bearer ", "")
        self.calls.append((url, tok))
        assert tok not in url
        if self.down:
            raise urllib.error.HTTPError(url, 503, "down", {}, None)
        if tok in self.forbidden:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if tok not in self.valid:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        if url.endswith(REPO_URL_END):
            return FakeResponse(b"{}")
        if "/contents/job.json" in url:
            return FakeResponse(json.dumps({"repo_ref": "master"}).encode())      # stops the cell right after the token step
        if "/commits/" in url:
            return FakeResponse(b"q" * 40)
        raise AssertionError("unexpected request " + url)


def run_bootstrap(tmp_path, monkeypatch, api, prompts, token_file):
    asked = []

    def fake_getpass(prompt="", stream=None):
        asked.append(prompt)
        return prompts[len(asked) - 1]

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.colab", types.SimpleNamespace(drive=types.SimpleNamespace(mount=lambda p: None)))
    monkeypatch.setattr("urllib.request.urlopen", api)
    monkeypatch.setattr("getpass.getpass", fake_getpass)
    cs = code_cells()
    ns = {"__name__": "nb"}
    exec(cs[0], ns)
    ns.update(TOKEN_PATH=str(token_file), WORKDIR=str(tmp_path / "work"))
    try:
        exec(cs[1], ns)
    except RuntimeError as e:
        return asked, e
    return asked, None


def test_default_token_path_is_the_crypto_secrets_folder():
    ns = {}
    exec(code_cells()[0], ns)
    assert ns["TOKEN_PATH"] == "/content/drive/MyDrive/crypto/secrets/github_token.txt"
    assert "MyDrive/secrets/" not in "\n".join(code_cells())
    src = open(os.path.join(ROOT, "tools", "colab_runner.py"), encoding="utf-8").read()
    assert "MyDrive/secrets" not in src


@pytest.mark.parametrize("existing", [None, "", "\n", "  \n"])
def test_first_run_asks_validates_then_writes_the_file(tmp_path, monkeypatch, capsys, existing):
    token_file = tmp_path / "MyDrive" / "crypto" / "secrets" / "github_token.txt"
    if existing is not None:
        token_file.parent.mkdir(parents=True)
        token_file.write_text(existing)
    api = Api({GOOD})
    asked, err = run_bootstrap(tmp_path, monkeypatch, api, [GOOD], token_file)
    assert len(asked) == 1 and GOOD not in asked[0]
    assert token_file.read_text() == GOOD + "\n"                                  # one line
    if os.name == "posix":
        assert (token_file.stat().st_mode & 0o777) == 0o600
        assert (token_file.parent.stat().st_mode & 0o200)
    repo_calls = [c for c in api.calls if c[0].endswith(REPO_URL_END)]
    assert len(repo_calls) == 1 and repo_calls[0][1] == GOOD                      # one authenticated GET to validate
    out = capsys.readouterr()
    assert GOOD not in out.out + out.err and GOOD not in (str(err) if err else "")


def test_valid_file_is_read_silently(tmp_path, monkeypatch, capsys):
    token_file = tmp_path / "t.txt"
    token_file.write_text(GOOD + "\n")
    api = Api({GOOD})
    asked, _ = run_bootstrap(tmp_path, monkeypatch, api, [], token_file)
    assert asked == []
    assert token_file.read_text() == GOOD + "\n"
    out = capsys.readouterr()
    assert GOOD not in out.out + out.err


def test_bad_token_in_file_says_401_and_asks_again_and_overwrites_only_after_validation(tmp_path, monkeypatch, capsys):
    token_file = tmp_path / "t.txt"
    token_file.write_text(OLD + "\n")
    bad = "gh" + "p_" + "Bad1" * 9
    api = Api({GOOD})
    asked, err = run_bootstrap(tmp_path, monkeypatch, api, [bad, GOOD], token_file)
    assert len(asked) == 2
    assert token_file.read_text() == GOOD + "\n"
    out = capsys.readouterr().out
    assert "401" in out and "rejected" in out.lower()
    for t in (OLD, bad, GOOD):
        assert t not in out
    # the file kept the old token until the new one validated
    token_file.write_text(OLD + "\n")
    api2 = Api(set())
    asked, err = run_bootstrap(tmp_path, monkeypatch, api2, [bad, bad, bad, bad], token_file)
    assert isinstance(err, RuntimeError) and "token" in str(err).lower()
    assert token_file.read_text() == OLD + "\n"
    assert OLD not in str(err) and bad not in str(err)


def test_a_token_without_access_to_the_repo_is_treated_like_a_bad_token(tmp_path, monkeypatch, capsys):
    token_file = tmp_path / "t.txt"
    other = "gh" + "p_" + "Oth3" * 9
    api = Api({GOOD}, forbidden={other})
    asked, _ = run_bootstrap(tmp_path, monkeypatch, api, [other, GOOD], token_file)
    assert len(asked) == 2 and token_file.read_text() == GOOD + "\n"
    assert other not in capsys.readouterr().out


def test_github_down_is_an_error_not_a_prompt_loop(tmp_path, monkeypatch):
    token_file = tmp_path / "t.txt"
    token_file.write_text(GOOD + "\n")
    asked, err = run_bootstrap(tmp_path, monkeypatch, Api({GOOD}, down=True), ["x"], token_file)
    assert asked == [] and isinstance(err, RuntimeError) and "503" in str(err)
    assert token_file.read_text() == GOOD + "\n"


def test_token_file_with_several_lines_or_spaces_is_not_used_as_is(tmp_path, monkeypatch):
    token_file = tmp_path / "t.txt"
    token_file.write_text(GOOD + "\nsecond line\n")
    asked, _ = run_bootstrap(tmp_path, monkeypatch, Api({GOOD}), [GOOD], token_file)
    assert len(asked) == 1 and token_file.read_text() == GOOD + "\n"
