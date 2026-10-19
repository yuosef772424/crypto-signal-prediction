import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import trim_dead_tails as tdt  # noqa: E402

HEADER = "timestamp,open,high,low,close,volume\n"


def bars(spec, start=1_700_000_000_000):
    """spec: list of volumes -> csv text, 15 min apart."""
    return HEADER + "".join(f"{start + i * 900000},1.5,2.0,1.0,1.25,{v}\n" for i, v in enumerate(spec))


def put(d, name, text):
    p = os.path.join(d, name)
    raw = text.encode()
    with open(p, "wb") as fh:
        fh.write(gzip.compress(raw, mtime=123) if name.endswith(".gz") else raw)
    return p


def content(p):
    return tdt.read_content(p)


@pytest.fixture(params=["csv", "csv.gz"])
def ext(request):
    return request.param


@pytest.fixture
def d(tmp_path):
    p = tmp_path / "history_15m"
    p.mkdir()
    return str(p)


def test_live_coin_untouched(d, ext):
    p = put(d, f"BTC.{ext}", bars([5, 0, 3, 2]))
    before = open(p, "rb").read()
    res = tdt.trim_dir(d)
    assert res[0]["status"] == "live" and open(p, "rb").read() == before
    assert not os.path.exists(tdt.tails_dir(d))


def test_trim_exact_and_midhistory_zero_kept(d, ext):
    p = put(d, f"AGIX.{ext}", bars([5, 0, 0, 3, 0, 4, 0, 0, 0]))
    orig = content(p)
    res = tdt.trim_dir(d)[0]
    assert res["status"] == "trimmed" and res["n_tail_bars"] == 3 and res["n_bars_before"] == 9
    assert content(p).decode() == bars([5, 0, 0, 3, 0, 4])           # mid zeros kept, last volume>0 bar is the last row
    tail = content(os.path.join(tdt.tails_dir(d), f"AGIX.{ext}")).decode()
    assert tail.startswith(HEADER) and tail.count("\n") == 4
    m = tdt.read_manifest(tdt.tails_dir(d))["AGIX"]
    assert m["n_tail_bars"] == "3" and m["original_content_sha256"] == tdt.sha256_bytes(orig)
    assert m["last_real_ts"] == str(1_700_000_000_000 + 5 * 900000) and m["first_tail_ts"] == str(1_700_000_000_000 + 6 * 900000)
    assert m["trimmed_sha256"] == tdt.sha256_file(p)


def test_restore_reproduces_original(d, ext):
    p = put(d, f"OCEAN.{ext}", bars([1, 2, 0, 0]))
    orig_file, orig_sha = open(p, "rb").read(), tdt.sha256_file(p)
    assert m_status(d) == "trimmed"
    r = tdt.restore_coin(d, "OCEAN")
    assert r["status"] == "restored"
    assert content(p) == (gzip.decompress(orig_file) if ext.endswith("gz") else orig_file)
    if ext == "csv":
        assert tdt.sha256_file(p) == orig_sha and r["file_bytes_identical"]
    assert "OCEAN" not in tdt.read_manifest(tdt.tails_dir(d))


def m_status(d):
    return tdt.trim_dir(d)[0]["status"]


def test_no_trailing_newline_roundtrip(d):
    p = put(d, "X.csv", bars([1, 0, 0]).rstrip("\n"))
    orig = open(p, "rb").read()
    assert m_status(d) == "trimmed"
    assert tdt.restore_coin(d, "X")["status"] == "restored" and open(p, "rb").read() == orig


def test_idempotent_keeps_original_sha(d, ext):
    put(d, f"FTM.{ext}", bars([1, 0, 0]))
    tdt.trim_dir(d)
    m1 = tdt.read_manifest(tdt.tails_dir(d))
    res = tdt.trim_dir(d)[0]
    assert res["status"] == "already_trimmed" and res["n_tail_bars"] == 0
    assert tdt.read_manifest(tdt.tails_dir(d)) == m1


def test_mismatch_does_not_replace(d, ext, monkeypatch):
    p = put(d, f"KLAY.{ext}", bars([1, 0, 0]))
    before = open(p, "rb").read()
    monkeypatch.setattr(tdt, "write_atomic", lambda path, c: open(path, "wb").write(b"corrupt"))
    res = tdt.trim_dir(d)[0]
    assert res["status"] == "error" and open(p, "rb").read() == before
    assert tdt.read_manifest(tdt.tails_dir(d)) == {}


def test_trimmed_readback_mismatch_does_not_replace(d, monkeypatch):
    p = put(d, "K.csv", bars([1, 0, 0]))
    before = open(p, "rb").read()
    monkeypatch.setattr(tdt.gzip, "decompress", lambda b: b"x")      # only used for .gz; force via name
    p2 = put(d, "G.csv.gz", bars([1, 0]))
    b2 = open(p2, "rb").read()
    res = {r["coin"]: r for r in tdt.trim_dir(d)}
    assert res["G"]["status"] == "error" and open(p2, "rb").read() == b2
    assert not os.path.exists(p2 + ".trim.tmp") and res["K"]["status"] == "trimmed" and open(p, "rb").read() != before


def test_existing_different_tail_is_error(d):
    p = put(d, "A.csv", bars([1, 0]))
    os.makedirs(tdt.tails_dir(d))
    put(tdt.tails_dir(d), "A.csv", "other\n")
    before = open(p, "rb").read()
    assert tdt.trim_dir(d)[0]["error"] == "tail_file_exists_with_different_content" and open(p, "rb").read() == before


def test_dry_run_writes_nothing(d, ext):
    p = put(d, f"AGIX.{ext}", bars([1, 0, 0, 0]))
    before = open(p, "rb").read()
    res = tdt.trim_dir(d, dry_run=True)[0]
    assert res["status"] == "would_trim" and res["n_tail_bars"] == 3
    assert open(p, "rb").read() == before and not os.path.exists(tdt.tails_dir(d))
    assert tdt.main(["trim", d, "--dry-run"]) == 0 and not os.path.exists(tdt.tails_dir(d))


def test_all_zero_volume_skipped_with_error(d):
    p = put(d, "DEAD.csv", bars([0, 0, 0]))
    before = open(p, "rb").read()
    res = tdt.trim_dir(d)[0]
    assert res["status"] == "error" and res["error"] == "all_zero_volume"
    assert open(p, "rb").read() == before and not os.path.exists(tdt.tails_dir(d))
    assert tdt.main(["trim", d]) == 1


def test_cli_restore_all_and_changed_file_refused(d):
    put(d, "A.csv", bars([1, 0]))
    put(d, "B.csv.gz", bars([2, 0, 0]))
    assert tdt.main(["trim", d]) == 0
    with open(os.path.join(d, "A.csv"), "ab") as fh:
        fh.write(b"extra\n")
    assert tdt.restore_coin(d, "A")["error"] == "file_changed_since_trim"
    assert tdt.main(["restore", d, "B"]) == 0
    assert content(os.path.join(d, "B.csv.gz")).decode() == bars([2, 0, 0])


def test_job_entry(d, tmp_path, monkeypatch):
    import trim_dead_tails_colab_job as job
    put(d, "A.csv", bars([1, 0, 0]))
    put(d, "B.csv", bars([1, 2]))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"data_dir": d, "dry_run": False}))
    out = tmp_path / "out"
    assert job.main(["--config", str(cfg), "--out", str(out), "--max-minutes", "5"]) == 0
    s = json.load(open(out / "summary.json"))
    assert s["coins_scanned"] == 2 and s["coins_trimmed"] == 1 and s["total_tail_bars"] == 2 and s["errors"] == 0
    assert (out / "trimmed_coins.csv").read_text().count("\n") == 2 and (out / "manifest.json").exists()
