"""
PURPOSE:  Single paste-and-run Colab cell for phase R0: mount Drive, fetch tools/data_manifest.py (GitHub token or Drive copy),
          run it over history_15m, print the paste bundle and also save it to Drive as lock_manifest_bundle.txt.
TAGS:     colab cell, data manifest, history_15m, paste bundle, drive mount, lock, R0, SOURCE github/drive
PITFALLS: Private repo: the GitHub source needs a token from Colab secrets (GITHUB_TOKEN). BRANCH names the branch that carries
          the script and goes stale after the merge (use "master" then). CONFIG paths are the only settings to edit; unknown
          keys make data_manifest.py raise. Re-running is safe (read-only on the data folders).
"""
# ═══════════════════════════════════════════════════════════════════════════════
# One Colab cell: mount Drive -> fetch data_manifest.py -> list history_15m (about 5 minutes) -> print the paste bundle.
# Paste the bundle (everything between "PASTE BUNDLE BEGIN" and "PASTE BUNDLE END", inclusive) into the build session.
# If the output is too long to select in the browser, open MyDrive/lock_manifest_bundle.txt and copy it from there.
# ═══════════════════════════════════════════════════════════════════════════════
import json
import os
import shutil
import subprocess
import sys
import urllib.request

# ── settings ──
SOURCE = "github"          # "github" (private repo: token from Colab secrets) | "drive" (a copy of the script in DRIVE_SCRIPT)
BRANCH = "claude/r0-lock"  # branch that carries tools/data_manifest.py ("master" once merged)
DRIVE_SCRIPT = "/content/drive/MyDrive/tools/data_manifest.py"                      # with SOURCE = "drive"
CONFIG = {"history_15m_dir": "/content/drive/MyDrive/history_15m",                  # the folder with <SYMBOL>.csv.gz
          "extra_folders": {"history_1d": "/content/drive/MyDrive/history_1d"}}     # folder rules of the touched ledger
MAX_MINUTES = 50
BUNDLE_COPY = "/content/drive/MyDrive/lock_manifest_bundle.txt"

# ── 1) Google Drive ──
from google.colab import drive  # noqa: E402
drive.mount("/content/drive")

# ── 2) the script ──
SCRIPT = "/content/data_manifest.py"
if SOURCE == "github":
    RAW = f"https://raw.githubusercontent.com/yuosef772424/crypto-signal-prediction/{BRANCH}/tools/data_manifest.py"
    try:
        from google.colab import userdata
        _tok = userdata.get("GITHUB_TOKEN") or ""
    except Exception:
        _tok = ""
    if not _tok:
        from getpass import getpass
        _tok = getpass("GitHub token (private repo): ")
    _req = urllib.request.Request(RAW, headers={"Authorization": f"token {_tok.strip()}"})
    with urllib.request.urlopen(_req) as _r, open(SCRIPT, "wb") as _f:
        _f.write(_r.read())
else:
    shutil.copy(DRIVE_SCRIPT, SCRIPT)
print(f"script ({SOURCE}): {os.path.getsize(SCRIPT):,} bytes")

# ── 3) run ──
json.dump(CONFIG, open("/content/data_manifest_config.json", "w"))
proc = subprocess.run([sys.executable, "-u", SCRIPT, "--config", "/content/data_manifest_config.json",
                       "--out", "/content/manifest_out", "--max-minutes", str(MAX_MINUTES)],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print(proc.stderr)
print(proc.stdout)
open(BUNDLE_COPY, "w", encoding="utf-8").write(proc.stdout)
print(f"exit code {proc.returncode} (0 = complete); bundle also saved to {BUNDLE_COPY}")
