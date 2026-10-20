# Drive sync: update the Drive folder from GitHub

`tools/drive_sync.py` copies one GitHub ref (branch, tag, or commit SHA) into the Drive folder the notebooks run from
(`/content/drive/MyDrive/crypto`). Rules, with no options to change them:

- a file present in GitHub **overwrites** the Drive copy (only if its content differs; counts are printed);
- a file only in Drive (data, models, checkpoints, secrets, outputs) is **kept**: nothing is ever deleted;
- no backup is made; `.git/` is skipped;
- `SYNCED_FROM.txt` in the Drive folder records `<full sha> <commit date> <subject>` of every run, and the sha is printed.
  Use it to decide whether the self-tests need to run again (see the `run_self_tests` cell of `main.ipynb`).

A file you delete from GitHub stays in Drive until you remove it by hand. The repo is private: keep the Colab secret `GITHUB_TOKEN`
(read access to Contents) enabled for the notebook; the tool reads it from the environment or Colab secrets and never prints it.

## One Colab cell (also the first run, from a fresh Drive)
It borrows the tool from a throwaway clone only when Drive does not hold `tools/drive_sync.py` yet, then calls it.
After the sync, restart the runtime so imported modules are reloaded.

```python
import base64, os, subprocess, sys
from google.colab import drive, userdata
drive.mount("/content/drive")
DEST, REF = "/content/drive/MyDrive/crypto", "master"      # REF: branch, tag or commit SHA
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
TOOLS = f"{DEST}/tools"
if not os.path.isfile(f"{TOOLS}/drive_sync.py"):             # first run only
    TOOLS = "/tmp/csp_boot/tools"
    auth = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
    r = subprocess.run(["git", "-c", f"http.extraHeader=Authorization: Basic {auth}", "clone", "--depth", "1", "-q",
                        "https://github.com/yuosef772424/crypto-signal-prediction.git", "/tmp/csp_boot"], capture_output=True)
    assert r.returncode == 0, "bootstrap clone failed: check the GITHUB_TOKEN secret"
subprocess.run([sys.executable, f"{TOOLS}/drive_sync.py", "--ref", REF, "--dest", DEST], check=True)
```

Add `"--dry-run"` to the argument list to see the plan without writing. From Python: `from drive_sync import sync` (with `tools/` on
`sys.path`), `sync(ref, dest, token=None)`. Tests: `tests/test_drive_sync.py`.
