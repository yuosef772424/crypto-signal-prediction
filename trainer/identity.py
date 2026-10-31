"""
PURPOSE:  Persisted run identity (audit C3): RunIdentity (config fingerprint + the canonical config it hashes, seed, optional data fingerprint, git sha, parent_hashes), identity.json in the run dir, and the resume check that raises with a key-level diff when the training objective changed.
TAGS:     RunIdentity, identity.json, run identity, collect_run_identity, check_run_identity, RunIdentityMismatch, resume check, config diff, parent_hashes, seed, code version, trainer
PITFALLS: The fingerprint is trainer.config.config_fingerprint (NOT a second hash) with the run LOCATION keys (run_dir, mirror_dir, mirror_every) also ignored, so a run folder copied/mirrored to another path still resumes; epochs / batch_size / seed / verbose / train_mode are ignored by it already (extending a run is legitimate). Only the fingerprint (and the data fingerprint when BOTH sides have one) can fail a resume; seed and code version differences are one-line notes (a resume on a newer commit is normal). A run dir with checkpoints but no identity.json (written before this module) resumes UNCHANGED with one warning and ADOPTS the identity (the file is written from the current config, which cannot be verified against what trained it). A corrupt identity.json raises (never guessed). identity.json is archived with the rest of the state by archive_state (STATE_ENTRIES). data_fingerprint and parent_hashes are plumbing only today: build_training_system takes them as optional arguments, nothing in the main path computes a data hash yet (cascaded stages / rolling retrain, docs/research/hardening/proposal_*.md, fill parent_hashes). Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its own).
"""
import dataclasses as _dc
import subprocess as _subprocess

IDENTITY_FILE = "identity.json"
IDENTITY_SCHEMA_VERSION = 1
#: run keys that say WHERE a run lives, not what it trains: ignored by the identity fingerprint (moving the folder is not a new objective)
IDENTITY_IGNORED_RUN_KEYS = ("run_dir", "mirror_dir", "mirror_every")


class RunIdentityMismatch(ValueError):
    """A resumed run dir was trained with a different objective than the config now given."""


@_dc.dataclass(frozen=True)
class RunIdentity:
    """Who a run is: written once at the start of a run, compared on every resume."""
    config_fingerprint: str
    #: the canonical config the fingerprint hashes (kept so a mismatch can say WHICH keys changed)
    config_payload: dict
    seed: int
    data_fingerprint: Optional[str] = None
    code_version: Optional[str] = None
    #: fingerprints of the runs this one derives from (cascaded stage / rolling retrain); empty today
    parent_hashes: tuple = ()
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = _dc.asdict(self)
        d["parent_hashes"] = list(self.parent_hashes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunIdentity":
        known = {f.name for f in _dc.fields(cls)}
        extra = set(d) - known
        if extra:
            raise ValueError(f"unknown identity field(s) {sorted(extra)}; known: {sorted(known)}")
        missing = {"config_fingerprint", "config_payload", "seed"} - set(d)
        if missing:
            raise ValueError(f"identity is missing field(s) {sorted(missing)}")
        return cls(**{**d, "parent_hashes": tuple(d.get("parent_hashes") or ())})


def _code_version() -> Optional[str]:
    """Git sha of the checkout this module runs from; None when git or the checkout is unavailable."""
    here = os.path.dirname(os.path.abspath(_code_version.__code__.co_filename))
    try:
        out = _subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
    except (OSError, _subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def collect_run_identity(config: dict, data_fingerprint=None, parent_hashes=()) -> RunIdentity:
    """Identity of the run ``config`` describes (seed = config["run"]["seed"])."""
    return RunIdentity(
        config_fingerprint=config_fingerprint(config, IDENTITY_IGNORED_RUN_KEYS),
        config_payload=fingerprint_payload(config, IDENTITY_IGNORED_RUN_KEYS),
        seed=int(config["run"]["seed"]), data_fingerprint=data_fingerprint,
        code_version=_code_version(), parent_hashes=tuple(parent_hashes))


def identity_path(run_dir: str) -> str:
    return os.path.join(run_dir, IDENTITY_FILE)


def read_run_identity(run_dir: str) -> Optional[RunIdentity]:
    """The stored identity of ``run_dir``; None when there is no identity.json; raises when it exists but cannot be read."""
    path = identity_path(run_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return RunIdentity.from_dict(json.load(f))
    except Exception as e:
        raise ValueError(f"❌ {path} exists but is not a valid run identity ({e}). Fix or remove it deliberately "
                         f"(removing it makes the run dir resume as a legacy run and adopt the current config).") from e


def write_run_identity(run_dir: str, identity: RunIdentity) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = identity_path(run_dir)
    atomic_write_json(path, identity.to_dict())
    return path


def _flatten(o, prefix=""):
    if isinstance(o, dict) and o:
        for k in sorted(o, key=str):
            yield from _flatten(o[k], f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, o


def config_payload_diff(old: dict, new: dict) -> List[Tuple[str, str]]:
    """``(dotted key, one-line description)`` per changed / added / removed key between two fingerprint payloads."""
    a, b = dict(_flatten(old)), dict(_flatten(new))
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            out.append((k, f"  - {k}: removed (was {a[k]!r})"))
        elif k not in a:
            out.append((k, f"  - {k}: added ({b[k]!r})"))
        elif a[k] != b[k]:
            out.append((k, f"  - {k}: {a[k]!r} -> {b[k]!r}"))
    return out


def check_run_identity(run_dir: str, current: RunIdentity, allow_changes=()) -> str:
    """Resume check. Returns "match" (identity.json present and equal) or "adopted" (legacy dir without one: written now, one warning).
    Raises RunIdentityMismatch when the config fingerprint changed (message = key-level diff), or when both sides carry a data
    fingerprint and they differ. ``allow_changes``: dotted config keys whose change is accepted (the mismatch is raised when any
    OTHER key changed); the stored identity is NOT rewritten."""
    stored = read_run_identity(run_dir)
    if stored is None:
        print(f"⚠️ {run_dir} has no {IDENTITY_FILE} (run started before run identities): resuming unchanged and adopting the "
              f"current config as its identity — it cannot be verified against what trained it")
        write_run_identity(run_dir, current)
        return "adopted"
    if stored.config_fingerprint != current.config_fingerprint:
        diff = [line for key, line in config_payload_diff(stored.config_payload, current.config_payload) if key not in set(allow_changes)]
        if diff or not allow_changes:
            raise RunIdentityMismatch(
                f"❌ run dir {run_dir} was trained with a different objective (config fingerprint {stored.config_fingerprint} "
                f"-> {current.config_fingerprint}); changed keys:\n" + ("\n".join(diff) or "  (payload not comparable)")
                + "\nResuming would continue the old optimizer / schedule / best-epoch state under the new objective. "
                  "Revert the change, use a new run_dir, or start from the saved weights deliberately with train_mode='warm_start' / 'new'.")
    if stored.data_fingerprint and current.data_fingerprint and stored.data_fingerprint != current.data_fingerprint:
        raise RunIdentityMismatch(f"❌ run dir {run_dir} was trained on different data (data fingerprint {stored.data_fingerprint} "
                                  f"-> {current.data_fingerprint}); use a new run_dir.")
    if stored.seed != current.seed:
        print(f"⚠️ resuming {run_dir} with seed {current.seed} (it was started with seed {stored.seed}): the seed is not part of the objective")
    if stored.code_version and current.code_version and stored.code_version != current.code_version:
        print(f"ℹ️ resuming {run_dir} on code {current.code_version[:10]} (started on {stored.code_version[:10]})")
    return "match"
