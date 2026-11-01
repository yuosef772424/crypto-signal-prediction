"""
PURPOSE:  TargetCodec registry (audit PR13 / C4): ONE registered component per target mode in use, owning BOTH directions of the target transform: forward (price columns -> target space, used by workflow.retarget) and inverse (target space -> the SIMPLE RETURN fut/last - 1 of the target's own price type, used by every consumer: evaluation, the utility / IC loss terms, live). Codecs: return, volnorm, volret, relative_median.
TAGS:     TargetCodec, CODECS, codec_name_for_mode, return codec, volnorm, volret, relative_median, decode_return, decode_absolute, target_vol, target_group_median, one component one interface, fit transform inverse
PITFALLS: Interface per codec: fit(train_P) -> state (JSON-safe; the codecs here learn nothing from train, the state records the parameters the inverse needs: scale, center, min_group), stamps(P, state) -> {split key: (n,) array} (per-row facts the inverse needs, stored WITH the split: target_vol), forward(P, target, state) -> z (before TargetSettings.clip and the reg scale), inverse(z, stamps, state, ops=None) -> simple return. `ops` supplies expm1 (numpy default; the trainer passes tf.math) so the formula exists once for numpy and TensorFlow. Clipping is lossy: a clipped row decodes to the clip, never beyond (document it where used). relative_median: its inverse returns the RELATIVE return (return minus the same-timestamp median over coins); the per-timestamp median is known only ex post, so `decode_absolute(z, median, state)` needs it explicitly (it is stamped on the split as target_group_median_<target>, raw return units, for EVALUATION only, never an input of the model); a relative variant of a vol mode has no codec (the median is of z, not of the return) and raises. Modes NOT migrated (target-mode-branch ratchet): scaled, magnitude, return_close, entry_range (+ its range_pos close), the chicks bridge and entry_range_to_prices still branch on the mode name. modes without a registered codec raise in codec_name_for_mode, never decode silently.
"""
import numpy as np
import pandas as pd

from core.price_norm import volret_forward, volret_inverse
from core.registry import Contract, Registry

__all__ = ["CODECS", "codec_name_for_mode", "decode_return", "decode_absolute"]


def _check_codec(name, codec, meta):
    problems = []
    for m in ("fit", "stamps", "forward", "inverse"):
        if not callable(getattr(codec, m, None)):
            problems.append(f"codec needs a callable {m}()")
    if getattr(codec, "name", None) != name:
        problems.append(f"registered as {name!r} but codec.name is {getattr(codec, 'name', None)!r}")
    return problems


CODECS = Registry("target codec", Contract("target codec", required_meta=("description",), optional_meta=(), validator=_check_codec))

_FUT = {"high": "future_high_max", "low": "future_low_min", "close": "future_close"}


class _Ops:
    expm1 = staticmethod(np.expm1)


class ReturnCodec:
    """z = (fut/last - 1) clipped, times reg_target_scale (as the pipeline); inverse = z / scale."""
    name = "return"

    def fit(self, train_P, scale=1.0):
        return {"scale": float(scale)}

    def stamps(self, P, state):
        return {}

    def forward(self, P, t, state):
        return P[_FUT[t]] / P[f"last_{t}"] - 1.0

    def inverse(self, z, stamps, state, ops=None):
        return z / state["scale"]


class VolnormCodec:
    """z = (fut/last - 1) / vol, vol = |window IQR| / last_close (base_params, past-only); inverse = z * vol (stamped target_vol)."""
    name = "volnorm"

    def fit(self, train_P, scale=1.0):
        return {"scale": 1.0}

    @staticmethod
    def vol(P):
        if "bp_scale" not in P:
            raise ValueError("وضع 'volnorm' يتطلّب base_params في كل الأقسام")
        vol = np.abs(P["bp_scale"]) / np.maximum(np.abs(P["last_close"]), 1e-12)
        return np.where(vol > 1e-6, vol, np.nanmedian(vol[vol > 1e-6]) if np.any(vol > 1e-6) else 1.0)

    def stamps(self, P, state):
        return {"target_vol": self.vol(P).astype("float32")}

    def forward(self, P, t, state):
        return (P[_FUT[t]] / P[f"last_{t}"] - 1.0) / self.vol(P)

    def inverse(self, z, stamps, state, ops=None):
        return z * stamps["target_vol"]


class VolretCodec:
    """z = log(fut/last) / vol, vol = EWMA std of the window's past close log returns (P['win_vol']); inverse = expm1(z * vol)."""
    name = "volret"

    def fit(self, train_P, scale=1.0):
        return {"scale": 1.0}

    def stamps(self, P, state):
        return {"target_vol": P["win_vol"].astype("float32")}

    def forward(self, P, t, state):
        if "win_vol" not in P:
            raise ValueError("وضع 'volret' يتطلّب win_vol (retarget_splits يحسبه من X: base_tf وclose_index)")
        with np.errstate(all="ignore"):
            z = volret_forward(np.log(P[_FUT[t]] / P[f"last_{t}"]), P["win_vol"])
        return np.where(np.isfinite(z), z, 0.0)

    def inverse(self, z, stamps, state, ops=None):
        return (ops or _Ops).expm1(volret_inverse(z, stamps["target_vol"]))


class RelativeMedianCodec:
    """z = r - median(r over the coins of the same timestamp group) (min_group coins), r = fut/last - 1, times reg_target_scale.
    inverse = z / scale = the RELATIVE return; decode_absolute adds the (ex-post) median."""
    name = "relative_median"

    def fit(self, train_P, scale=1.0, center="median", min_group=5, group_freq=None):
        return {"scale": float(scale), "center": center if isinstance(center, str) else getattr(center, "__name__", "custom"),
                "min_group": int(min_group), "group_freq": group_freq}

    def stamps(self, P, state):
        return {}

    def forward(self, P, t, state):
        return P[_FUT[t]] / P[f"last_{t}"] - 1.0

    def stamps_for(self, base, medians):
        """The ex-post per-timestamp medians ({target: array}) as split stamps; only for the 'return' base (a relative vol mode has no
        return median, hence no codec)."""
        return {f"target_group_median_{t}": c.astype("float64") for t, c in medians.items()} if base == "return" else {}

    @staticmethod
    def center(r, keys, center="median"):
        """Per-group center of ``r`` (same length): the ONE place the cross-sectional median is computed. Returns (r - c, c)."""
        c = pd.Series(r).groupby(keys).transform(center).to_numpy()
        return r - c, c

    def inverse(self, z, stamps, state, ops=None):
        return z / state["scale"]


for _c, _d in ((ReturnCodec(), "identity x reg_target_scale (the pipeline's 'return')"),
               (VolnormCodec(), "return / (window IQR / last close), past-only"),
               (VolretCodec(), "log return / EWMA std of the past close log returns inside the window, past-only"),
               (RelativeMedianCodec(), "return minus the per-timestamp median over coins (min_group); inverse = relative return")):
    CODECS.register(_c.name, _c, version=1, description=_d)
del _c, _d


def codec_name_for_mode(mode):
    """The codec of a target mode string (``None`` = the pipeline's 'return'); modes without a codec raise (listing them)."""
    if mode is None:
        return "return"
    base, _, suffix = str(mode).partition("+")
    if base == "relative" and not suffix:
        return "relative_median"
    if suffix == "relative":
        if base == "return":
            return "relative_median"
        raise ValueError(f"target mode {mode!r} has no TargetCodec: the median of a vol-normalised target is not a return median")
    if base in CODECS:
        return base
    raise ValueError(f"target mode {mode!r} has no TargetCodec (registered: {list(CODECS.names())}; not migrated: scaled, magnitude, "
                     "return_close, entry_range)")


def _stamps_of(split):
    return {k: split[k] for k in ("target_vol",) if k in split}


def decode_return(split, z, target="close", ops=None):
    """Simple return ``fut/last - 1`` of ``target`` from model output/label ``z`` of ONE split dict (train/val or one test asset),
    through the codec of the split's stamped target_mode: the one decode every consumer calls. ``relative_median`` returns the RELATIVE
    return (see decode_absolute)."""
    if "y" not in split:
        raise ValueError("decode_return takes one split (train / val / one test asset), not a dict of assets: decode per asset")
    name = codec_name_for_mode(split.get("target_mode"))
    codec = CODECS.get(name)
    per_target = split.get("reg_target_scales") or {}
    scale = float(per_target.get(target, split.get("reg_target_scale", 1.0)))
    stamps = _stamps_of(split)
    if name in ("volnorm", "volret") and "target_vol" not in stamps:
        raise ValueError(f"split has target_mode {split.get('target_mode')!r} but no target_vol stamp (retargeted before the codec "
                         "existed): run retarget_splits again")
    return codec.inverse(z, stamps, {"scale": scale}, ops)


def decode_absolute(split, z, median, target="close"):
    """Absolute simple return of a 'relative' split: ``decode_return + median`` where ``median`` is the realised per-timestamp median
    return in RAW return units (ex post: pass ``split['target_group_median_<target>']`` for evaluation, or a separate market forecast)."""
    if codec_name_for_mode(split.get("target_mode")) != "relative_median":
        raise ValueError("decode_absolute is for 'relative' splits only")
    return decode_return(split, z, target) + np.asarray(median, dtype="float64")
