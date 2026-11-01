"""
PURPOSE:  The split ROLES declared once (tier 0, stdlib only): train / calibration / val / test / holdout (sealed), the additive role stamp every split carries (`split_role`), and the check a learned-state component runs before it fits (`require_role`): data whose role is not allowed raises, unstamped data raises (no silent default).
TAGS:     roles, split role, role stamp, calibration role, holdout sealed, require_role, stamp_role, role_of, RoleError, leakage control, fit on the wrong data, calibrator role check, tier 0
PITFALLS: The stamp is an ADDITIVE key (`split_role`) on a split dict (or on `DataFrame.attrs` for a flat table); arrays are never copied or touched (stamp_role returns a shallow copy). A `{asset: split}` dict (the shape of `test`) is never stamped on the container, only on each member (consumers iterate its values as splits); `role_of` of such a dict is the common role of its members (mixed roles raise). `DataFrame.attrs` survives copy/slicing but NOT groupby/merge/concat: stamp the table right before the component that needs it. Today's `val` split is selection AND calibration AND (side, q) choice at once: the `calibration` role exists so a later PR can carve a separate slice (tail of train with an embargo); until then a calibrator is allowed to fit on `val` (default allowed roles = calibration, val) and NEVER on train or holdout. A component with a learned state records `fit_role` in its state so a consumer can refuse a state fitted on the wrong role. Pure stdlib: pandas objects are recognised by duck typing (`.attrs` / `.copy`), never imported.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Tuple

__all__ = ["TRAIN", "CALIBRATION", "VAL", "TEST", "HOLDOUT", "ROLES", "ROLE_KEY", "NEVER_FIT_ROLES", "CALIBRATOR_ROLES", "CALIBRATION_CARVES",
           "RoleError", "check_role", "stamp_role", "role_of", "require_role"]

TRAIN = "train"              # the model is fitted here (also the only data a Normalizer may be fitted on)
CALIBRATION = "calibration"  # post-hoc calibrators (confidence / uncertainty / probability maps) are fitted here
VAL = "val"                  # selection (early stopping, best epoch, (side, q)); today ALSO the calibration data
TEST = "test"                # reporting
HOLDOUT = "holdout"          # sealed: opened once, at release, never fitted or selected on

#: Every declared role (the only valid values of the stamp).
ROLES: Tuple[str, ...] = (TRAIN, CALIBRATION, VAL, TEST, HOLDOUT)

#: Name of the stamp key on a split dict (and of the `DataFrame.attrs` entry of a flat table).
ROLE_KEY = "split_role"

#: Roles a post-hoc calibrator can never be fitted on: train errors are in-sample (over-confident out of sample), holdout is sealed.
NEVER_FIT_ROLES: Tuple[str, ...] = (TRAIN, HOLDOUT)

#: Default roles a calibrator may fit on: the dedicated calibration slice, or val (today's behaviour, until a calibration split exists).
CALIBRATOR_ROLES: Tuple[str, ...] = (CALIBRATION, VAL)


#: Where an (opt-in) calibration split is carved from by data/split_plan.py: the last rows of train ("train_tail": train shrinks) or the first
#: rows of val ("val_head": val starts later); an embargo is on both sides of the slice.
CALIBRATION_CARVES: Tuple[str, ...] = ("train_tail", "val_head")


class RoleError(ValueError):
    """Data of the wrong role (or no role stamp at all) was handed to a component that checks roles."""


def check_role(role: Any) -> str:
    """Return ``role`` if it is a declared role, else raise RoleError listing the declared ones."""
    if not isinstance(role, str) or role not in ROLES:
        raise RoleError(f"unknown split role {role!r}; declared roles: {list(ROLES)}")
    return role


def _is_frame(obj: Any) -> bool:
    return hasattr(obj, "attrs") and hasattr(obj, "copy") and hasattr(obj, "columns")


def _is_assets(obj: Mapping) -> bool:
    """A non-empty `{asset: split}` dict (every value a mapping, no 'y' key): the shape of `test`, never stamped itself."""
    return "y" not in obj and ROLE_KEY not in obj and len(obj) > 0 and all(isinstance(v, Mapping) for v in obj.values())


def stamp_role(obj: Any, role: str):
    """A shallow copy of ``obj`` carrying the role stamp; the original and every array in it are untouched.

    ``obj``: a split dict (adds ``split_role``), a ``{asset: split}`` dict (every member is stamped, the container is not), or a
    pandas DataFrame (``attrs['split_role']``). An already stamped object with a DIFFERENT role raises (re-labelling data is how a
    leak is made); the same role is a no-op copy."""
    role = check_role(role)
    if _is_frame(obj):
        have = obj.attrs.get(ROLE_KEY)
        if have is not None and have != role:
            raise RoleError(f"table already stamped {have!r}, refusing to re-stamp it {role!r}")
        out = obj.copy(deep=False)
        out.attrs = dict(obj.attrs)
        out.attrs[ROLE_KEY] = role
        return out
    if isinstance(obj, Mapping):
        if _is_assets(obj):
            return type(obj)((k, stamp_role(v, role)) for k, v in obj.items())
        have = obj.get(ROLE_KEY)
        if have is not None and have != role:
            raise RoleError(f"split already stamped {have!r}, refusing to re-stamp it {role!r}")
        out = dict(obj)
        out[ROLE_KEY] = role
        return out
    raise RoleError(f"cannot stamp a role on {type(obj).__name__}: expected a split dict, a {{asset: split}} dict or a DataFrame")


def role_of(obj: Any) -> str:
    """The role stamped on ``obj`` (split dict, `{asset: split}` dict of one common role, or DataFrame). Unstamped or mixed raises."""
    if _is_frame(obj):
        have = obj.attrs.get(ROLE_KEY)
        if have is None:
            raise RoleError("table has no role stamp (attrs['split_role']): stamp it with core.roles.stamp_role(table, role) "
                            "from the split it was built from; an unstamped table is never fitted on")
        return check_role(have)
    if isinstance(obj, Mapping):
        if _is_assets(obj):
            roles = {role_of(v) for v in obj.values()}
            if len(roles) != 1:
                raise RoleError(f"{{asset: split}} dict mixes roles {sorted(roles)}")
            return roles.pop()
        have = obj.get(ROLE_KEY)
        if have is None:
            raise RoleError("split has no role stamp ('split_role'): it was not produced by data.split_data / split_holdout / "
                            "rolling_splits, or was rebuilt without the key; an unstamped split is never fitted on")
        return check_role(have)
    raise RoleError(f"{type(obj).__name__} carries no role stamp")


def require_role(obj: Any, allowed: Iterable[str], *, what: str, never: Iterable[str] = ()) -> str:
    """Return the role of ``obj`` after checking it is in ``allowed`` and not in ``never``; otherwise raise RoleError naming ``what``."""
    allowed, never = tuple(allowed), tuple(never)
    role = role_of(obj)
    if role in never:
        raise RoleError(f"{what} can never be fitted on role {role!r} (never: {list(never)})")
    if role not in allowed:
        raise RoleError(f"{what} requires data of role {list(allowed)}, got {role!r}")
    return role
