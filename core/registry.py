"""
PURPOSE:  The ONE generic registry primitive (tier 0, stdlib only): `Registry(kind, contract)` registers objects by name with a declared contract (name, version, kind-specific required metadata, optional validator run AT REGISTRATION), raises on a duplicate name, and raises on an unknown name with the list of known names and a did-you-mean hint.
TAGS:     registry, Registry, Contract, register, contract check at registration, fail early, duplicate name, unknown name, did you mean, one component one interface, plugin, tier 0
PITFALLS: Read access is the dict protocol (`REG[name]`, `name in REG`, `REG.get(name, default)`, iteration, `keys/values/items` over the registered OBJECTS; `entries()` for the full records) so a plain dict can be replaced by a Registry without touching its readers; writes are NOT: `REG[name] = x` and `del REG[name]` raise TypeError (register / unregister are the only mutations). Nothing in the repo uses it yet: the existing registries (TASK_REGISTRY, HEAD_REGISTRY, NORMALIZERS, sample filters, signal_eval hypotheses) migrate in a separate `refactor` PR that proves identical contents. There is NO overwrite flag: registering a name twice raises, so a module that registers at import time must not be re-executed in the same process (use `unregister` in tests, or build a fresh Registry). The validator runs once, at `register`; it must be cheap and must not import TensorFlow at module level (core stays stdlib-only), so a heavy probe (e.g. calling a loss on tiny tensors) is injected by the caller as the validator. A validator reports problems by RETURNING a list of strings (or raising); `check_all()` re-runs it over every entry for a "whole registry still honours its contract" test. `UnknownNameError` subclasses both KeyError and ValueError so a later migration keeps `except KeyError` / `except ValueError` callers working; its str() is the plain message (no KeyError quoting). Metadata is stored by reference, not copied deeply.

Usage:
    TASKS = Registry("task", Contract("task", required_meta=("stat_keys",), validator=check_task))
    TASKS.register("nig", nig_loss, version=1, stat_keys=("mae", "nll"))
    TASKS.get("nig")             # the object; TASKS.get("nigg") raises UnknownNameError (known names + "did you mean 'nig'")
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Tuple

__all__ = ["Registry", "Contract", "Entry", "RegistryError", "DuplicateNameError", "UnknownNameError", "ContractError"]


class RegistryError(Exception):
    """Base class of every error raised by this module."""


class DuplicateNameError(RegistryError, ValueError):
    """A name was registered twice."""


class UnknownNameError(RegistryError, KeyError, ValueError):
    """A lookup of a name that is not registered; the message lists the known names."""

    def __str__(self) -> str:  # KeyError would show the repr (quoted) of its message
        return str(self.args[0]) if self.args else ""


class ContractError(RegistryError, ValueError):
    """The object or its metadata violates the registry's declared contract."""


_MISSING: Any = object()      # `get(name)` without a default raises; `get(name, None)` returns None

#: validator(name, obj, meta) -> None | iterable of problem strings (empty = fine); may also raise.
Validator = Callable[[str, Any, Mapping[str, Any]], Optional[Iterable[str]]]


@dataclass(frozen=True)
class Contract:
    """What every entry of one registry must declare.

    kind:          what is registered ("task", "head", "normalizer", ...); used in error messages.
    required_meta: metadata keys every registration must pass (besides name and version).
    optional_meta: extra keys allowed; None = any extra key is accepted, a tuple = only these (unknown keys raise).
    validator:     optional check run at registration (fail early, not hours into a run).
    """
    kind: str
    required_meta: Tuple[str, ...] = ()
    optional_meta: Optional[Tuple[str, ...]] = None
    validator: Optional[Validator] = None

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ContractError("Contract.kind must be a non-empty string")
        object.__setattr__(self, "required_meta", tuple(self.required_meta))
        if self.optional_meta is not None:
            object.__setattr__(self, "optional_meta", tuple(self.optional_meta))
        reserved = {"name", "version", "obj"} & set(self.required_meta + (self.optional_meta or ()))
        if reserved:
            raise ContractError(f"Contract({self.kind!r}): {sorted(reserved)} are reserved (name/version/obj are not metadata)")

    def check(self, name: str, obj: Any, version: Any, meta: Mapping[str, Any]) -> None:
        """Raise ContractError listing EVERY problem found (all at once, not one per attempt)."""
        problems: list[str] = []
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            problems.append(f"name must be a non-empty string without surrounding spaces, got {name!r}")
        ok_int = isinstance(version, int) and not isinstance(version, bool) and version >= 1
        ok_str = isinstance(version, str) and bool(version.strip())
        if not (ok_int or ok_str):
            problems.append(f"version must be a positive int or a non-empty string, got {version!r}")
        missing = [k for k in self.required_meta if k not in meta]
        if missing:
            problems.append(f"missing required metadata {missing} (required: {list(self.required_meta)})")
        if self.optional_meta is not None:
            unknown = sorted(set(meta) - set(self.required_meta) - set(self.optional_meta))
            if unknown:
                allowed = list(self.required_meta + self.optional_meta)
                problems.append(f"unknown metadata {unknown} (allowed: {allowed})")
        if not problems and self.validator is not None:
            try:
                found = self.validator(name, obj, meta)
            except Exception as e:  # a validator that raises is a violation, with the cause chained
                raise ContractError(f"{self.kind} {name!r}: contract validator raised {type(e).__name__}: {e}") from e
            problems.extend(str(p) for p in (found or ()))
        if problems:
            raise ContractError(f"{self.kind} {name!r} violates its contract: " + "; ".join(problems))


@dataclass(frozen=True)
class Entry:
    """One registered component: the object plus its declared identity and metadata."""
    name: str
    version: Any
    obj: Any = field(repr=False)
    meta: Mapping[str, Any] = field(default_factory=dict)


class Registry:
    """Name -> object registry with a declared contract. Insertion order is kept (`names()`)."""

    def __init__(self, kind: str, contract: Optional[Contract] = None):
        if contract is None:
            contract = Contract(kind)
        elif contract.kind != kind:
            raise ContractError(f"Registry kind {kind!r} != its contract kind {contract.kind!r}")
        self.kind = kind
        self.contract = contract
        self._entries: dict[str, Entry] = {}

    # ------------------------------------------------------------------ registration
    def register(self, name: str, obj: Any, *, version: Any, **meta: Any) -> Any:
        """Register `obj` under `name`; returns `obj` (so it can be used inline). Raises on a duplicate name or on any
        contract violation, BEFORE anything is stored (a failed registration leaves the registry unchanged)."""
        self.contract.check(name, obj, version, meta)
        if name in self._entries:
            old = self._entries[name]
            raise DuplicateNameError(f"{self.kind} {name!r} is already registered (version {old.version!r}); "
                                     f"a name is registered once: pick a new name (or unregister it in a test)")
        self._entries[name] = Entry(name, version, obj, dict(meta))
        return obj

    def decorator(self, name: str, *, version: Any, **meta: Any) -> Callable[[Any], Any]:
        """`@REG.decorator("name", version=1, key=...)` registers the decorated function/class and returns it unchanged."""
        def deco(obj: Any) -> Any:
            return self.register(name, obj, version=version, **meta)
        return deco

    def unregister(self, name: str) -> Entry:
        """Remove and return an entry (tests / explicit reload only). Unknown name raises like `get`."""
        self._require(name)
        return self._entries.pop(name)

    # ------------------------------------------------------------------ lookup
    def _require(self, name: Any) -> None:
        if name in self._entries:
            return
        known = sorted(self._entries)
        close = difflib.get_close_matches(str(name), known, n=1) if known else []
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise UnknownNameError(f"unknown {self.kind} {name!r}. Known {self.kind}s: {known}.{hint}")

    def entry(self, name: str) -> Entry:
        self._require(name)
        return self._entries[name]

    def get(self, name: Any, default: Any = _MISSING) -> Any:
        """The registered object. An unknown name raises UnknownNameError (known names + did-you-mean) unless a `default` is
        passed, in which case the default is returned (the `dict.get(name, default)` pattern of the code this replaces)."""
        if default is _MISSING:
            return self.entry(name).obj
        return self._entries[name].obj if name in self._entries else default

    def __getitem__(self, name: Any) -> Any:
        """`REG[name]` = `REG.get(name)`: the registered object; an unknown name raises UnknownNameError (a KeyError)."""
        return self.entry(name).obj

    def __setitem__(self, name: Any, obj: Any) -> None:
        raise TypeError(f"{self.kind} registry is read-only through [] (a name is registered once, with its version and "
                        f"contract): use register({name!r}, obj, version=..., ...)")

    def __delitem__(self, name: Any) -> None:
        raise TypeError(f"{self.kind} registry does not support del / pop: use unregister({name!r}) (tests / explicit reload only)")

    def names(self) -> Tuple[str, ...]:
        """Registered names in registration order."""
        return tuple(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def keys(self) -> list[str]:
        return list(self._entries)

    def values(self) -> list[Any]:
        """The registered objects in registration order (like `dict.values()`)."""
        return [e.obj for e in self._entries.values()]

    def items(self) -> list[tuple[str, Any]]:
        """(name, object) pairs in registration order (like `dict.items()`); `entries()` has the full Entry records."""
        return [(n, e.obj) for n, e in self._entries.items()]

    def entries(self) -> list[Entry]:
        """The full Entry records (name, version, obj, meta) in registration order."""
        return list(self._entries.values())

    # ------------------------------------------------------------------ whole-registry views
    def describe(self) -> list[dict[str, Any]]:
        """[{name, version, **meta}] in registration order: the content to fingerprint before/after a migration."""
        return [{"name": e.name, "version": e.version, **e.meta} for e in self._entries.values()]

    def check_all(self) -> list[str]:
        """Re-run the contract on every entry; the list of violation messages (empty = the registry honours its contract)."""
        out = []
        for e in self._entries.values():
            try:
                self.contract.check(e.name, e.obj, e.version, e.meta)
            except ContractError as err:
                out.append(str(err))
        return out

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {list(self._entries)})"
