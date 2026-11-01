"""
PURPOSE:  Tier-0 package: the pure constants and schema names that several packages used to copy (last_candles column schema, price target columns, target-mode names, NIG alpha floor). Normal Python package, no side effects, imports nothing from this repo.
TAGS:     core, tier 0, registry, constants, schema, single source, LAST_COLUMNS, TS_COL, NIG_ALPHA_DEN_MIN, TARGET_MODES, shared constants
PITFALLS: core may import NOTHING from the repo (tools/check_deps.py enforces it) and must stay stdlib-only with no import-time work, so any package, tool or runner can `from core.schema import ...` in milliseconds. The shared-namespace packages (data, model, workflow, ...) bind these names into their namespaces with a plain import, so notebooks and tests still see `LAST_COLUMNS`, `TS_COL`, ... under the same names. Add only pure constants here (the one exception is the generic `core.registry` primitive, which has no repo-specific logic): a value change here changes every consumer at once, so keep it byte-identical unless a card says otherwise. The repo root must be on sys.path (every runner notebook does it before importing its package).

Modules:
  schema     last_candles columns and indexes, target price columns, target-mode names
  constants  numeric constants shared by model and evaluation (NIG alpha floor)
  registry   generic Registry/Contract primitive: register by name with a declared contract, duplicate/unknown names raise
  roles      split roles (train / calibration / val / test / holdout), the role stamp and the role check of learned-state components
  prediction_schema  the PredictionTable column names (required columns, time columns, key, head-output prefix)
  seeds      seed_everything(seed): the one place python/numpy/TF are seeded (framework seeder injected by the caller; core stays stdlib-only)
"""
