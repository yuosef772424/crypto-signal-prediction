# How to add a price normalizer

Every price normalization mode is a `Normalizer` registered in `data/normalizers.py` (`NORMALIZERS`, `get_normalizer(name)`,
`normalizer_for_mode(mode)`). Consumers (windows, pipeline, split, `invert_reg_predictions`, `build_dataset_live`, evaluation
decode) call the registry and never branch on a mode name.

1. Subclass `Normalizer` (or `_LogretBase` for a log-return family) in `data/normalizers.py`; set `name`.
2. Implement only what differs from the base defaults (all hooks default to "do nothing" = `window_scale`):
   `fit(train, feature_order=, timeframes=) -> state` (train only; a JSON-safe dict with `state['mode'] == name`),
   `transform_inputs(X, state)`, `normalize_targets(y, state)`, `normalize_prices` / `inverse_targets(pred, state, last_close, head, output)`
   (keep the one inverse formula in `core/price_norm.py` if evaluation must share it), and the build/split/live hooks
   (`check_config`, `build_inputs`, `dataset_fields`, `fit_at_split`, `stamp_split`, `prepare_live`).
3. Add an instance to `NORMALIZERS` and the name to `PRICE_NORM_MODES` (`data/normalize.py`). Nothing else to edit.
4. Extend `tests/test_normalizer.py` (registry set, train-only fit, dict round trip, fingerprints, round trip of targets).

Rules: a new option's default must reproduce the old output byte-for-byte; unknown names and unsupported combinations raise.
