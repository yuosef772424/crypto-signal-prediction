# stale_bars — dead-coin tails vs exact-zero close_reg

Prepared Colab job (not queued): `job.json` runs `tools/stale_bars_colab_job.py` on Drive `history_15m` at 1h/4h/1D.
Question it answers: of the exact-zero `close_reg` targets (train 7.1%, val 17.1% on the owner's rebuild), how many come from
bars after a coin's last real trade or inside long zero-volume flat runs (removable by `CONFIG['stale_bars']`), and how many are
genuine zero returns of live coins (volume > 0)? Fill `repo_ref` with the pushed 40-hex SHA before queueing.

stale-bars-001 result: derive_min_run gave 6645/5058/853 bars (inconsistent). `job_002.json` (stale-bars-002, not queued) uses `min_run_hours: 168` (7 days) at every timeframe.
