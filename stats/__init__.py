"""
PURPOSE:  Tier-1 package: the ONE implementation of each evaluation metric (rank-IC / Spearman, Pearson, AUC, Wilson interval, pinball loss, quantile ECE, ...) with ONE explicit tie/NaN policy, importable by every code package.
TAGS:     stats, tier 1, metrics, single owner, spearman, pearson, auc, wilson, pinball, ece, tie policy, nan policy, hardening C7
PITFALLS: stats may import ONLY `core` (tools/check_deps.py enforces it), the standard library and third-party numeric libraries (numpy, pandas, scipy); it must never import data/model/trainer/evaluation/workflow, so every package, tool or runner can `from stats.metrics import ...`. No import-time work. The repo root must be on sys.path (every runner notebook does it before importing its package). A metric changes only through a card: every consumer moves with it.

Modules:
  prediction_table  PredictionTable: build_table, validate_table (the leakage / key contract), write_table, read_table (csv.gz + .meta.json), PredictionTableSpec (the opt-in of the evaluation predict path); schema names in core/prediction_schema.py
  metrics    rank_average, pearson, spearman, partial_spearman, auc, wilson_interval, pinball_loss, mean_pinball, ece_quantile, qss, quantile_coverage, interval_coverage, winkler_score, threshold_trades, profit_factor, win_loss_ratio, net_sharpe, evaluate_heads (the per-head policy)
"""
