"""
PURPOSE:  Tier-1 package: the ONE implementation of each evaluation metric (rank-IC / Spearman, Pearson, AUC, Wilson interval, pinball loss, quantile ECE, ...) with ONE explicit tie/NaN policy, importable by every code package.
TAGS:     stats, tier 1, metrics, single owner, spearman, pearson, auc, wilson, pinball, ece, tie policy, nan policy, hardening C7
PITFALLS: stats may import ONLY `core` (tools/check_deps.py enforces it) and third-party numeric libraries (numpy, pandas, scipy); it must never import data/model/trainer/evaluation/workflow, so every package, tool or runner can `from stats.metrics import ...`. No import-time work. The repo root must be on sys.path (every runner notebook does it before importing its package). A metric changes only through a card: every consumer moves with it.

Modules:
  metrics    rank_average, pearson, spearman, partial_spearman, auc, wilson_interval, pinball_loss, mean_pinball, ece_quantile
"""
