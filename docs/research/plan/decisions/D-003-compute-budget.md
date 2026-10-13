# D-003: Colab budget ≤ 1 hour per phase

**Status:** Accepted (owner, 2026-10-13).

**Decision.** Each phase's Colab runs together fit in one hour. Scripts stop cleanly at `--max-minutes` (default 50)
and record what was skipped in `manifest.json`.

**Consequences.** Universes are capped (e.g. 20 coins for R1), configuration grids are small and fixed in the card,
and heavy model training is out of scope for this plan.
