# External research takeaways (2026-10-20)

Source: the owner's summary of 4 LLM search tools. **All citations are unverified**. Verify a paper before citing it in a card.
Items marked "checked" were confirmed in this repo's code or data.

## Agreed across the 4 sources
1. A common market factor dominates crypto panels. Residual targets (beta-neutral vs BTC, an equal-weight index or PC1, with rolling past-only betas) or cross-sectional rank targets would add the most effective information. Caveat: this changes the question from absolute to relative moves, and recombining them needs a separate market forecast. **Waits for the owner's decision.**
2. NIG head collapse is a model or optimization problem, not proof that the data holds no signal (Wu et al. AAAI 2024 "evidence contraction"; Oh & Shin AAAI 2022 gradient shrinkage).
3. Strong linear/GBM baselines come first. A deep model is justified only if it beats them under a dependence-aware evaluation (Gu, Kelly & Xiu 2020 RFS).
4. Lowering the stride or adding synthetic data adds little independent information.
5. Compare models per timestamp: paired loss differential d_t, block bootstrap over whole timestamps plus Newey–West / Driscoll–Kraay. Never bootstrap coin-time rows.

## Effective sample size (checked)
- Kish per timestamp: m/(1+(m−1)ρ̄) = 200/56.7 ≈ 3.5 coins, which gives about 1.4–1.5k samples over about 440 timestamps, before any temporal correction.
- The "47" printed by the health report came from a unit bug in `workflow/capacity.py`: the span was measured in sample spacing but divided by the window in bars. Fixed in PR #45. The corrected figure is about 1.5k. No temporal design-effect term is applied yet. It should be estimated per target and for d_t (T2).

## High/low direction labels (checked, T4)
- `y_high_class` = next high > last HIGH, and `y_low_class` = next low > last LOW (`data/windows.py:342`, each compared with its own type).
- A trivial rule (close position in the last bar's range) scores AUC 0.76–0.83 on a pure random walk and 0.75–0.78 on real BTC/ETH/SOL 4h data. So the model's 0.74–0.78 is range mechanics, not price-direction information.
- Under `logret_std` the reg targets use last close as the anchor, while the class labels use the same-type anchor. The two heads therefore disagree on the reference.

## NIG / deep evidential regression (input for T3)
- Floors ν=0.1, α=2, β=0.01 give V = β(1+ν)/(ν(α−1)) = 0.11, so σ ≈ 0.33. α_min=2 forces Student-t df ≥ 4 (DER only needs α > 1).
- A negative NLL is not an error (a density can exceed 1).
- Kendall weighting e^{-s}·L + s is unbounded below when L < 0, so it must never apply to the NIG loss (verify in code).
- The regularizer λ|y−γ|(2ν+α) pushes ν and α down, but not β. If all three sit at the floor, also check the parameterization, init and target scale.
- Hard clamps give zero gradient below the floor. Prefer ν = ε + softplus, α = 1 + softplus, β = ε + softplus.
- β is in target units². Init β0 = s_e²·ν0(α0−1)/(ν0+1); with ν0=1, α0=2.5, s_e²=1 this gives β0 = 0.75, not 0.01.
- Wu et al. L_U = (y−γ)²·ν(α−1)/(β(ν+1)), with the gradient of the uncertainty factor stopped. A starting sweep of η2 ∈ {1e-4, 1e-3, 1e-2} (not from the paper).
- Ablation order: audit → fixed task weights → λ sweep {0, .001, .003, .01, .03, .1} → smooth parameterization + β init → MSE/Huber head or L_U. Record the floor-hit share per parameter and the raw-logit percentiles.

## Other leads (later, separate cards)
- Market-context inputs (BTC return/vol, breadth, dispersion) without changing the target.
- Volume/OHLC structure features and multi-horizon context. RevIN ablation (instance norm may remove the volatility level; see E-norm-002).
- Purged/embargoed walk-forward and uniqueness weights matter mainly when label windows overlap (not the case at stride 32 with H=1).
- Multi-horizon range targets (T5 / E-tgt-001).

## References: verify before citing
- Classic or likely real: Kish (design effect); Driscoll & Kraay 1998; Newey–West; Politis & Romano 1994; López de Prado 2018 (AFML); Gu, Kelly & Xiu 2020 RFS; Amini et al. NeurIPS 2020; Kendall, Gal & Cipolla CVPR 2018; Wu et al. AAAI 2024; Oh & Shin AAAI 2022; Kim et al. RevIN ICLR 2022.
- Unverified or possibly hallucinated: anything dated 2026 (Liu et al. JFE 2026, Dezons 2026, Li 2026 PBFJ, Zhang 2026 npj AI), "Cakici 2024", "Berger 2024", "Dudek 2024".
