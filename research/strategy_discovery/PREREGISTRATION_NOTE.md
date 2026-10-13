# Pre-registration note for PR #15 (strategy discovery, indicator strategy, hedge recovery)

## Statement
Every experiment card in this PR was added to the repository in the SAME commit as its results. The git history therefore
cannot show that a card existed before its run. Each card says "written and committed BEFORE the run", but that claim cannot
be verified from the commits: the order of writing and running is known only from the author's account.

Consequence: **all results of this PR are exploration, not confirmation.** Accept/reject rules inside the cards were stated in
advance by the author, but no reviewer can check this. No result here counts as evidence that satisfies a pre-registered bar.
A confirmation needs a card committed on its own, before the run, on data that was not used here (see
`docs/research/plan/ROADMAP.md` section R5, holdout bucket H4).

## Cards and the commit that introduced each one together with its results
| Card | Commit (card and results together) | Subject |
|---|---|---|
| `research/hedge_recovery/cards/E-hedge-001.md` | `931690d` | exp: E-hedge-001 and E-hedge-002 hedge-recovery rules, both rejected |
| `research/hedge_recovery/cards/E-hedge-002.md` | `931690d` | same commit |
| `research/indicator_strategy/cards/E-ind-001.md` | `6b1acf7` | exp: indicator screen and daily breakout refinement, E-ind-001 to E-ind-003 |
| `research/indicator_strategy/cards/E-ind-002.md` | `6b1acf7` | same commit |
| `research/indicator_strategy/cards/E-ind-003.md` | `6b1acf7` | same commit (the TEST run "once" is likewise unverifiable) |
| `research/strategy_discovery/cards/E-disc-001.md` | `66e02d5` | exp: strategy discovery campaign E-disc-001 to E-disc-006 |
| `research/strategy_discovery/cards/E-disc-002.md` | `66e02d5` | same commit |
| `research/strategy_discovery/cards/E-disc-003.md` | `66e02d5` | same commit |
| `research/strategy_discovery/cards/E-disc-004.md` | `66e02d5` | same commit |
| `research/strategy_discovery/cards/E-disc-005.md` | `66e02d5` | same commit |
| `research/strategy_discovery/cards/E-disc-006.md` | `66e02d5` | same commit |
| `research/strategy_discovery/cards/E-srch-002.md` | `a68b583` | exp: E-srch-002 MACD momentum on 15m and 1h, rejected |
| `research/strategy_discovery/cards/E-srch-003.md` | `0e07d72` | exp: E-srch-003 15m time-of-day seasonality, rejected |
| `research/strategy_discovery/cards/E-fwd-001.md` | `ae1457d` | exp: E-fwd-001 forward test of one frozen hypothesis |

Verified with `git log --diff-filter=A -- <card>` and `git show --stat <sha>`: in each commit the card and at least one results
file appear together.

## What this changes
- The "candidate" and "accepted" wording in the cards and READMEs means "survived the author's own screen", not "confirmed".
- Multiple-testing corrections inside a card (Benjamini-Hochberg over 60 configurations in E-disc-005) are only as good as the
  count of configurations the author reports; the count itself is unverifiable.
- Later corrections to this PR (significance table for E-disc-003 and E-disc-006, failure rows F-0076 and F-0077) are
  post-hoc by construction and are labelled as such.
- Future cards must be committed alone, before the run, with the run in a later commit.
