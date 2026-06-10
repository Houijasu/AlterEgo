# Stockfish Ladder Anchors

Real-opponent strength measurements vs Stockfish 18 (`UCI_LimitStrength`),
16 games per rung, 500ms/move, both engines single-thread, 8 parallel lanes.

## Anchor 1 — 2026-06-10

Config: commit `c969ab3`+P0, embedded net4, default search (all experimental
features off, baseline-exact bench 276,598).

| SF UCI_Elo | Result | Score | Implied AlterEgo Elo |
|---|---|---|---|
| 2000 | +16 =0 −0 | 100% | ≫ |
| 2300 | +13 =2 −1 | 87.5% | ~2638 |
| 2600 | +11 =1 −4 | 71.9% | ~2763 |
| 2900 | +4 =2 −10 | 31.2% | ~2763 |
| 3190 (max) | +0 =1 −15 | 3.1% | ~2593 |

**Anchor: ≈ 2760 ± 50 (SF UCI_Elo scale)** — the 2600/2900 rungs agree to the point.

Caveats: SF's UCI_Elo calibration assumes its own conditions; 16-game rungs are
coarse (±~100); full-strength SF18 (no limiter) remains far beyond the scale top.

Distance to goal: full Stockfish ≈ 3640+ CCRL ⇒ ~900+ Elo to close.
