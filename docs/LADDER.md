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

## Anchor 2 — 2026-06-10 (net7 promoted)

net7 (king-bucketed, 10.7M samples) beat net4 **+60 Elo over 64 games**
(direct same-binary A/B — the precise instrument for deltas). SF rungs with
net7: 71.9% @2600, 15.6% @2900, 15.6% @3190 — scatter ±150 at 16-game
resolution; absolute anchor unchanged within error: **≈2760 ± 100**.
Ladder rungs need 64 games each to resolve deltas under ~100 Elo.

## Anchor 3 — 2026-06-10 (net7 + singular extensions promoted)

Promoted package (commit `41a8745`): 78.1% @2600 (~2821), 15.6% @2900 (~2607).
Direct A/B evidence behind the package: net7 +60 (64g), singular +38 (128g).
Absolute anchor: **≈2760–2820 ± 100** at 16-game rung resolution.

## Smoke — 2026-06-11 (LMP promoted, lmpbase=2)

Direct A/B evidence: LMP +58 (128g), lmpbase=2 +24 over lmpbase=3 (128g).
SF smoke rungs: 62.5% @2600, 21.9% @2900 — within rung noise of Anchor 3.
Cumulative direct-A/B gains since first anchor ≈ +140; ladder confirmation
needs 64-game rungs (self-play gains also typically compress vs external
opponents). Bench signature with full stack: 367,191 nodes.
