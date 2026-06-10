# MACHINE — the AlterEgo search algorithm

**M**etareasoning · **A**nytime · **C**alibrated · **H**ybrid · **I**nformation-directed · **N**ested-opponent · **E**xpectimax

Every letter is a load-bearing component:

| Letter | Component | What it does |
|---|---|---|
| M | Metareasoning | Compute is spent only where its expected value exceeds its time cost; probe fidelity is itself a decision |
| A | Anytime | A best move is always ready; quality degrades gracefully under clock pressure, never correctness |
| C | Calibrated | Probe scalars become posterior WDL belief distributions via a fitted, online-updated calibration model |
| H | Hybrid | Explicit best-first layer (the Alter) over an αβ+NNUE probe oracle (the Ego) — respects all four structural locks |
| I | Information-directed | Expansion targets the node most likely to flip the root decision, weighted by reachability |
| N | Nested-opponent | A configured copy of the target opponent runs inside the engine, predicting its actual replies |
| E | Expectimax | Backups at opponent nodes blend "what they will play" with "what they could play," under a hard risk floor |

## Architecture: two selves and a memory

- **The Ego** — a conventional, fully tuned αβ+NNUE engine (Stockfish-class). It is never modified; its SPRT-tuned heuristic stack, TT economics, and NNUE incrementality stay intact (Locks 1, 2, 4). It runs in two configurations:
  - *Prober*: depth-`d` searches returning `(score, PV, move ranking)`.
  - *Caged opponent*: configured to match the target opponent's settings and time-equivalent depth, returning predicted reply distributions.
- **The Alter** — an explicit best-first DAG of 10⁴–10⁶ nodes with tree reuse across moves. Nodes carry *beliefs*, not values:
  - posterior WDL distribution + concentration (how settled the value is)
  - probe history (which depths have been run)
  - predicted opponent reply distribution (at opponent-to-move nodes)
  - current fidelity level
- **The Memory** — two small models, both trained offline and updated online during play:
  - *Calibration model*: `(probe score, probe depth, position features) → WDL posterior`. Every deep probe that supersedes a shallow one is a free labeled example.
  - *Opponent reliability model*: `λ(depth gap, position class)` — how often the caged copy's prediction matches the real opponent's move. Updated after every actual opponent move.

## The loop

```
loop:
    // SELECT — information-directed descent
    leaf ← argmax over frontier of
             P(resolving this node flips the root choice)        // decision conspiracy
             × reach-probability under (our policy, their model)

    // PROBE — bounded-rational simulation
    f ← fidelity(time_left, expected_information_gain)           // depth ladder d0 < d1 < d2 …
    (score, ranking) ← Ego.search(leaf.pos, depth = f)

    // CALIBRATE — scalar → belief
    leaf.belief ← bayesUpdate(leaf.belief,
                              calibrate(score, f, features(leaf.pos)))

    // PREDICT — consult the caged opponent
    if leaf.toMove = THEM and leaf.replyDist = none:
        leaf.replyDist ← CagedOpponent.predict(leaf.pos)

    // BACKUP — distributional, hard, asymmetric
    for n in path(leaf → root):
        if n.toMove = US:
            n.belief ← distributional MAX over children w.r.t. U_risk
        else:
            n.belief ← λ · E[ child(m).belief  for m ~ n.replyDist ]   // what they WILL do
                     + (1−λ) · distributional MIN over children         // what they COULD do
        // hard constraint, never violated:
        // floor(n.belief) ≥ floor(risk-neutral best) − RISK_BUDGET

until P(root choice changes with further search) < ε    // decision is stable
   or the clock says act                                  // "there is no perfect move"

play argmax over root moves of U_risk(belief)
```

**Utility.** `U_risk = P(win) + γ·P(draw)`, with γ set by match context (must-win games lower γ; the engine presses). Distributional MAX/MIN are hard minimax operators over beliefs — never visit-weighted averaging (Lock 3).

**The risk floor is constitutional.** Trap-seeking (steering toward positions where the opponent's predicted move diverges from belief-optimal play) is only permitted while the worst-case belief stays within `RISK_BUDGET` of the sound alternative. The engine exploits its opponent; it never gambles the position to do so.

## Learning loops

- **Offline (cage matches):** adversarial sparring against the frozen opponent at a curriculum of budgets. Discovered blind-spot motifs become steering priors — a utility bonus for entering position classes where the opponent's judgment is measurably miscalibrated.
- **Online (during the game):** calibration posteriors tighten with every probe pair; λ updates with every observed opponent move; the Alter's subtree and all beliefs persist across moves.
- **Optional Ego upgrade:** a cheap uncertainty/policy head on NNUE features makes SELECT strictly smarter. Pure αβ cannot consume that information at all — this is the asymmetric upgrade path the original cannot follow without becoming MACHINE.

## Why this can beat the original engine

All four structural locks are respected — NNUE incrementality lives inside probes, the explicit tree is small, backups are hard, the tuned heuristic stack is untouched — while adding four things scalar αβ cannot represent:

1. **Global allocation** — budget pours into the contested lines instead of a rigid depth frontier
2. **Uncertainty** — settled lines stop consuming compute; unsettled ones get drilled
3. **Opponent asymmetry** — it plays the opponent in front of it, not a worst-case abstraction
4. **Risk shaping** — among equal-scalar moves it systematically picks the one whose distribution is fat-tailed in its favor, along lines the real opponent will actually enter

Expected edge concentrates at long time controls, high core counts (near-linear probe parallelism vs. lazy-SMP saturation), unbalanced openings, and fixed-node testing. Honest failure modes: a miscalibrated belief model is worse than scalars; top-layer overhead must stay under ~20–30%; opponent predictions from shallower-than-opponent probes need the λ discount to stay honest.

## Provenance

| Component | Origin |
|---|---|
| Ego probes as simulations | The Machine's forward simulations ("If-Then-Else") |
| Fidelity ladder | The placeholder-dialogue degraded simulation |
| Stop rule | Harold: there is no perfect first move — you have to play |
| Caged opponent | The sandboxed Samaritan copy, billions of accelerated battles |
| Repeated probing of one pivotal node | Simulation 6,741 |
| The inviolable risk floor | "People are not pieces" — values as hard constraints, not penalties |

## Build order

1. Bitboards + magic move generation, perft-verified
2. The Ego: αβ + TT + standard pruning + NNUE inference (also serves as the caged opponent)
3. Logging + calibration pipeline (probe pairs → WDL posterior model)
4. The Alter: explicit DAG, distributional backups, information-directed SELECT, stop rule
5. Cage-match training loop and λ estimation
6. SPRT harness; fixed-node and fixed-time regression vs. the unmodified Ego
