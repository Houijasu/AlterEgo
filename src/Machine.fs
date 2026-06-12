module AlterEgo.Machine

// MACHINE rung 4 — the Alter layer over the Ego (see docs/MACHINE.md).
//
// Phase 1 (85% of budget): one shared iterative-deepening pass — the Ego at
// full strength (SMP-aware). Phase 2 (15%): root verification. Screening is
// FREE: the seed's transposition table already holds values for every root
// child, so contenders are identified by TT lookup, not fresh searches. Only
// the top contenders get a deep verification probe, and the seeded leader
// stands unless a challenger beats it by a clear margin at comparable depth.
// Hard minimax values throughout — never averaged, never outranked by
// shallower evidence.

open System.Threading
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

type Arm =
    { Move: Move
      mutable Cp: int        // latest value, root-stm POV
      mutable Depth: int     // depth of the evidence behind Cp
      mutable Verified: bool } // Cp is exact (seed/full-window probe), not a TT-screen bound

// ---- WDL-aware root selection (opt-in: ALTEREGO_ENABLE=wdlroot) ----
// Static WDL probes of root children add an expected-score utility signal
// U = P(win) + gamma * P(draw) (root POV) to the final switch decision.
// Strictly additive: it can promote a challenger inside the cp band
// [wdlfloor, switchMargin) but never vetoes a cp-margin switch — per
// doctrine, a depth-0 probe must not outrank depth-(idDepth-1) evidence.
let private useWdlRoot = SearchConfig.optIn "wdlroot"
let private wdlGamma = float32 (SearchConfig.tuned "wdlgamma" 500) / 1000.0f   // draw weight in utility (500 = expected score)
let private wdlMargin = float32 (SearchConfig.tuned "wdlmargin" 25) / 1000.0f  // utility switch margin (~20cp near equality)
let private wdlFloor = SearchConfig.tuned "wdlfloor" 0                         // challenger must not be cp-worse than this

let private probeAt (pos: Position) (st: State) (arm: Arm) (d: int) =
    st.ContIdx.[1] <- pos.Mailbox.[moveFrom arm.Move] * 64 + moveTo arm.Move
    makeMove pos arm.Move
    let v = -(search pos st d 1 (-Infinity) Infinity true)
    unmakeMove pos arm.Move
    if not st.Stop.Value then
        arm.Cp <- v
        arm.Depth <- d
        arm.Verified <- true

/// MACHINE move selection under a millisecond budget.
let think (pos: Position) (st: State) (budgetMs: int64) (verbose: bool) : Move =
    let swTotal = System.Diagnostics.Stopwatch.StartNew()

    let moves = Array.zeroCreate<Move> 256
    let n = generate pos moves
    let legal = ResizeArray<Move>()
    for i in 0 .. n - 1 do
        makeMove pos moves.[i]
        if wasLegal pos then legal.Add moves.[i]
        unmakeMove pos moves.[i]

    if legal.Count = 0 then NoMove
    elif legal.Count = 1 then legal.[0]
    else
        // ---- Phase 1: the seed gets the FULL budget (identical to raw Ego) ----
        // The Ego's soft-stop ends early rather than start an unfinishable depth;
        // Phase 2 runs purely on that leftover time — verification is free.
        // verbose passes through: GUIs see live per-depth lines during timed play
        let idMove = AlterEgo.Search.think pos st { defaultLimits with MoveTimeMs = budgetMs } verbose
        let idScore = st.BestScore
        let idDepth = st.CompletedDepth
        let seedNodes = st.Nodes

        let arms = ResizeArray<Arm>()
        let mutable leader: Arm = Unchecked.defaultof<Arm>
        for m in legal do
            let arm = { Move = m; Cp = -Infinity; Depth = 0; Verified = false }
            if m = idMove then
                arm.Cp <- idScore
                arm.Depth <- max 1 idDepth
                arm.Verified <- true   // full-budget seed = exact evidence
                leader <- arm
            arms.Add arm

        let remaining = budgetMs - swTotal.ElapsedMilliseconds - 20L
        // an external stop/quit during the seed must end the move NOW — only the
        // soft-stop may fund Phase 2 (Stop is set by both; Abort disambiguates)
        if st.Abort.Value || remaining < 50L || obj.ReferenceEquals(leader, null) || idDepth < 4 then
            if verbose then
                printfn "info string machine verdict %s (seed only) depth %d cp %d"
                    (moveToUci idMove) idDepth idScore
            idMove
        else
            st.Stop.Value <- false
            st.Sw.Restart()
            st.HardMs <- max 1L remaining
            st.NodeLimit <- 0UL

            // ---- free screen: the seed's TT already valued every root child ----
            for a in arms do
                if not (obj.ReferenceEquals(a, leader)) then
                    makeMove pos a.Move
                    let tte = st.Tt.Probe pos.Key
                    if tte.Hit then
                        a.Cp <- -(scoreFromTt tte.Score 1)
                        a.Depth <- tte.Depth
                    unmakeMove pos a.Move

            // ---- verify: deep probes for the top TT-screened contenders only ----
            let screenMargin = 40
            let switchMargin = 20
            let contenders =
                arms
                |> Seq.filter (fun a ->
                    not (obj.ReferenceEquals(a, leader))
                    && a.Depth > 0
                    && a.Cp >= leader.Cp - screenMargin)
                |> Seq.sortByDescending (fun a -> a.Cp)
                |> Seq.truncate 2
                |> Seq.toArray

            if contenders.Length > 0 then
                ensureHelpers st
                let lanes = max 1 (min st.ThreadCount (min contenders.Length (st.HelperStates.Length + 1)))
                let lanePos i = if i = 0 then pos else st.HelperPositions.[i - 1]
                let laneSt i = if i = 0 then st else st.HelperStates.[i - 1]
                for i in 1 .. lanes - 1 do
                    let hs = st.HelperStates.[i - 1]
                    hs.Nodes <- 0UL
                    hs.HardMs <- max 1L remaining
                    hs.NodeLimit <- 0UL
                    hs.Sw.Restart()
                    copyInto pos st.HelperPositions.[i - 1]
                let verifyDepth = max 1 (idDepth - 1)
                let threads =
                    [| for li in 1 .. lanes - 1 ->
                         let t = Thread((fun () ->
                             try probeAt (lanePos li) (laneSt li) contenders.[li] verifyDepth
                             with ex -> logCrash "machine lane" ex), 16 * 1024 * 1024)
                         t.IsBackground <- true
                         t.Start()
                         t |]
                probeAt (lanePos 0) (laneSt 0) contenders.[0] verifyDepth
                for t in threads do t.Join(int remaining + 2000) |> ignore
                if verbose then
                    for c in contenders do
                        printfn "info string machine verify %s depth %d cp %d (leader %s cp %d)"
                            (moveToUci c.Move) c.Depth c.Cp (moveToUci leader.Move) leader.Cp

            // Decision: the seeded leader stands unless a challenger beats it by a
            // clear margin at comparable depth. Shallow evidence never outranks deep.
            // With wdlroot, a WDL-utility lead may also promote a cp-comparable arm.
            let mutable chosen = leader
            // (utility, win, draw, loss) of the root child, root POV; the child
            // probe is opponent POV so win/loss flip. Mate-band cp stays scalar.
            // only verified arms qualify: TT-screen cp values are bounds, often
            // optimistic (root fail-lows) — letting them through with a 0cp floor
            // was a -575 Elo disaster in the first 128g gate
            let probeU (a: Arm) =
                if not useWdlRoot || not a.Verified || abs a.Cp >= 27000 then ValueNone
                else
                    makeMove pos a.Move
                    let r = AlterEgo.Nnue.evaluateWdl pos.AccStack.[pos.Ply] pos.Stm
                    unmakeMove pos a.Move
                    match r with
                    | ValueSome (struct (cw, cd, cl)) ->
                        ValueSome (struct (cl + wdlGamma * cd, cl, cd, cw))
                    | ValueNone -> ValueNone
            let mutable chosenU = probeU chosen
            if verbose && useWdlRoot then
                match chosenU with
                | ValueSome (struct (u, w, d, l)) ->
                    printfn "info string machine wdl %s cp %d w %.3f d %.3f l %.3f u %.3f (leader)"
                        (moveToUci chosen.Move) chosen.Cp (float w) (float d) (float l) (float u)
                | ValueNone -> ()
            for a in arms do
                if not (obj.ReferenceEquals(a, chosen)) && a.Depth >= idDepth - 1 then
                    let aU = probeU a
                    if verbose && useWdlRoot then
                        match aU with
                        | ValueSome (struct (u, w, d, l)) ->
                            printfn "info string machine wdl %s cp %d w %.3f d %.3f l %.3f u %.3f"
                                (moveToUci a.Move) a.Cp (float w) (float d) (float l) (float u)
                        | ValueNone -> ()
                    let byCp = a.Cp >= chosen.Cp + switchMargin
                    let byWdl =
                        match chosenU, aU with
                        | ValueSome (struct (uc, _, _, _)), ValueSome (struct (ua, _, _, _)) ->
                            ua >= uc + wdlMargin && a.Cp >= chosen.Cp + wdlFloor
                        | _ -> false
                    if byCp || byWdl then
                        if verbose && useWdlRoot then
                            printfn "info string machine switch %s -> %s (%s)"
                                (moveToUci chosen.Move) (moveToUci a.Move)
                                (if byCp then "cp" else "wdl")
                        chosen <- a
                        chosenU <- aU

            if verbose then
                let ms = max 1L swTotal.ElapsedMilliseconds
                let mutable nodes = st.Nodes
                for h in st.HelperStates do nodes <- nodes + h.Nodes
                printfn "info depth %d score cp %d%s nodes %d nps %d time %d pv %s"
                    chosen.Depth chosen.Cp (AlterEgo.Search.wdlInfoString pos)
                    nodes (nodes * 1000UL / uint64 ms) ms (moveToUci chosen.Move)
                printfn "info string machine verdict %s depth %d cp %d (seed %s d%d cp %d) contenders %d"
                    (moveToUci chosen.Move) chosen.Depth chosen.Cp
                    (moveToUci idMove) idDepth idScore
                    (arms |> Seq.filter (fun a -> a.Depth > 0 && not (obj.ReferenceEquals(a, leader)) && a.Cp >= leader.Cp - screenMargin) |> Seq.length)
            chosen.Move
