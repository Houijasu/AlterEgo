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
      mutable Depth: int }   // depth of the evidence behind Cp

let private probeAt (pos: Position) (st: State) (arm: Arm) (d: int) =
    st.ContIdx.[1] <- pos.Mailbox.[moveFrom arm.Move] * 64 + moveTo arm.Move
    makeMove pos arm.Move
    let v = -(search pos st d 1 (-Infinity) Infinity true)
    unmakeMove pos arm.Move
    if not st.Stop.Value then
        arm.Cp <- v
        arm.Depth <- d

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
        let idMove = AlterEgo.Search.think pos st { defaultLimits with MoveTimeMs = budgetMs } false
        let idScore = st.BestScore
        let idDepth = st.CompletedDepth
        let seedNodes = st.Nodes

        let arms = ResizeArray<Arm>()
        let mutable leader: Arm = Unchecked.defaultof<Arm>
        for m in legal do
            let arm = { Move = m; Cp = -Infinity; Depth = 0 }
            if m = idMove then
                arm.Cp <- idScore
                arm.Depth <- max 1 idDepth
                leader <- arm
            arms.Add arm

        let remaining = budgetMs - swTotal.ElapsedMilliseconds - 20L
        if remaining < 50L || obj.ReferenceEquals(leader, null) || idDepth < 4 then
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
            let mutable chosen = leader
            for a in arms do
                if a.Depth >= idDepth - 1 && a.Cp >= chosen.Cp + switchMargin then chosen <- a

            if verbose then
                let ms = max 1L swTotal.ElapsedMilliseconds
                let mutable nodes = st.Nodes
                for h in st.HelperStates do nodes <- nodes + h.Nodes
                printfn "info depth %d score cp %d nodes %d nps %d time %d pv %s"
                    chosen.Depth chosen.Cp nodes (nodes * 1000UL / uint64 ms) ms (moveToUci chosen.Move)
                printfn "info string machine verdict %s depth %d cp %d (seed %s d%d cp %d) contenders %d"
                    (moveToUci chosen.Move) chosen.Depth chosen.Cp
                    (moveToUci idMove) idDepth idScore
                    (arms |> Seq.filter (fun a -> a.Depth > 0 && not (obj.ReferenceEquals(a, leader)) && a.Cp >= leader.Cp - screenMargin) |> Seq.length)
            chosen.Move
