module AlterEgo.Machine

// MACHINE rung 2 — the Alter layer over the Ego (see docs/MACHINE.md).
//
// Hybrid budget: Phase 1 runs one shared iterative-deepening pass (the Ego at
// full strength — with lazy SMP when Threads > 1). Phase 2 spends the remaining
// budget information-directed: screen every other root move with one cheap
// warm-TT probe, then deep-verify the top contenders; switch only on a clear
// margin at comparable depth. With Threads > 1, screening and verification
// probes run on parallel lanes (per-lane Position + State, shared TT + stop).

open System.Threading
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

type Arm =
    { Move: Move
      mutable Cp: int        // latest probe value, root-stm POV
      mutable Depth: int }   // deepest completed probe

let private probeAt (pos: Position) (st: State) (arm: Arm) (d: int) =
    st.ContIdx.[1] <- pos.Mailbox.[moveFrom arm.Move] * 64 + moveTo arm.Move
    makeMove pos arm.Move
    let v = -(search pos st d 1 (-Infinity) Infinity true)
    unmakeMove pos arm.Move
    if not st.Stop.Value then
        arm.Cp <- v
        arm.Depth <- d
        true
    else false

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
        // ---- Phase 1: shared iterative deepening (the Ego seed, SMP-aware) ----
        let seedMs = max 1L (budgetMs * 7L / 10L)
        let idMove = AlterEgo.Search.think pos st { defaultLimits with MoveTimeMs = seedMs } false
        let idScore = st.BestScore
        let idDepth = st.CompletedDepth
        let seedNodes = st.Nodes

        // ---- Phase 2: information-directed verification of contenders ----
        let arms = ResizeArray<Arm>()
        let mutable leader: Arm = Unchecked.defaultof<Arm>
        for m in legal do
            let arm = { Move = m; Cp = 0; Depth = 0 }
            if m = idMove then
                arm.Cp <- idScore
                arm.Depth <- max 1 idDepth
                leader <- arm
            arms.Add arm

        let remaining = budgetMs - swTotal.ElapsedMilliseconds - 20L
        if remaining < 30L || obj.ReferenceEquals(leader, null) then
            if verbose then
                printfn "info string machine verdict %s (seed only) depth %d cp %d"
                    (moveToUci idMove) idDepth idScore
            idMove
        else
            st.Nodes <- seedNodes
            st.Stop.Value <- false
            st.Sw.Restart()
            st.HardMs <- max 1L remaining
            st.NodeLimit <- 0UL

            // lane resources: lane 0 = main st/pos, lanes 1.. = cached helpers
            ensureHelpers st
            let lanes = max 1 (min st.ThreadCount (st.HelperStates.Length + 1))
            let lanePos i = if i = 0 then pos else st.HelperPositions.[i - 1]
            let laneSt i = if i = 0 then st else st.HelperStates.[i - 1]
            for i in 1 .. lanes - 1 do
                let hs = st.HelperStates.[i - 1]
                hs.Nodes <- 0UL
                hs.HardMs <- max 1L remaining
                hs.NodeLimit <- 0UL
                hs.Sw.Restart()
                copyInto pos st.HelperPositions.[i - 1]

            /// run f over indices [0..count-1] strided across the lanes
            let parallelOver (count: int) (f: int -> int -> unit) =
                if lanes <= 1 || count <= 1 then
                    for k in 0 .. count - 1 do f 0 k
                else
                    let threads =
                        [| for li in 1 .. lanes - 1 ->
                             let t = Thread((fun () ->
                                 try
                                     let mutable k = li
                                     while k < count do
                                         f li k
                                         k <- k + lanes
                                 with ex -> logCrash "machine lane" ex), 16 * 1024 * 1024)
                             t.IsBackground <- true
                             t.Start()
                             t |]
                    let mutable k = 0
                    while k < count do
                        f 0 k
                        k <- k + lanes
                    for t in threads do t.Join(int remaining + 2000) |> ignore

            // ---- screen: one cheap warm-TT probe per non-leader arm, in parallel ----
            let screenDepth = max 5 (idDepth - 6)
            let others = arms |> Seq.filter (fun a -> not (obj.ReferenceEquals(a, leader))) |> Seq.toArray
            parallelOver others.Length (fun li k ->
                probeAt (lanePos li) (laneSt li) others.[k] screenDepth |> ignore)

            // ---- verify: deepen the top contenders near the leader, in parallel ----
            let screenMargin = 60
            let switchMargin = 20
            let contenders =
                others
                |> Array.filter (fun a -> a.Depth > 0 && a.Cp >= leader.Cp - screenMargin)
                |> Array.sortByDescending (fun a -> a.Cp)
                |> Array.truncate (max 2 (lanes - 1))
            parallelOver contenders.Length (fun li k ->
                let c = contenders.[k]
                let p = lanePos li
                let s = laneSt li
                probeAt p s c (max 1 (idDepth - 2)) |> ignore
                if not s.Stop.Value && c.Cp > leader.Cp then
                    probeAt p s c idDepth |> ignore
                if verbose && not s.Stop.Value then
                    printfn "info string machine verify %s depth %d cp %d (leader %s cp %d) time %d"
                        (moveToUci c.Move) c.Depth c.Cp (moveToUci leader.Move) leader.Cp
                        swTotal.ElapsedMilliseconds)

            // Decision: the seeded leader stands unless a challenger beats it by a
            // clear margin at comparable depth. Shallow noise never outranks depth.
            let mutable chosen = leader
            for a in arms do
                if a.Depth >= idDepth - 1 && a.Cp >= chosen.Cp + switchMargin then chosen <- a

            if verbose then
                let ms = max 1L swTotal.ElapsedMilliseconds
                let mutable nodes = st.Nodes
                for h in st.HelperStates do nodes <- nodes + h.Nodes
                let nps = nodes * 1000UL / uint64 ms
                printfn "info depth %d score cp %d nodes %d nps %d time %d pv %s"
                    chosen.Depth chosen.Cp nodes nps ms (moveToUci chosen.Move)
                printfn "info string machine verdict %s depth %d cp %d (seed %s d%d cp %d) lanes %d"
                    (moveToUci chosen.Move) chosen.Depth chosen.Cp
                    (moveToUci idMove) idDepth idScore lanes
            chosen.Move
