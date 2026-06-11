module AlterEgo.Uci

open System
open System.Threading
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

let private applyUciMove (pos: Position) (token: string) =
    let buf = Array.zeroCreate<Move> 256
    let n = generate pos buf
    let mutable ok = false
    let mutable i = 0
    while not ok && i < n do
        if moveToUci buf.[i] = token then
            makeMove pos buf.[i]
            if wasLegal pos then ok <- true else unmakeMove pos buf.[i]
        i <- i + 1
    ok

let private parsePosition (pos: Position) (tokens: string[]) =
    let movesIdx = Array.tryFindIndex ((=) "moves") tokens
    if tokens.Length > 1 && tokens.[1] = "fen" then
        let stop = defaultArg movesIdx tokens.Length
        let fen = String.Join(" ", tokens.[2 .. stop - 1])
        setFen pos fen
    else
        setFen pos StartFen
    match movesIdx with
    | Some idx ->
        for i in idx + 1 .. tokens.Length - 1 do
            applyUciMove pos tokens.[i] |> ignore
    | None -> ()

let private parseGo (pos: Position) (tokens: string[]) =
    let mutable limits = defaultLimits
    let value (i: int) = if i + 1 < tokens.Length then int64 tokens.[i + 1] else 0L
    let mutable i = 1
    while i < tokens.Length do
        match tokens.[i] with
        | "depth" -> limits <- { limits with Depth = int (value i) }; i <- i + 1
        | "movetime" -> limits <- { limits with MoveTimeMs = value i }; i <- i + 1
        | "nodes" -> limits <- { limits with NodeLimit = uint64 (value i) }; i <- i + 1
        | "movestogo" -> limits <- { limits with MovesToGo = int (value i) }; i <- i + 1
        | "wtime" -> (if pos.Stm = White then limits <- { limits with TimeMs = value i }); i <- i + 1
        | "btime" -> (if pos.Stm = Black then limits <- { limits with TimeMs = value i }); i <- i + 1
        | "winc" -> (if pos.Stm = White then limits <- { limits with IncMs = value i }); i <- i + 1
        | "binc" -> (if pos.Stm = Black then limits <- { limits with IncMs = value i }); i <- i + 1
        | "infinite" -> limits <- { limits with Infinite = true }
        | _ -> ()
        i <- i + 1
    limits

let run () =
    AlterEgo.Magics.init ()
    // zero-configuration NNUE: the embedded default net loads unless one is already active
    if not AlterEgo.Nnue.active then
        if AlterEgo.Nnue.loadEmbedded () then
            printfn "info string NNUE: embedded default network loaded"
        else
            printfn "info string NNUE: no embedded network — PST evaluation"
    let pos = fromFen StartFen
    let mutable st = createState 64
    let mutable worker: Thread = null

    let stopSearch () =
        st.Stop.Value <- true
        if worker <> null && worker.IsAlive then worker.Join()

    let mutable running = true
    while running do
        let line = Console.ReadLine()
        if line = null then
            // EOF: let any running search finish (it owns bestmove output), then exit
            if worker <> null && worker.IsAlive then worker.Join()
            running <- false
        else
            let tokens = line.Split(' ') |> Array.filter (fun t -> t <> "")
            if tokens.Length > 0 then
                match tokens.[0] with
                | "uci" ->
                    printfn "id name AlterEgo 0.3"
                    printfn "id author Houijasu"
                    printfn "option name Hash type spin default 64 min 1 max 8192"
                    printfn "option name Threads type spin default 1 min 1 max 256"
                    printfn "option name EvalFile type string default <embedded>"
                    printfn "uciok"
                | "isready" -> printfn "readyok"
                | "setoption" ->
                    // setoption name Hash value N
                    stopSearch ()
                    let nameIdx = Array.tryFindIndex ((=) "name") tokens
                    let valueIdx = Array.tryFindIndex ((=) "value") tokens
                    match nameIdx, valueIdx with
                    | Some ni, Some vi when ni + 1 < tokens.Length && vi + 1 < tokens.Length ->
                        match tokens.[ni + 1].ToLowerInvariant() with
                        | "hash" ->
                            let mb = max 1 (min 8192 (int tokens.[vi + 1]))
                            st.Tt <- AlterEgo.TT.Table(mb)
                            AlterEgo.Search.ensureHelpers st
                        | "threads" ->
                            st.ThreadCount <- max 1 (min 256 (int tokens.[vi + 1]))
                            AlterEgo.Search.ensureHelpers st
                        | "evalfile" ->
                            let path = String.Join(" ", tokens.[vi + 1 ..])
                            if AlterEgo.Nnue.load path then
                                // sync accumulators with the position that predates the load
                                AlterEgo.Nnue.buildInto pos.AccStack.[pos.Ply] pos.ByPiece
                                printfn "info string NNUE loaded: %s" path
                            else
                                printfn "info string NNUE load FAILED: %s" path
                        | _ -> ()
                    | _ -> ()
                | "ucinewgame" ->
                    stopSearch ()
                    st.Tt.Clear()
                    Array.Clear(st.History, 0, st.History.Length)
                    Array.Clear(st.CorrHist, 0, st.CorrHist.Length)
                    Array.Clear(st.ContHist, 0, st.ContHist.Length)
                    Array.Clear(st.CaptHist, 0, st.CaptHist.Length)
                | "position" ->
                    stopSearch ()
                    try
                        parsePosition pos tokens
                    with ex ->
                        // malformed FEN/moves must not kill the engine or leave
                        // a half-mutated position behind
                        logCrash "position parse" ex
                        setFen pos StartFen
                        printfn "info string invalid position command — reset to startpos"
                | "go" ->
                    stopSearch ()
                    // reset on the command thread BEFORE the worker exists: a later
                    // "stop" can then never be overwritten by the search thread
                    st.Stop.Value <- false
                    let limits = parseGo pos tokens
                    // MACHINE is the engine's algorithm (hardwired). It allocates a
                    // millisecond budget; budgetless modes (depth/nodes/infinite,
                    // e.g. GUI analysis) run the Ego core directly — MACHINE's
                    // allocation is meaningless without a clock.
                    let machineBudget =
                        if limits.MoveTimeMs > 0L then limits.MoveTimeMs
                        elif limits.TimeMs > 0L then
                            let mtg = if limits.MovesToGo > 0 then int64 limits.MovesToGo + 1L else 32L
                            limits.TimeMs / mtg + limits.IncMs / 2L
                        else 0L
                    worker <- Thread((fun () ->
                        try
                            let bm =
                                if machineBudget > 0L then AlterEgo.Machine.think pos st machineBudget true
                                else think pos st limits true
                            printfn "bestmove %s" (moveToUci bm)
                        with ex ->
                            // a crashed search must not kill the engine process
                            logCrash "search worker" ex
                            printfn "info string search error: %s (see alterego-crash.log)" ex.Message
                            printfn "bestmove 0000"), 16 * 1024 * 1024)
                    worker.IsBackground <- true
                    worker.Start()
                | "stop" -> stopSearch ()
                | "d" ->
                    print pos
                    let nn =
                        if AlterEgo.Nnue.active
                        then sprintf "%+d" (AlterEgo.Nnue.evaluateAcc pos.AccStack.[pos.Ply] pos.Stm)
                        else "off"
                    pos.ForcePst <- true
                    let pst = AlterEgo.Eval.evaluate pos
                    pos.ForcePst <- false
                    printfn "eval nnue %s  pst %+d (stm POV)" nn pst
                | "quit" ->
                    stopSearch ()
                    running <- false
                | _ -> ()
