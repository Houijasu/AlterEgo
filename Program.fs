module AlterEgo.Program

open System.Diagnostics
open AlterEgo.Position
open AlterEgo.Perft

[<Literal>]
let Kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

/// (name, fen, [(depth, expected)])
let perftSuite =
    [ "startpos", StartFen,
        [ 1, 20UL; 2, 400UL; 3, 8902UL; 4, 197281UL; 5, 4865609UL; 6, 119060324UL ]
      "kiwipete", Kiwipete,
        [ 1, 48UL; 2, 2039UL; 3, 97862UL; 4, 4085603UL; 5, 193690690UL ]
      "pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        [ 1, 14UL; 2, 191UL; 3, 2812UL; 4, 43238UL; 5, 674624UL; 6, 11030083UL ]
      "pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [ 1, 6UL; 2, 264UL; 3, 9467UL; 4, 422333UL; 5, 15833292UL ]
      "pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [ 1, 44UL; 2, 1486UL; 3, 62379UL; 4, 2103487UL; 5, 89941194UL ]
      "pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [ 1, 46UL; 2, 2079UL; 3, 89890UL; 4, 3894594UL; 5, 164075551UL ] ]

let runSuite (maxDepth: int) =
    AlterEgo.Magics.init ()
    let sw = Stopwatch.StartNew()
    let mutable totalNodes = 0UL
    let mutable failures = 0
    for (name, fen, expectations) in perftSuite do
        let pos = fromFen fen
        for (depth, expected) in expectations do
            if depth <= maxDepth then
                let got = perftRoot pos depth
                totalNodes <- totalNodes + got
                if got = expected then
                    printfn "PASS  %-9s d%d  %d" name depth got
                else
                    failures <- failures + 1
                    printfn "FAIL  %-9s d%d  got %d, expected %d" name depth got expected
        // incremental accumulators must survive the whole make/unmake walk bit-exactly
        let fresh = fromFen fen
        if pos.Mg <> fresh.Mg || pos.Eg <> fresh.Eg || pos.Phase <> fresh.Phase || pos.Key <> fresh.Key then
            failures <- failures + 1
            printfn "FAIL  %-9s accumulator drift (mg %d/%d eg %d/%d phase %d/%d)"
                name pos.Mg fresh.Mg pos.Eg fresh.Eg pos.Phase fresh.Phase
    sw.Stop()
    let nps = float totalNodes / sw.Elapsed.TotalSeconds / 1_000_000.0
    printfn ""
    printfn "%d nodes in %.2fs (%.1f Mnps)%s"
        totalNodes sw.Elapsed.TotalSeconds nps
        (if failures = 0 then " — ALL PASS" else sprintf " — %d FAILURES" failures)
    if failures > 0 then 1 else 0

let private dispatch argv =
    match argv with
    | [| "perft"; depth |] ->
        AlterEgo.Magics.init ()
        let pos = fromFen StartFen
        let d = int depth
        let sw = Stopwatch.StartNew()
        let nodes = perftRoot pos d
        sw.Stop()
        printfn "perft(%d) = %d  (%.2fs, %.1f Mnps)" d nodes sw.Elapsed.TotalSeconds
            (float nodes / sw.Elapsed.TotalSeconds / 1_000_000.0)
        0
    | [| "perft"; depth; fen |] ->
        AlterEgo.Magics.init ()
        let pos = fromFen fen
        let nodes = perftRoot pos (int depth)
        printfn "perft(%s) = %d" depth nodes
        0
    | [| "divide"; depth; fen |] ->
        AlterEgo.Magics.init ()
        let pos = fromFen fen
        divide pos (int depth) |> ignore
        0
    | [| "suite" |] -> runSuite 5
    | [| "suite"; "deep" |] -> runSuite 6
    | [| "bench" |] | [| "bench"; _ |] ->
        AlterEgo.Magics.init ()
        let depth = if argv.Length > 1 then int argv.[1] else 9
        let st = AlterEgo.Search.createState 64
        let sw = Stopwatch.StartNew()
        let mutable total = 0UL
        for (name, fen, _) in perftSuite do
            st.Tt.Clear()
            let pos = fromFen fen
            let bm = AlterEgo.Search.think pos st { AlterEgo.Search.defaultLimits with Depth = depth } false
            total <- total + st.Nodes
            printfn "%-9s bestmove %-6s nodes %10d" name (AlterEgo.Types.moveToUci bm) st.Nodes
        sw.Stop()
        printfn ""
        printfn "bench: %d nodes in %.2fs (%.0f knps)" total sw.Elapsed.TotalSeconds
            (float total / sw.Elapsed.TotalSeconds / 1000.0)
        0
    | [| "machine"; ms |] | [| "machine"; ms; _ |] ->
        AlterEgo.Magics.init ()
        let fen = if argv.Length > 2 then argv.[2] else StartFen
        let pos = fromFen fen
        let st = AlterEgo.Search.createState 64
        let bm = AlterEgo.Machine.think pos st (int64 ms) true
        printfn "bestmove %s" (AlterEgo.Types.moveToUci bm)
        0
    | [| "match"; games; ms |] | [| "match"; games; ms; _ |] ->
        AlterEgo.Magics.init ()
        let lanes = if argv.Length > 3 then int argv.[3] else max 1 (System.Environment.ProcessorCount / 2 - 1)
        AlterEgo.Arena.runMatch AlterEgo.Arena.PlayerKind.Machine AlterEgo.Arena.PlayerKind.Ego
            (int games) (int64 ms) 32 lanes
        0
    // datagen <games> <nodes> <out> [net] [lanes]
    | [| "datagen"; games; nodes; out |]
    | [| "datagen"; games; nodes; out; _ |]
    | [| "datagen"; games; nodes; out; _; _ |] ->
        AlterEgo.Magics.init ()
        let defaultLanes = max 1 (System.Environment.ProcessorCount / 2 - 1)
        let netFile, lanes =
            match argv.[4..] with
            | [||] -> None, defaultLanes
            | [| a |] ->
                match System.Int32.TryParse a with
                | true, l -> None, l
                | _ -> Some a, defaultLanes
            | rest -> Some rest.[0], int rest.[1]
        match netFile with
        | Some nf when not (AlterEgo.Nnue.load nf) ->
            printfn "failed to load %s" nf
            1
        | _ ->
            netFile |> Option.iter (printfn "datagen with NNUE %s")
            AlterEgo.Datagen.run (int games) (uint64 nodes) out lanes
            0
    | [| "train"; dataFile; epochs; out |] ->
        AlterEgo.Train.run dataFile (int epochs) out 4
        0
    | [| "train"; dataFile; epochs; out; kb |] ->
        AlterEgo.Train.run dataFile (int epochs) out (int kb)
        0
    | [| "scrub"; inFile; outFile |] ->
        // Recover a datagen file with interior partial records (killed runs):
        // validate each 100-byte record; on corruption, slide byte-by-byte until
        // 3 consecutive valid records confirm re-alignment.
        let data = System.IO.File.ReadAllBytes inFile
        let recordOk (off: int) =
            if off + 100 > data.Length then false
            else
                let stm = data.[off + 96]
                let result = sbyte data.[off + 99]
                let boards = Array.init 12 (fun i -> System.BitConverter.ToUInt64(data, off + i * 8))
                let mutable union = 0UL
                let mutable overlap = false
                for b in boards do
                    if union &&& b <> 0UL then overlap <- true
                    union <- union ||| b
                let pop (b: uint64) = AlterEgo.Bitboards.popcount b
                stm <= 1uy
                && (result = -1y || result = 0y || result = 1y)
                && not overlap
                && pop boards.[5] = 1 && pop boards.[11] = 1                  // one king each
                && pop boards.[0] <= 8 && pop boards.[6] <= 8                 // pawn counts
                && (boards.[0] ||| boards.[6]) &&& 0xFF000000000000FFUL = 0UL // no pawns on rims
                && pop union <= 32
        use out = new System.IO.BinaryWriter(System.IO.File.Create outFile)
        let mutable off = 0
        let mutable kept = 0L
        let mutable skippedBytes = 0L
        while off + 100 <= data.Length do
            if recordOk off then
                out.Write(data, off, 100)
                kept <- kept + 1L
                off <- off + 100
            else
                // slide until 3 consecutive valid records line up
                let mutable o = off + 1
                let mutable found = false
                while not found && o + 300 <= data.Length do
                    if recordOk o && recordOk (o + 100) && recordOk (o + 200) then found <- true
                    else o <- o + 1
                skippedBytes <- skippedBytes + int64 ((if found then o else data.Length) - off)
                off <- if found then o else data.Length
        printfn "scrub: kept %d records, skipped %d bytes -> %s" kept skippedBytes outFile
        0
    | [| "evalcheck"; netFile |] ->
        AlterEgo.Magics.init ()
        if not (AlterEgo.Nnue.load netFile) then printfn "load failed"; 1
        else
            let positions =
                [ "startpos (should be ~0)", StartFen
                  "white +queen (should be ++)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                  "black +queen, white to move (should be --)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
                  "KQK white (should be ++)", "k7/8/8/8/8/8/8/KQ6 w - - 0 1"
                  "kiwipete (sharp, modest)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" ]
            for (label, fen) in positions do
                let pos = fromFen fen
                let nn = AlterEgo.Nnue.evaluateAcc pos.AccStack.[pos.Ply] pos.Stm
                pos.ForcePst <- true
                let pst = AlterEgo.Eval.evaluate pos
                pos.ForcePst <- false
                printfn "%-36s nnue %+5d  pst %+5d" label nn pst
            0
    | [| "nnuetest"; netFile |] ->
        // verify incremental accumulators == full rebuild over a deep make/unmake walk
        AlterEgo.Magics.init ()
        if not (AlterEgo.Nnue.load netFile) then printfn "load failed"; 1
        else
            let mutable checks = 0
            let mutable bad = 0
            let buf = Array.init 8 (fun _ -> Array.zeroCreate<AlterEgo.Types.Move> 256)
            let scratch = Array.zeroCreate<int16> AlterEgo.Nnue.AccSize
            let rec walk (pos: AlterEgo.Position.Position) depth =
                let inc = AlterEgo.Nnue.evaluateAcc pos.AccStack.[pos.Ply] pos.Stm
                AlterEgo.Nnue.buildInto scratch pos.ByPiece
                let reference = AlterEgo.Nnue.evaluateAcc scratch pos.Stm
                checks <- checks + 1
                if inc <> reference then
                    bad <- bad + 1
                    if bad < 5 then printfn "MISMATCH inc %d ref %d at key %016X" inc reference pos.Key
                if depth > 0 then
                    let moves = buf.[depth]
                    let n = AlterEgo.MoveGen.generate pos moves
                    for i in 0 .. n - 1 do
                        AlterEgo.Position.makeMove pos moves.[i]
                        if AlterEgo.MoveGen.wasLegal pos then walk pos (depth - 1)
                        AlterEgo.Position.unmakeMove pos moves.[i]
            for (_, fen, _) in perftSuite do
                walk (fromFen fen) 3
            printfn "%d positions checked, %d mismatches%s" checks bad
                (if bad = 0 then " — INCREMENTAL OK" else " — BROKEN")
            if bad = 0 then 0 else 1
    // cage <games> <ms> <net> <engine> [elo>=500 | lanes<500] [lanes]
    | [| "cage"; games; ms; netFile; sfPath |]
    | [| "cage"; games; ms; netFile; sfPath; _ |]
    | [| "cage"; games; ms; netFile; sfPath; _; _ |] ->
        AlterEgo.Magics.init ()
        if not (AlterEgo.Nnue.load netFile) then printfn "failed to load net"; 1
        else
            let extra = argv.[5..] |> Array.map int
            let elo = extra |> Array.tryFind (fun v -> v >= 500)
            let lanes =
                match extra |> Array.tryFind (fun v -> v < 500) with
                | Some l -> l
                | None -> max 1 (System.Environment.ProcessorCount / 2 - 1)
            let opts =
                [ "Threads", "1"; "Hash", "64" ]
                @ (match elo with
                   | Some e -> [ "UCI_LimitStrength", "true"; "UCI_Elo", string e ]
                   | None -> [])
            AlterEgo.Arena.runGauntlet sfPath opts (int games) (int64 ms) 64 false lanes
            0
    | [| "matchnnue"; games; ms; netFile |] | [| "matchnnue"; games; ms; netFile; _ |] ->
        AlterEgo.Magics.init ()
        if not (AlterEgo.Nnue.load netFile) then
            printfn "failed to load net %s" netFile
            1
        else
            let lanes = if argv.Length > 4 then int argv.[4] else max 1 (System.Environment.ProcessorCount / 2 - 1)
            AlterEgo.Arena.runMatch AlterEgo.Arena.PlayerKind.Ego AlterEgo.Arena.PlayerKind.PstEgo
                (int games) (int64 ms) 32 lanes
            0
    | [||] | [| "uci" |] -> AlterEgo.Uci.run (); 0
    | _ ->
        // never exit on unrecognized arguments — behave like a UCI engine regardless
        printfn "info string unrecognized arguments %s — entering UCI mode" (String.concat " " argv)
        AlterEgo.Uci.run ()
        0

[<EntryPoint>]
let main argv =
    // any unhandled crash on ANY thread leaves evidence before the process dies
    System.AppDomain.CurrentDomain.UnhandledException.Add(fun e ->
        match e.ExceptionObject with
        | :? exn as ex -> AlterEgo.Types.logCrash "unhandled (AppDomain)" ex
        | _ -> ())
    // UTF-8 console on locales whose OEM codepage .NET lacks (e.g. Turkish CP857)
    if not (System.Console.IsInputRedirected) then
        try
            System.Console.InputEncoding <- System.Text.Encoding.UTF8
            System.Console.OutputEncoding <- System.Text.Encoding.UTF8
        with _ -> ()
    try
        dispatch argv
    with ex ->
        AlterEgo.Types.logCrash (sprintf "main, args=%A" argv) ex
        eprintfn "fatal: %s (see alterego-crash.log)" ex.Message
        2
