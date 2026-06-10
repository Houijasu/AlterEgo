module AlterEgo.Datagen

// Self-play training data generation for NNUE (M5).
// Sample record (binary, little-endian): 12 x uint64 bitboards, stm byte,
// score int16 (White POV, cp), result sbyte (White POV: +1/0/-1) = 100 bytes.

open System
open System.IO
open AlterEgo.Types
open AlterEgo.Bitboards
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

// per-lane RNG (parallel datagen) — module rng kept for single-lane back-compat
let private rng = Random(20260610)

let private legalMoves (pos: Position) =
    let buf = Array.zeroCreate<Move> 256
    let n = generate pos buf
    let result = ResizeArray<Move>()
    for i in 0 .. n - 1 do
        makeMove pos buf.[i]
        if wasLegal pos then result.Add buf.[i]
        unmakeMove pos buf.[i]
    result

let private isThreefold (pos: Position) =
    let limit = max 0 (pos.Ply - pos.Half)
    let mutable count = 0
    for i in limit .. pos.Ply - 1 do
        if pos.Undos.[i].Key = pos.Key then count <- count + 1
    count >= 2

let private insufficient (pos: Position) =
    let minors =
        pos.ByPiece.[Knight] ||| pos.ByPiece.[Bishop]
        ||| pos.ByPiece.[6 + Knight] ||| pos.ByPiece.[6 + Bishop]
    let kings = pos.ByPiece.[King] ||| pos.ByPiece.[6 + King]
    occupancy pos = (kings ||| minors) && popcount minors <= 1

[<Struct>]
type private Sample =
    { Boards: uint64[]
      Stm: byte
      ScoreWhite: int16 }

/// Play one self-play game at fixed nodes per move; returns samples + White result
let private playGame (st: State) (rng: Random) (nodesPerMove: uint64) =
    let pos = fromFen StartFen
    let samples = ResizeArray<Sample>()
    // random opening: 6-9 plies
    let openingPlies = 6 + rng.Next 4
    let mutable aborted = false
    for _ in 1 .. openingPlies do
        if not aborted then
            let legal = legalMoves pos
            if legal.Count = 0 then aborted <- true
            else makeMove pos legal.[rng.Next legal.Count]
    if aborted then samples, 0
    else
        let mutable result = ValueNone
        let mutable plies = 0
        while result.IsNone do
            let legal = legalMoves pos
            if legal.Count = 0 then
                result <-
                    if inCheck pos then ValueSome (if pos.Stm = White then -1 else 1)
                    else ValueSome 0
            elif pos.Half >= 100 || isThreefold pos || insufficient pos then result <- ValueSome 0
            elif plies >= 400 then result <- ValueSome 0
            elif plies < 16 && rng.NextDouble() < 0.08 then
                // exploration: occasional random early move for game diversity (unsampled)
                makeMove pos legal.[rng.Next legal.Count]
                plies <- plies + 1
            else
                st.Stop.Value <- false
                let mv = think pos st { defaultLimits with NodeLimit = nodesPerMove; Depth = 32 } false
                let scoreStm = st.BestScore
                let scoreWhite = if pos.Stm = White then scoreStm else -scoreStm
                // resign adjudication keeps games short and labels clean
                if abs scoreStm >= 1500 then
                    result <- ValueSome (if scoreWhite > 0 then 1 else -1)
                else
                    // keep quiet, non-extreme positions only
                    let mv' = if mv = NoMove then legal.[0] else mv
                    let quiet =
                        not (inCheck pos)
                        && pos.Mailbox.[moveTo mv'] < 0
                        && moveFlag mv' = FlagNormal
                        && abs scoreStm < 1200
                    if quiet then
                        samples.Add
                            { Boards = Array.copy pos.ByPiece
                              Stm = byte pos.Stm
                              ScoreWhite = int16 (max -30000 (min 30000 scoreWhite)) }
                    makeMove pos mv'
                    plies <- plies + 1
        samples, (match result with ValueSome r -> r | ValueNone -> 0)

/// Generate training data: `games` self-play games at `nodesPerMove` across
/// `lanes` concurrent workers, appended to `outPath` (writer fully serialized:
/// each game's samples are written as one atomic block — no partial records).
let run (games: int) (nodesPerMove: uint64) (outPath: string) (lanes: int) =
    let lanes = max 1 (min lanes games)
    use fs = new FileStream(outPath, FileMode.Append, FileAccess.Write)
    use bw = new BinaryWriter(fs)
    let sync = obj ()
    let nextGame = [| 0 |]
    let doneGames = [| 0 |]
    let totalSamples = [| 0L |]
    let sw = System.Diagnostics.Stopwatch.StartNew()
    let worker (lane: int) =
        try
            let st = createState 128
            let rng = Random(20260610 + lane * 7919)
            let mutable g = System.Threading.Interlocked.Increment(&nextGame.[0])
            while g <= games do
                if g % 50 = 0 then st.Tt.Clear()   // periodic TT reset for game diversity
                let samples, result = playGame st rng nodesPerMove
                lock sync (fun () ->
                    for s in samples do
                        for b in s.Boards do bw.Write b
                        bw.Write s.Stm
                        bw.Write s.ScoreWhite
                        bw.Write (sbyte result)
                    bw.Flush()
                    totalSamples.[0] <- totalSamples.[0] + int64 samples.Count
                    doneGames.[0] <- doneGames.[0] + 1
                    if doneGames.[0] % 10 = 0 || doneGames.[0] = games then
                        printfn "game %d/%d  samples %d  (%.0f samples/s)"
                            doneGames.[0] games totalSamples.[0]
                            (float totalSamples.[0] / sw.Elapsed.TotalSeconds))
                g <- System.Threading.Interlocked.Increment(&nextGame.[0])
        with ex -> logCrash "datagen lane" ex
    let threads =
        [| for lane in 1 .. lanes - 1 ->
             let t = System.Threading.Thread((fun () -> worker lane), 16 * 1024 * 1024)
             t.IsBackground <- true
             t.Start()
             t |]
    worker 0
    for t in threads do t.Join()
    printfn "done: %d samples -> %s" totalSamples.[0] outPath
