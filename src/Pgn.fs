module AlterEgo.Pgn

// PGN -> NNUE training samples. Replays standard/TCEC PGN games and labels
// every non-book position with a fixed-node search of the current network
// (the gen-3 datagen recipe: own-eval labels + true game result), written in
// Datagen's 100-byte record format. Output is created fresh, never appended
// (interior partial records from appends poisoned net4 v1 — see scrub).

open System
open System.IO
open System.Text.RegularExpressions
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

type private PgnGame =
    { Result: int        // White POV: +1 / 0 / -1
      Sans: string[]
      Book: bool[] }     // per move: TCEC-style "book" comment (not sampled)

/// Single-pass scanner: tag lines at depth 0, {}-comments (multi-line safe),
/// ()-variations skipped, SAN tokens collected with per-move book flags.
let private parseGames (text: string) : PgnGame[] =
    let games = ResizeArray<PgnGame>()
    let sans = ResizeArray<string>()
    let book = ResizeArray<bool>()
    let mutable result = ValueNone
    let finalize () =
        if sans.Count > 0 then
            while book.Count < sans.Count do book.Add false
            match result with
            | ValueSome r -> games.Add { Result = r; Sans = sans.ToArray(); Book = book.ToArray() }
            | ValueNone -> ()   // unfinished game ("*"): no result label, drop
        sans.Clear()
        book.Clear()
        result <- ValueNone
    let mutable depth = 0
    let mutable i = 0
    let n = text.Length
    while i < n do
        let c = text.[i]
        if c = '[' && depth = 0 && (i = 0 || text.[i - 1] = '\n') then
            let eol = text.IndexOf('\n', i)
            let eol = if eol < 0 then n else eol
            let line = text.Substring(i, eol - i).TrimEnd()
            // any tag line after movetext starts a new game: don't rely on
            // result tokens or [Event leading the tag roster — relying on
            // [Event alone can merge games and poison result labels
            if sans.Count > 0 then finalize ()
            let m = Regex.Match(line, "^\\[Result \"([^\"]+)\"\\]")
            if m.Success then
                result <-
                    match m.Groups.[1].Value with
                    | "1-0" -> ValueSome 1
                    | "0-1" -> ValueSome -1
                    | "1/2-1/2" -> ValueSome 0
                    | _ -> ValueNone
            i <- eol + 1
        elif c = '{' then
            let close = text.IndexOf('}', i + 1)
            let close = if close < 0 then n else close
            // first comment after a move carries its metadata ("book, ...")
            if depth = 0 && book.Count = sans.Count - 1 then
                let comment = text.Substring(i + 1, close - i - 1).TrimStart()
                book.Add (comment.StartsWith "book")
            i <- close + 1
        elif c = '(' then depth <- depth + 1; i <- i + 1
        elif c = ')' then depth <- max 0 (depth - 1); i <- i + 1
        elif c = ';' then
            let eol = text.IndexOf('\n', i)
            i <- if eol < 0 then n else eol + 1
        elif Char.IsWhiteSpace c then i <- i + 1
        else
            let mutable j = i
            while j < n && not (Char.IsWhiteSpace text.[j])
                  && text.[j] <> '{' && text.[j] <> '(' && text.[j] <> ')' && text.[j] <> ';' do
                j <- j + 1
            let tok = text.Substring(i, j - i)
            i <- j
            if depth > 0 then ()
            elif tok = "1-0" || tok = "0-1" || tok = "1/2-1/2" || tok = "*" then finalize ()
            elif tok.[0] = '$' then ()
            elif Char.IsDigit tok.[0] && Seq.forall (fun ch -> Char.IsDigit ch || ch = '.') tok then ()
            else
                while book.Count < sans.Count do book.Add false   // prior move had no comment
                sans.Add tok
    finalize ()
    games.ToArray()

/// Match a SAN token against the position's legal moves.
/// Returns NoMove when nothing (or more than one move) matches.
let private sanToMove (pos: Position) (sanRaw: string) (buf: Move[]) : Move =
    let san = sanRaw.TrimEnd([| '+'; '#'; '!'; '?' |])
    let n = generate pos buf
    let candidates = ResizeArray<Move>()
    let addIfLegal (m: Move) =
        makeMove pos m
        if wasLegal pos then candidates.Add m
        unmakeMove pos m
    if san.StartsWith "O-O" || san.StartsWith "0-0" then
        let long = san.StartsWith "O-O-O" || san.StartsWith "0-0-0"
        let target = (if pos.Stm = White then 0 else 56) + (if long then 2 else 6)
        for k in 0 .. n - 1 do
            if moveFlag buf.[k] = FlagCastle && moveTo buf.[k] = target then addIfLegal buf.[k]
    elif san.Length >= 2 then
        let mutable core = san
        let mutable promo = -1
        let eq = core.IndexOf '='
        if eq >= 0 && eq + 1 < core.Length then
            promo <-
                match core.[eq + 1] with
                | 'N' -> Knight | 'B' -> Bishop | 'R' -> Rook | 'Q' -> Queen | _ -> -2
            core <- core.Substring(0, eq)
        let piece, rest =
            match core.[0] with
            | 'N' -> Knight, core.Substring 1
            | 'B' -> Bishop, core.Substring 1
            | 'R' -> Rook, core.Substring 1
            | 'Q' -> Queen, core.Substring 1
            | 'K' -> King, core.Substring 1
            | _ -> Pawn, core
        let rest = rest.Replace("x", "")
        if rest.Length >= 2 then
            let target = parseSquare (rest.Substring(rest.Length - 2))
            let mutable dFile = -1
            let mutable dRank = -1
            for ch in rest.Substring(0, rest.Length - 2) do
                if ch >= 'a' && ch <= 'h' then dFile <- int ch - int 'a'
                elif ch >= '1' && ch <= '8' then dRank <- int ch - int '1'
            if target >= 0 then
                for k in 0 .. n - 1 do
                    let m = buf.[k]
                    let from = moveFrom m
                    if moveTo m = target
                       && moveFlag m <> FlagCastle
                       && pos.Mailbox.[from] = pos.Stm * 6 + piece
                       && (dFile < 0 || fileOf from = dFile)
                       && (dRank < 0 || rankOf from = dRank)
                       && (if promo >= 0 then moveFlag m = FlagPromo && movePromo m = promo
                           else moveFlag m <> FlagPromo) then
                        addIfLegal m
    if candidates.Count = 1 then candidates.[0] else NoMove

[<Struct>]
type private Sample =
    { Boards: uint64[]
      Stm: byte
      ScoreWhite: int16 }

/// Replay one game, searching every non-book position for its label.
/// Returns collected samples and whether replay broke on an unmatched SAN
/// (samples gathered before the break are still valid positions).
let private processGame (st: State) (pos: Position) (game: PgnGame) (nodes: uint64) (buf: Move[]) =
    setFen pos StartFen
    let samples = ResizeArray<Sample>()
    let mutable failed = ValueNone
    let mutable idx = 0
    while failed.IsNone && idx < game.Sans.Length do
        let mv = sanToMove pos game.Sans.[idx] buf
        if mv = NoMove then failed <- ValueSome game.Sans.[idx]
        else
            if not game.Book.[idx] then
                st.Stop.Value <- false
                think pos st { defaultLimits with NodeLimit = nodes; Depth = 32 } false |> ignore
                let scoreStm = st.BestScore
                // datagen's quiet filter, applied to the game's actual move
                let quiet =
                    not (inCheck pos)
                    && pos.Mailbox.[moveTo mv] < 0
                    && moveFlag mv = FlagNormal
                    && abs scoreStm < 1200
                if quiet then
                    let scoreWhite = if pos.Stm = White then scoreStm else -scoreStm
                    samples.Add
                        { Boards = Array.copy pos.ByPiece
                          Stm = byte pos.Stm
                          ScoreWhite = int16 (max -30000 (min 30000 scoreWhite)) }
            makeMove pos mv
            idx <- idx + 1
    samples, failed

/// Convert a PGN file into labeled training samples across `lanes` workers.
let run (pgnPath: string) (outPath: string) (nodes: uint64) (lanes: int) =
    let games = parseGames (File.ReadAllText pgnPath)
    printfn "parsed %d games from %s (labeling at %d nodes/position)" games.Length pgnPath nodes
    let lanes = max 1 (min lanes games.Length)
    use fs = new FileStream(outPath, FileMode.Create, FileAccess.Write)
    use bw = new BinaryWriter(fs)
    let sync = obj ()
    let next = [| -1 |]
    let doneGames = [| 0 |]
    let totalSamples = [| 0L |]
    let sanFailures = [| 0 |]
    let laneFailures = [| 0 |]
    let sw = Diagnostics.Stopwatch.StartNew()
    let worker (_: int) =
        try
            let st = createState 128
            let pos = fromFen StartFen
            let buf = Array.zeroCreate<Move> 256
            let mutable g = Threading.Interlocked.Increment(&next.[0])
            while g < games.Length do
                let samples, sanFail = processGame st pos games.[g] nodes buf
                lock sync (fun () ->
                    for s in samples do
                        for b in s.Boards do bw.Write b
                        bw.Write s.Stm
                        bw.Write s.ScoreWhite
                        bw.Write (sbyte games.[g].Result)
                    bw.Flush()
                    totalSamples.[0] <- totalSamples.[0] + int64 samples.Count
                    doneGames.[0] <- doneGames.[0] + 1
                    match sanFail with
                    | ValueSome san ->
                        sanFailures.[0] <- sanFailures.[0] + 1
                        printfn "game %d: unmatched SAN '%s' — kept %d samples up to the break" (g + 1) san samples.Count
                    | ValueNone -> ()
                    if doneGames.[0] % 5 = 0 || doneGames.[0] = games.Length then
                        printfn "game %d/%d  samples %d  (%.0f samples/s)"
                            doneGames.[0] games.Length totalSamples.[0]
                            (float totalSamples.[0] / sw.Elapsed.TotalSeconds))
                g <- Threading.Interlocked.Increment(&next.[0])
        with ex ->
            Threading.Interlocked.Increment(&laneFailures.[0]) |> ignore
            logCrash "pgnconv lane" ex
    let threads =
        [| for lane in 1 .. lanes - 1 ->
             let t = Threading.Thread((fun () -> worker lane), 16 * 1024 * 1024)
             t.IsBackground <- true
             t.Start()
             t |]
    worker 0
    for t in threads do t.Join()
    if laneFailures.[0] > 0 then
        printfn "WARNING: %d lane(s) crashed — conversion incomplete (see alterego-crash.log)" laneFailures.[0]
    if sanFailures.[0] > 0 then
        printfn "WARNING: %d game(s) had unmatched SAN tokens" sanFailures.[0]
    printfn "done: %d samples from %d games -> %s" totalSamples.[0] games.Length outPath
