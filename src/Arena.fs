module AlterEgo.Arena

// M7: in-process match harness. Plays engine configurations against each other
// with alternating colors over a balanced opening set, adjudicates results, and
// reports score, Elo difference and likelihood of superiority.

open AlterEgo.Types
open AlterEgo.Bitboards
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Search

type PlayerKind =
    | Ego
    | Machine
    | PstEgo   // Ego with NNUE suppressed (A/B baseline)

type GameResult =
    | WhiteWins
    | BlackWins
    | Draw

/// GSPRT log-likelihood ratio for H1: elo=elo1 vs H0: elo=elo0 (trinomial,
/// draw ratio fixed at the observed value). Bounds for alpha=beta=0.05: +/-2.94.
/// +0.5 pseudo-counts keep shutout results informative instead of returning 0.
let sprtLlr (wins: int) (draws: int) (losses: int) (elo0: float) (elo1: float) =
    let n = wins + draws + losses
    if n = 0 then 0.0
    else
        let w = float wins + 0.5
        let l = float losses + 0.5
        let d = float draws
        let total = w + l + d
        let dr = d / total
        let expScore e = 1.0 / (1.0 + 10.0 ** (-e / 400.0))
        let probs e =
            let p = expScore e
            let pw = max 1e-9 (p - dr / 2.0)
            let pl = max 1e-9 (1.0 - p - dr / 2.0)
            pw, pl
        let w0, l0 = probs elo0
        let w1, l1 = probs elo1
        w * log (w1 / w0) + l * log (l1 / l0)

[<Literal>]
let SprtUpper = 2.94    // accept H1 (alpha = 0.05)
[<Literal>]
let SprtLower = -2.94   // accept H0 (beta = 0.05)

// short, balanced opening lines (UCI moves from startpos)
let private openings =
    [| "e2e4 e7e5 g1f3 b8c6"
       "d2d4 d7d5 c2c4 e7e6"
       "g1f3 g8f6 c2c4 c7c5"
       "e2e4 c7c5 g1f3 d7d6"
       "d2d4 g8f6 c2c4 e7e6"
       "e2e4 e7e6 d2d4 d7d5"
       "c2c4 e7e5 b1c3 g8f6"
       "d2d4 d7d5 g1f3 g8f6" |]

let private applyMoves (pos: Position) (moveStr: string) =
    for tok in moveStr.Split(' ') do
        let buf = Array.zeroCreate<Move> 256
        let n = generate pos buf
        let mutable i = 0
        let mutable applied = false
        while not applied && i < n do
            if moveToUci buf.[i] = tok then
                makeMove pos buf.[i]
                if wasLegal pos then applied <- true else unmakeMove pos buf.[i]
            i <- i + 1
        if not applied then failwithf "bad opening move %s" tok

let private legalMoves (pos: Position) =
    let buf = Array.zeroCreate<Move> 256
    let n = generate pos buf
    let result = ResizeArray<Move>()
    for i in 0 .. n - 1 do
        makeMove pos buf.[i]
        if wasLegal pos then result.Add buf.[i]
        unmakeMove pos buf.[i]
    result

/// threefold repetition relative to game history
let private isThreefold (pos: Position) =
    let limit = max 0 (pos.Ply - pos.Half)
    let mutable count = 0
    for i in limit .. pos.Ply - 1 do
        if pos.Undos.[i].Key = pos.Key then count <- count + 1
    count >= 2

let private insufficientMaterial (pos: Position) =
    let occ = occupancy pos
    let minors =
        pos.ByPiece.[Knight] ||| pos.ByPiece.[Bishop]
        ||| pos.ByPiece.[6 + Knight] ||| pos.ByPiece.[6 + Bishop]
    let kings = pos.ByPiece.[King] ||| pos.ByPiece.[6 + King]
    occ = (kings ||| minors) && popcount minors <= 1

/// Play one game; returns result from White's point of view.
let playGame (white: PlayerKind) (black: PlayerKind) (opening: string) (moveMs: int64)
             (stWhite: State) (stBlack: State) : GameResult =
    let pos = fromFen StartFen
    applyMoves pos opening
    let mutable result = ValueNone
    let mutable plies = 0
    while result.IsNone do
        let legal = legalMoves pos
        if legal.Count = 0 then
            result <-
                if inCheck pos then
                    ValueSome (if pos.Stm = White then BlackWins else WhiteWins)
                else ValueSome Draw
        elif pos.Half >= 100 || isThreefold pos || insufficientMaterial pos then
            result <- ValueSome Draw
        elif plies >= 500 then
            result <- ValueSome Draw
        else
            let kind = if pos.Stm = White then white else black
            let st = if pos.Stm = White then stWhite else stBlack
            pos.ForcePst <- (kind = PstEgo)
            st.Stop.Value <- false
            let mv =
                match kind with
                | Ego | PstEgo -> think pos st { defaultLimits with MoveTimeMs = moveMs } false
                | Machine -> AlterEgo.Machine.think pos st moveMs false
            let mv = if mv = NoMove || not (legal.Contains mv) then legal.[0] else mv
            makeMove pos mv
            plies <- plies + 1
    match result with
    | ValueSome r -> r
    | ValueNone -> Draw

/// One game vs an external UCI engine. Returns AlterEgo's points (1 / 0.5 / 0).
let playGameVsExternal (ext: AlterEgo.UciEngine.Engine) (weAreWhite: bool) (opening: string)
                       (moveMs: int64) (st: State) (useMachine: bool) : float =
    let pos = fromFen StartFen
    let moveList = System.Text.StringBuilder()
    applyMoves pos opening
    moveList.Append opening |> ignore
    ext.NewGame()
    let mutable result = ValueNone   // points for AlterEgo
    let mutable plies = 0
    while result.IsNone do
        let legal = legalMoves pos
        let weMove = (pos.Stm = White) = weAreWhite
        if legal.Count = 0 then
            result <-
                if inCheck pos then ValueSome (if weMove then 0.0 else 1.0)
                else ValueSome 0.5
        elif pos.Half >= 100 || isThreefold pos || insufficientMaterial pos then
            result <- ValueSome 0.5
        elif plies >= 500 then
            result <- ValueSome 0.5
        else
            let mvUci =
                if weMove then
                    st.Stop.Value <- false
                    let mv =
                        if useMachine then AlterEgo.Machine.think pos st moveMs false
                        else think pos st { defaultLimits with MoveTimeMs = moveMs } false
                    let mv = if mv = NoMove || not (legal.Contains mv) then legal.[0] else mv
                    moveToUci mv
                else
                    ext.BestMove(moveList.ToString(), moveMs)
            // apply by UCI token (validates external moves too)
            let mutable applied = false
            let mutable i = 0
            while not applied && i < legal.Count do
                if moveToUci legal.[i] = mvUci then
                    makeMove pos legal.[i]
                    applied <- true
                i <- i + 1
            if not applied then
                // external engine sent an illegal move: it forfeits
                result <- ValueSome (if weMove then 0.0 else 1.0)
            else
                if moveList.Length > 0 then moveList.Append ' ' |> ignore
                moveList.Append mvUci |> ignore
                plies <- plies + 1
    match result with
    | ValueSome r -> r
    | ValueNone -> 0.5

/// Gauntlet vs an external engine, alternating colors over the opening set.
/// `lanes` concurrent games, each lane with its own engine instance and state.
let runGauntlet (extPath: string) (extOptions: (string * string) list)
                (games: int) (moveMs: int64) (ttMb: int) (useMachine: bool) (lanes: int) =
    let lanes = max 1 (min lanes games)
    let sync = obj ()
    let nextGame = [| -1 |]
    let tally = [| 0; 0; 0 |]   // wins, draws, losses
    let earlyStop = [| false |]
    let mutable score = 0.0
    printfn "cage: AlterEgo%s vs %s — %d games at %dms/move, %d lanes"
        (if useMachine then "(MACHINE)" else "") extPath games moveMs lanes
    let worker () =
        try
            use ext = new AlterEgo.UciEngine.Engine(extPath, extOptions)
            let st = createState ttMb
            let mutable g = System.Threading.Interlocked.Increment(&nextGame.[0])
            while g < games && not (System.Threading.Volatile.Read(&earlyStop.[0])) do
                let opening = openings.[(g / 2) % openings.Length]
                let weAreWhite = g % 2 = 0
                st.Tt.Clear()
                let pts = playGameVsExternal ext weAreWhite opening moveMs st useMachine
                lock sync (fun () ->
                    score <- score + pts
                    if pts > 0.75 then tally.[0] <- tally.[0] + 1
                    elif pts > 0.25 then tally.[1] <- tally.[1] + 1
                    else tally.[2] <- tally.[2] + 1
                    let llr = sprtLlr tally.[0] tally.[1] tally.[2] 0.0 5.0
                    printfn "game %2d  [AlterEgo as %s]  %s   +%d =%d -%d  LLR %+.2f"
                        (g + 1) (if weAreWhite then "W" else "B")
                        (if pts > 0.75 then "WIN" elif pts > 0.25 then "draw" else "loss")
                        tally.[0] tally.[1] tally.[2] llr
                    if llr >= SprtUpper || llr <= SprtLower then
                        System.Threading.Volatile.Write(&earlyStop.[0], true))
                g <- System.Threading.Interlocked.Increment(&nextGame.[0])
        with ex -> logCrash "gauntlet lane" ex
    let threads =
        [| for _ in 1 .. lanes - 1 ->
             let t = System.Threading.Thread(worker, 16 * 1024 * 1024)
             t.IsBackground <- true
             t.Start()
             t |]
    worker ()
    for t in threads do t.Join()
    let played = tally.[0] + tally.[1] + tally.[2]
    let pct = if played = 0 then 0.5 else score / float played
    let eloDiff =
        if pct <= 0.0 then -999.0
        elif pct >= 1.0 then 999.0
        else -400.0 * log10 (1.0 / pct - 1.0)
    let llr = sprtLlr tally.[0] tally.[1] tally.[2] 0.0 5.0
    let verdict =
        if llr >= SprtUpper then "H1 accepted (stronger)"
        elif llr <= SprtLower then "H0 accepted (not stronger)"
        else "inconclusive"
    printfn ""
    printfn "cage result: +%d =%d -%d (%d games)  score %.1f%%  elo %+.0f  LLR %+.2f [%s]"
        tally.[0] tally.[1] tally.[2] played (pct * 100.0) eloDiff llr verdict

/// Run a match: `kindA` vs `kindB`, alternating colors across the opening set.
/// `lanes` games run concurrently, each lane with its own pair of states.
let runMatch (kindA: PlayerKind) (kindB: PlayerKind) (games: int) (moveMs: int64) (ttMb: int) (lanes: int) =
    let lanes = max 1 (min lanes games)
    let sync = obj ()
    let nextGame = [| -1 |]
    let earlyStop = [| false |]
    let mutable winsA = 0
    let mutable winsB = 0
    let mutable draws = 0
    let nameOf k = match k with Ego -> "Ego" | Machine -> "MACHINE" | PstEgo -> "Ego(PST)"
    printfn "match: %s vs %s — %d games at %dms/move, %d lanes" (nameOf kindA) (nameOf kindB) games moveMs lanes
    let worker () =
        try
            let stA = createState ttMb
            let stB = createState ttMb
            let mutable g = System.Threading.Interlocked.Increment(&nextGame.[0])
            while g < games && not (System.Threading.Volatile.Read(&earlyStop.[0])) do
                let opening = openings.[(g / 2) % openings.Length]
                let aIsWhite = g % 2 = 0
                stA.Tt.Clear()
                stB.Tt.Clear()
                let result =
                    if aIsWhite then playGame kindA kindB opening moveMs stA stB
                    else playGame kindB kindA opening moveMs stB stA
                lock sync (fun () ->
                    let aPoint =
                        match result, aIsWhite with
                        | WhiteWins, true | BlackWins, false -> winsA <- winsA + 1; "1-0 (A)"
                        | WhiteWins, false | BlackWins, true -> winsB <- winsB + 1; "0-1 (B)"
                        | Draw, _ -> draws <- draws + 1; "1/2"
                    let llr = sprtLlr winsA draws winsB 0.0 5.0
                    printfn "game %2d  [%s as %s]  %s   +%d =%d -%d  LLR %+.2f"
                        (g + 1) (nameOf kindA) (if aIsWhite then "W" else "B") aPoint winsA draws winsB llr
                    if llr >= SprtUpper || llr <= SprtLower then
                        System.Threading.Volatile.Write(&earlyStop.[0], true))
                g <- System.Threading.Interlocked.Increment(&nextGame.[0])
        with ex -> logCrash "match lane" ex
    let threads =
        [| for _ in 1 .. lanes - 1 ->
             let t = System.Threading.Thread(worker, 16 * 1024 * 1024)
             t.IsBackground <- true
             t.Start()
             t |]
    worker ()
    for t in threads do t.Join()
    let total = float (winsA + winsB + draws)
    let score = (float winsA + 0.5 * float draws) / total
    let eloDiff =
        if score <= 0.0 then -999.0
        elif score >= 1.0 then 999.0
        else -400.0 * log10 (1.0 / score - 1.0)
    // likelihood of superiority via normal approximation
    let erf (x: float) =
        let t = 1.0 / (1.0 + 0.3275911 * abs x)
        let y =
            1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                     - 0.284496736) * t + 0.254829592) * t) * exp (-x * x)
        if x >= 0.0 then y else -y
    let mu = float winsA - float winsB
    let var = float winsA + float winsB
    let los =
        if var <= 0.0 then 0.5
        else 0.5 * (1.0 + erf (mu / sqrt (2.0 * var)))
    printfn ""
    let llr = sprtLlr winsA draws winsB 0.0 5.0
    let verdict =
        if llr >= 2.94 then "H1 accepted (stronger)"
        elif llr <= -2.94 then "H0 accepted (not stronger)"
        else "inconclusive — more games needed"
    printfn "result: +%d =%d -%d  score %.1f%%  elo %+.0f  LOS %.1f%%  LLR %.2f [%s]"
        winsA draws winsB (score * 100.0) eloDiff (los * 100.0) llr verdict
