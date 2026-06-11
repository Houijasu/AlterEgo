module AlterEgo.Search

open System.Diagnostics
open System.Threading
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Eval
open AlterEgo.TT

[<Literal>]
let Infinity = 30000
[<Literal>]
let MateValue = 29000
[<Literal>]
let MaxPly = 128

let inline isMateScore (s: int) = abs s > MateValue - 2 * MaxPly

type Limits =
    { Depth: int
      MoveTimeMs: int64
      TimeMs: int64
      IncMs: int64
      MovesToGo: int
      NodeLimit: uint64
      Infinite: bool }

let defaultLimits =
    { Depth = 0; MoveTimeMs = 0L; TimeMs = 0L; IncMs = 0L; MovesToGo = 0; NodeLimit = 0UL; Infinite = false }

/// Shared stop signal — volatile so helper threads observe main-thread writes
type StopFlag() =
    [<DefaultValue>]
    val mutable private V: bool
    member this.Value
        with get () = Volatile.Read(&this.V)
        and set x = Volatile.Write(&this.V, x)

type State =
    { mutable Tt: Table
      mutable Nodes: uint64
      Stop: StopFlag
      Sw: Stopwatch
      mutable HardMs: int64
      mutable NodeLimit: uint64
      Killers: Move[]        // 2 per ply
      History: int[]         // [stm * 4096 + from*64+to]
      CorrHist: int[]        // [stm * 16384 + (pawnKey & 16383)] — eval correction
      ContHist: int[]        // [(prevPiece*64+prevTo) * 768 + piece*64+to] — continuation
      ContIdx: int[]         // per-ply: previous move's piece*64+to, or -1 (null/root)
      EvalStack: int[]       // per-ply static eval (MinValue = in check / unknown)
      Buffers: Move[][]
      Scores: int[][]
      QuietBuf: Move[][]     // staged-generation / probcut scratch
      TriedQuiets: Move[][]  // quiets searched per ply, for history maluses
      mutable BestMove: Move
      mutable BestScore: int
      mutable CompletedDepth: int
      // lazy SMP: cached helper resources (zero per-move allocation)
      mutable ThreadCount: int
      mutable HelperStates: State[]
      mutable HelperPositions: Position[] }

let private newState (tt: Table) (stop: StopFlag) =
    { Tt = tt
      Nodes = 0UL
      Stop = stop
      Sw = Stopwatch()
      HardMs = 0L
      NodeLimit = 0UL
      Killers = Array.zeroCreate (MaxPly * 2)
      History = Array.zeroCreate (2 * 4096)
      CorrHist = Array.zeroCreate (2 * 16384)
      ContHist = Array.zeroCreate (768 * 768)
      ContIdx = Array.create (MaxPly + 2) -1
      EvalStack = Array.create (MaxPly + 2) System.Int32.MinValue
      Buffers = Array.init MaxPly (fun _ -> Array.zeroCreate<Move> 256)
      Scores = Array.init MaxPly (fun _ -> Array.zeroCreate<int> 256)
      QuietBuf = Array.init MaxPly (fun _ -> Array.zeroCreate<Move> 256)
      TriedQuiets = Array.init MaxPly (fun _ -> Array.zeroCreate<Move> 128)
      BestMove = NoMove
      BestScore = 0
      CompletedDepth = 0
      ThreadCount = 1
      HelperStates = [||]
      HelperPositions = [||] }

let createState (ttMb: int) = newState (Table ttMb) (StopFlag())

/// Size the cached helper pool to ThreadCount - 1 (idempotent)
let ensureHelpers (st: State) =
    let need = max 0 (st.ThreadCount - 1)
    if st.HelperStates.Length <> need then
        st.HelperStates <- Array.init need (fun _ -> newState st.Tt st.Stop)
        st.HelperPositions <- Array.init need (fun _ -> create ())
    else
        for h in st.HelperStates do h.Tt <- st.Tt

// Search features ship default-on only once individually SPRT-proven.
// Promoted: singular (+38 @128g), lmp (+58 @128g, base2 +24), improving (+24 @128g).
// Unproven (opt-in via ALTEREGO_ENABLE): probcut, corrhist, conthist, seequiet.
// Promoted features can be switched off via ALTEREGO_DISABLE for A/B runs.
let private parseSet (envVar: string) =
    match System.Environment.GetEnvironmentVariable envVar with
    | null -> Set.empty
    | s -> s.Split(',') |> Array.map (fun x -> x.Trim().ToLowerInvariant()) |> Set.ofArray

let private enabled = parseSet "ALTEREGO_ENABLE"
let private disabled = parseSet "ALTEREGO_DISABLE"

let private useProbcut = enabled.Contains "probcut"
let private useSingular = not (disabled.Contains "singular")
let private useCorrHist = enabled.Contains "corrhist"
let private useContHist = enabled.Contains "conthist"
let private useLmp = not (disabled.Contains "lmp")
let private useSeeQuiet = enabled.Contains "seequiet"
let private useImproving = not (disabled.Contains "improving")

// margin knobs for tuning sweeps: ALTEREGO_TUNE=sbetamult=3,corrdiv=32,pcmargin=200
let private tune =
    match System.Environment.GetEnvironmentVariable "ALTEREGO_TUNE" with
    | null -> Map.empty
    | s ->
        s.Split(',')
        |> Array.choose (fun kv ->
            match kv.Split('=') with
            | [| k; v |] ->
                match System.Int32.TryParse(v.Trim()) with
                | true, n -> Some (k.Trim().ToLowerInvariant(), n)
                | _ -> None
            | _ -> None)
        |> Map.ofArray

let private tuned key dflt = tune |> Map.tryFind key |> Option.defaultValue dflt

let private sBetaMult = tuned "sbetamult" 3     // singular margin: ttScore - mult*depth (3 = SPRT winner)
let private sDepthGate = tuned "sdepth" 6       // singular minimum depth (6 = SPRT winner)
let private corrDiv = tuned "corrdiv" 16        // correction strength divisor
let private corrW = tuned "corrw" 16            // correction update weight cap
let private pcMargin = tuned "pcmargin" 160     // probcut beta margin
let private pcDepthGate = tuned "pcdepth" 5     // probcut minimum depth
let private lmpBase = tuned "lmpbase" 2         // LMP threshold: base + depth^2 (2 = SPRT winner)
let private lmpMaxDepth = tuned "lmpdepth" 8    // LMP maximum depth
let private seeQMargin = tuned "seeqmargin" 60  // SEE quiet pruning: cp lost per depth
let private seeQDepth = tuned "seeqdepth" 8     // SEE quiet pruning maximum depth

// log-based late-move-reduction table
let private lmrTable =
    Array2D.init 64 64 (fun d m ->
        if d < 1 || m < 1 then 0
        else int (0.77 + log (float d) * log (float m) / 2.36))

// mate scores are ply-relative in search, root-relative in TT
let inline scoreToTt (s: int) (ply: int) =
    if s >= MateValue - 2 * MaxPly then s + ply
    elif s <= -(MateValue - 2 * MaxPly) then s - ply
    else s

let inline scoreFromTt (s: int) (ply: int) =
    if s >= MateValue - 2 * MaxPly then s - ply
    elif s <= -(MateValue - 2 * MaxPly) then s + ply
    else s

let inline private checkUp (st: State) =
    if st.Nodes &&& 2047UL = 0UL then
        if st.HardMs > 0L && st.Sw.ElapsedMilliseconds >= st.HardMs then st.Stop.Value <- true
        elif st.NodeLimit > 0UL && st.Nodes >= st.NodeLimit then st.Stop.Value <- true

let inline private isCapture (pos: Position) (m: Move) =
    pos.Mailbox.[moveTo m] >= 0 || moveFlag m = FlagEnPassant

let inline private historyIndex (pos: Position) (m: Move) =
    pos.Stm * 4096 + (int m &&& 0xFFF)

/// piece*64+to of move m — MUST be computed before makeMove (reads the mailbox)
let inline private moveContIdx (pos: Position) (m: Move) =
    pos.Mailbox.[moveFrom m] * 64 + moveTo m

/// combined quiet-move ordering score (capped below killers when contHist is on;
/// raw main history otherwise — exact baseline ordering parity)
let inline private quietScore (pos: Position) (st: State) (m: Move) (ply: int) =
    if not useContHist then st.History.[historyIndex pos m]
    else
        let h = st.History.[historyIndex pos m]
        let prev = st.ContIdx.[ply]
        let c = if prev >= 0 then st.ContHist.[prev * 768 + moveContIdx pos m] else 0
        min 79_000 (h + c)

// ---- correction history: static eval bias per (stm, pawn structure) ----
let inline private corrIndex (pos: Position) =
    pos.Stm * 16384 + int (pos.PawnKey &&& 16383UL)

let inline private correctedEval (pos: Position) (st: State) (raw: int) =
    if not useCorrHist then raw
    else
        let bound = MateValue - 2 * MaxPly
        let c = raw + st.CorrHist.[corrIndex pos] / corrDiv
        if c > bound then bound elif c < -bound then -bound else c

let inline private pickNext (moves: Move[]) (scores: int[]) (i: int) (n: int) =
    let mutable bi = i
    for j in i + 1 .. n - 1 do
        if scores.[j] > scores.[bi] then bi <- j
    if bi <> i then
        let tm = moves.[i] in moves.[i] <- moves.[bi]; moves.[bi] <- tm
        let ts = scores.[i] in scores.[i] <- scores.[bi]; scores.[bi] <- ts

let rec qsearch (pos: Position) (st: State) (alphaIn: int) (beta: int) (ply: int) : int =
    st.Nodes <- st.Nodes + 1UL
    checkUp st
    if st.Stop.Value then 0
    elif ply >= MaxPly - 1 then evaluate pos
    else
        let inChk = inCheck pos
        let mutable alpha = alphaIn
        let standPat = if inChk then -Infinity else correctedEval pos st (evaluate pos)
        if not inChk && standPat >= beta then standPat
        else
            if standPat > alpha then alpha <- standPat
            let moves = st.Buffers.[ply]
            let scores = st.Scores.[ply]
            let n =
                if inChk then generate pos moves           // all evasions
                else generateCaptures pos moves            // tactical moves only
            for i in 0 .. n - 1 do
                let m = moves.[i]
                scores.[i] <-
                    if isCapture pos m then
                        let victim =
                            if moveFlag m = FlagEnPassant then Pawn
                            else pieceType pos.Mailbox.[moveTo m]
                        100_000 + victim * 8 - pieceType pos.Mailbox.[moveFrom m]
                    elif moveFlag m = FlagPromo then 90_000 + movePromo m
                    else 0
            let mutable best = standPat
            let mutable legalCount = 0
            let mutable i = 0
            let mutable stop = false
            while not stop && i < n do
                pickNext moves scores i n
                let m = moves.[i]
                let skip =
                    not inChk
                    && moveFlag m <> FlagPromo
                    && isCapture pos m
                    && ((let victim =
                            if moveFlag m = FlagEnPassant then Pawn
                            else pieceType pos.Mailbox.[moveTo m]
                         standPat + pieceValue.[victim] + 200 <= alpha)
                        || not (seeGe pos m 0))
                if not skip then
                    makeMove pos m
                    if not (wasLegal pos) then unmakeMove pos m
                    else
                        legalCount <- legalCount + 1
                        let v = -qsearch pos st (-beta) (-alpha) (ply + 1)
                        unmakeMove pos m
                        if st.Stop.Value then stop <- true
                        else
                            if v > best then best <- v
                            if v > alpha then alpha <- v
                            if alpha >= beta then stop <- true
                i <- i + 1
            if st.Stop.Value then 0
            elif inChk && legalCount = 0 then -(MateValue - ply)
            else best

/// Full search with an optional excluded move (singular verification).
let rec searchEx (pos: Position) (st: State) (depthIn: int) (ply: int) (alphaIn: int) (betaIn: int) (nullOk: bool) (excluded: Move) : int =
    if depthIn <= 0 then qsearch pos st alphaIn betaIn ply
    else
        st.Nodes <- st.Nodes + 1UL
        checkUp st
        let isRoot = ply = 0
        if st.Stop.Value then 0
        elif not isRoot && (pos.Half >= 100 || isRepetition pos) then 0
        elif ply >= MaxPly - 1 then evaluate pos
        else
            let mutable alpha = alphaIn
            let beta = betaIn
            let key = pos.Key
            let tte = st.Tt.Probe key
            let ttMove = if tte.Hit then tte.Move else NoMove
            // TT cutoff (never inside an exclusion search)
            let mutable ttCut = System.Int32.MinValue
            if not isRoot && excluded = NoMove && tte.Hit && tte.Depth >= depthIn then
                let s = scoreFromTt tte.Score ply
                if tte.Bound = BoundExact then ttCut <- s
                elif tte.Bound = BoundLower && s >= beta then ttCut <- s
                elif tte.Bound = BoundUpper && s <= alpha then ttCut <- s
            if ttCut <> System.Int32.MinValue then ttCut
            else
                let inChk = inCheck pos
                let depth = if inChk then depthIn + 1 else depthIn   // check extension
                let rawEval = if inChk then -Infinity else evaluate pos
                let staticEval = if inChk then -Infinity else correctedEval pos st rawEval
                // improving: static eval trend vs two plies ago (unknown => true)
                st.EvalStack.[ply] <- if inChk then System.Int32.MinValue else staticEval
                let improving =
                    not inChk
                    && (ply < 2
                        || st.EvalStack.[ply - 2] = System.Int32.MinValue
                        || staticEval > st.EvalStack.[ply - 2])
                // reverse futility pruning (improving: slightly smaller margin)
                let rfpDepth = if useImproving && improving then max 1 (depth - 1) else depth
                let mutable earlyCut = System.Int32.MinValue
                if not inChk && not isRoot && depth <= 8
                   && not (isMateScore beta) && staticEval - 80 * rfpDepth >= beta then
                    earlyCut <- staticEval
                // null-move pruning
                if earlyCut = System.Int32.MinValue
                   && nullOk && excluded = NoMove && not inChk && not isRoot && depth >= 3
                   && not (isMateScore beta) && hasNonPawnMaterial pos
                   && staticEval >= beta then
                    let r = 2 + depth / 6
                    st.ContIdx.[ply + 1] <- -1
                    makeNull pos
                    let v = -searchEx pos st (depth - 1 - r) (ply + 1) (-beta) (-beta + 1) false NoMove
                    unmakeNull pos
                    if not st.Stop.Value && v >= beta then
                        earlyCut <- if isMateScore v then beta else v
                // probcut: a good capture beating beta by a margin at reduced depth
                if useProbcut && earlyCut = System.Int32.MinValue
                   && not isRoot && excluded = NoMove && not inChk && depth >= pcDepthGate
                   && not (isMateScore beta)
                   && not (tte.Hit && tte.Depth >= depth - 3 && scoreFromTt tte.Score ply < beta + pcMargin) then
                    let probCutBeta = beta + pcMargin
                    let pcMoves = st.QuietBuf.[ply]
                    let pn = generateCaptures pos pcMoves
                    let mutable pi = 0
                    while earlyCut = System.Int32.MinValue && pi < pn do
                        let m = pcMoves.[pi]
                        if seeGe pos m (probCutBeta - staticEval) then
                            st.ContIdx.[ply + 1] <- moveContIdx pos m
                            makeMove pos m
                            if not (wasLegal pos) then unmakeMove pos m
                            else
                                let mutable v = -qsearch pos st (-probCutBeta) (-probCutBeta + 1) (ply + 1)
                                if v >= probCutBeta && depth >= 6 then
                                    v <- -searchEx pos st (depth - 4) (ply + 1) (-probCutBeta) (-probCutBeta + 1) true NoMove
                                unmakeMove pos m
                                if not st.Stop.Value && v >= probCutBeta then
                                    st.Tt.Store(key, m, scoreToTt v ply, depth - 3, BoundLower)
                                    earlyCut <- v
                        pi <- pi + 1
                if st.Stop.Value then 0
                elif earlyCut <> System.Int32.MinValue then earlyCut
                else
                    // singular extension / multicut via exclusion search
                    let mutable singularExt = 0
                    let mutable mcCut = System.Int32.MinValue
                    if useSingular && not isRoot && excluded = NoMove && depth >= sDepthGate
                       && ttMove <> NoMove && tte.Hit && tte.Depth >= depth - 3
                       && tte.Bound <> BoundUpper && not (isMateScore tte.Score) then
                        let sBeta = scoreFromTt tte.Score ply - sBetaMult * depth
                        let v = searchEx pos st ((depth - 1) / 2) ply (sBeta - 1) sBeta false ttMove
                        if not st.Stop.Value then
                            if v < sBeta then singularExt <- 1          // TT move is singular: extend
                            elif sBeta >= beta then mcCut <- sBeta      // multicut: several moves beat beta
                    if st.Stop.Value then 0
                    elif mcCut <> System.Int32.MinValue then mcCut
                    else
                        // ---- move loop: full generation, unified ordering ----
                        // (staged generation reverted: a quiet TT move emitted late
                        //  cost -100 Elo; needs a pseudo-legality stage-0 to return)
                        let moves = st.Buffers.[ply]
                        let scores = st.Scores.[ply]
                        let tried = st.TriedQuiets.[ply]
                        let n = generate pos moves
                        let k0 = st.Killers.[ply * 2]
                        let k1 = st.Killers.[ply * 2 + 1]
                        for idx in 0 .. n - 1 do
                            let m = moves.[idx]
                            scores.[idx] <-
                                if m = ttMove then 1_000_000
                                elif isCapture pos m then
                                    let victim =
                                        if moveFlag m = FlagEnPassant then Pawn
                                        else pieceType pos.Mailbox.[moveTo m]
                                    let attacker = pieceType pos.Mailbox.[moveFrom m]
                                    let mvvLva = victim * 8 - attacker
                                    if seeGe pos m 0 then 100_000 + mvvLva + (if moveFlag m = FlagPromo then 50_000 else 0)
                                    else 70_000 + mvvLva
                                elif moveFlag m = FlagPromo then 90_000 + movePromo m
                                elif m = k0 then 80_000
                                elif m = k1 then 79_999
                                else quietScore pos st m ply
                        let futilityOk =
                            not inChk && not isRoot && depth <= 6
                            && not (isMateScore alpha)
                            && staticEval + 120 + 100 * depth <= alpha
                        let mutable best = -Infinity
                        let mutable bestMove = NoMove
                        let mutable bound = BoundUpper
                        let mutable legalCount = 0
                        let mutable triedQuiets = 0
                        let mutable i = 0
                        let mutable cut = false
                        while not cut && i < n do
                            if true then
                                pickNext moves scores i n
                                let m = moves.[i]
                                let mScore = scores.[i]
                                let quiet = not (isCapture pos m) && moveFlag m <> FlagPromo
                                // late move pruning: enough legal moves searched at
                                // shallow depth => remaining quiets are skipped
                                let lmpThreshold =
                                    let b = lmpBase + depth * depth
                                    if useImproving && not improving then b / 2 else b
                                let lmpSkip =
                                    useLmp && quiet && not inChk && not isRoot
                                    && depth <= lmpMaxDepth
                                    && not (isMateScore alpha)
                                    && legalCount >= lmpThreshold
                                // SEE pruning: skip quiets that lose material on arrival
                                // (seeGe last — it's the costly test, && short-circuits)
                                let seeSkip =
                                    useSeeQuiet && quiet && not inChk && not isRoot
                                    && depth <= seeQDepth
                                    && legalCount > 0
                                    && not (isMateScore alpha)
                                    && not (seeGe pos m (-(seeQMargin * depth)))
                                let skipMove =
                                    m = excluded
                                    || (futilityOk && quiet && legalCount > 0)
                                    || lmpSkip
                                    || seeSkip
                                if not skipMove then
                                    st.ContIdx.[ply + 1] <- moveContIdx pos m
                                    makeMove pos m
                                    if not (wasLegal pos) then unmakeMove pos m
                                    else
                                        legalCount <- legalCount + 1
                                        if quiet && triedQuiets < 128 then
                                            tried.[triedQuiets] <- m
                                            triedQuiets <- triedQuiets + 1
                                        let newDepth = depth - 1 + (if m = ttMove then singularExt else 0)
                                        let v =
                                            if legalCount = 1 then
                                                -searchEx pos st newDepth (ply + 1) (-beta) (-alpha) true NoMove
                                            else
                                                let mutable r = 0
                                                if quiet && depth >= 3 && legalCount > 3 && not inChk then
                                                    r <- lmrTable.[min depth 63, min legalCount 63]
                                                    if mScore >= 79_999 then r <- r - 1   // killers
                                                    if r > newDepth - 1 then r <- newDepth - 1
                                                    if r < 0 then r <- 0
                                                let mutable v' = -searchEx pos st (newDepth - r) (ply + 1) (-alpha - 1) (-alpha) true NoMove
                                                if v' > alpha && r > 0 then
                                                    v' <- -searchEx pos st newDepth (ply + 1) (-alpha - 1) (-alpha) true NoMove
                                                if v' > alpha && v' < beta then
                                                    v' <- -searchEx pos st newDepth (ply + 1) (-beta) (-alpha) true NoMove
                                                v'
                                        unmakeMove pos m
                                        if st.Stop.Value then cut <- true
                                        else
                                            if v > best then
                                                best <- v
                                                bestMove <- m
                                            if v > alpha then
                                                alpha <- v
                                                bound <- BoundExact
                                                if isRoot then st.BestMove <- m
                                            if alpha >= beta then
                                                bound <- BoundLower
                                                if quiet then
                                                    let k0 = st.Killers.[ply * 2]
                                                    if m <> k0 then
                                                        st.Killers.[ply * 2 + 1] <- k0
                                                        st.Killers.[ply * 2] <- m
                                                    let bonus = depth * depth
                                                    let prev = if useContHist then st.ContIdx.[ply] else -1
                                                    let hIdx = historyIndex pos m
                                                    st.History.[hIdx] <- min 100_000 (st.History.[hIdx] + bonus)
                                                    if prev >= 0 then
                                                        let cIdx = prev * 768 + moveContIdx pos m
                                                        st.ContHist.[cIdx] <- min 100_000 (st.ContHist.[cIdx] + bonus)
                                                    for q in 0 .. triedQuiets - 1 do
                                                        if tried.[q] <> m then
                                                            let qIdx = historyIndex pos tried.[q]
                                                            st.History.[qIdx] <- max -100_000 (st.History.[qIdx] - bonus)
                                                            if prev >= 0 then
                                                                let qc = prev * 768 + moveContIdx pos tried.[q]
                                                                st.ContHist.[qc] <- max -100_000 (st.ContHist.[qc] - bonus)
                                                cut <- true
                                i <- i + 1
                        if st.Stop.Value then 0
                        elif legalCount = 0 then
                            if excluded <> NoMove then alpha
                            elif inChk then -(MateValue - ply)
                            else 0
                        else
                            if excluded = NoMove then
                                st.Tt.Store(key, bestMove, scoreToTt best ply, depth, bound)
                                // correction history: learn the eval bias for this pawn structure
                                if not inChk && not (isMateScore best)
                                   && (bestMove = NoMove || not (isCapture pos bestMove))
                                   && not (bound = BoundLower && best <= rawEval)
                                   && not (bound = BoundUpper && best >= rawEval) then
                                    let ci = corrIndex pos
                                    let diff = best - rawEval
                                    let w = min (depth + 1) corrW
                                    let e = st.CorrHist.[ci]
                                    let nv = (e * (64 - w) + diff * 16 * w) / 64
                                    st.CorrHist.[ci] <- max -4096 (min 4096 nv)
                            best

/// Standard entry point (no excluded move)
let search (pos: Position) (st: State) (depth: int) (ply: int) (alpha: int) (beta: int) (nullOk: bool) : int =
    searchEx pos st depth ply alpha beta nullOk NoMove

/// Walk the TT to build a PV string (best-effort)
let getPvString (pos: Position) (st: State) (maxLen: int) =
    let sb = System.Text.StringBuilder()
    let played = ResizeArray<Move>()
    let buf = Array.zeroCreate<Move> 256
    let mutable cont = true
    while cont && played.Count < maxLen do
        let e = st.Tt.Probe pos.Key
        if e.Hit && e.Move <> NoMove then
            let n = generate pos buf
            let mutable applied = false
            let mutable i = 0
            while not applied && i < n do
                if buf.[i] = e.Move then
                    makeMove pos e.Move
                    if wasLegal pos then applied <- true
                    else unmakeMove pos e.Move
                    i <- n
                i <- i + 1
            if applied then
                sb.Append(' ').Append(moveToUci e.Move) |> ignore
                played.Add e.Move
            else cont <- false
        else cont <- false
    for i in played.Count - 1 .. -1 .. 0 do
        unmakeMove pos played.[i]
    sb.ToString()

let private scoreString (s: int) =
    if isMateScore s then
        let mateIn = (MateValue - abs s + 1) / 2
        sprintf "mate %d" (if s > 0 then mateIn else -mateIn)
    else
        sprintf "cp %d" s

/// Helper thread body: feed the shared TT from staggered depths until stopped
let private helperLoop (pos: Position) (st: State) (maxDepth: int) (offset: int) =
    try
        st.ContIdx.[0] <- -1
        let mutable depth = 1 + (offset % 2)
        while not st.Stop.Value && depth <= maxDepth do
            search pos st depth 0 (-Infinity) Infinity true |> ignore
            depth <- depth + 1
    with ex -> logCrash "smp helper" ex

/// Iterative deepening driver. Returns the best move.
/// CONTRACT: the caller resets st.Stop on the command thread BEFORE calling —
/// think never un-sets a stop request (doing so races a concurrent "stop").
let think (pos: Position) (st: State) (limits: Limits) (verbose: bool) : Move =
    st.Nodes <- 0UL
    st.BestMove <- NoMove
    st.BestScore <- 0
    st.CompletedDepth <- 0
    st.ContIdx.[0] <- -1
    System.Array.Clear(st.Killers, 0, st.Killers.Length)
    for i in 0 .. st.History.Length - 1 do
        st.History.[i] <- st.History.[i] / 8
    st.Sw.Restart()

    let softMs, hardMs =
        if limits.Infinite then 0L, 0L
        elif limits.MoveTimeMs > 0L then limits.MoveTimeMs, limits.MoveTimeMs
        elif limits.TimeMs > 0L then
            let mtg = if limits.MovesToGo > 0 then int64 limits.MovesToGo + 1L else 32L
            let soft = limits.TimeMs / mtg + limits.IncMs / 2L
            let hard = min (limits.TimeMs / 4L) (soft * 5L)
            let safety = max 1L (limits.TimeMs - 50L)
            min soft safety, min (max soft hard) safety
        else 0L, 0L
    st.HardMs <- hardMs
    st.NodeLimit <- limits.NodeLimit

    let maxDepth = if limits.Depth > 0 then min limits.Depth (MaxPly - 8) else MaxPly - 8

    // lazy SMP: launch cached helpers on cloned positions, sharing TT + stop flag
    let helperThreads =
        if st.ThreadCount > 1 then
            ensureHelpers st
            [| for i in 0 .. st.ThreadCount - 2 ->
                 let hs = st.HelperStates.[i]
                 let hp = st.HelperPositions.[i]
                 hs.Nodes <- 0UL
                 hs.HardMs <- 0L
                 hs.NodeLimit <- 0UL
                 hs.BestMove <- NoMove
                 System.Array.Clear(hs.Killers, 0, hs.Killers.Length)
                 copyInto pos hp
                 let t = Thread((fun () -> helperLoop hp hs maxDepth i), 16 * 1024 * 1024)
                 t.IsBackground <- true
                 t.Start()
                 t |]
        else [||]

    let totalNodes () =
        let mutable t = st.Nodes
        for h in st.HelperStates do t <- t + h.Nodes
        t

    let mutable lastBest = NoMove
    let mutable prevScore = 0
    let mutable depth = 1
    while depth <= maxDepth && not st.Stop.Value do
        let mutable delta = 25
        let mutable alpha = if depth >= 5 then max (-Infinity) (prevScore - delta) else -Infinity
        let mutable betaW = if depth >= 5 then min Infinity (prevScore + delta) else Infinity
        let mutable score = 0
        let mutable settled = false
        while not settled && not st.Stop.Value do
            score <- search pos st depth 0 alpha betaW true
            if st.Stop.Value then ()
            elif score <= alpha then
                betaW <- (alpha + betaW) / 2
                alpha <- max (-Infinity) (alpha - delta)
                delta <- delta * 2
            elif score >= betaW then
                betaW <- min Infinity (betaW + delta)
                delta <- delta * 2
            else settled <- true
        if not st.Stop.Value || lastBest = NoMove then
            if st.BestMove <> NoMove then lastBest <- st.BestMove
            prevScore <- score
            if settled then
                st.BestScore <- score
                st.CompletedDepth <- depth
            if verbose && settled then
                let ms = st.Sw.ElapsedMilliseconds
                let nodes = totalNodes ()
                let nps = if ms > 0L then nodes * 1000UL / uint64 ms else 0UL
                printfn "info depth %d score %s nodes %d nps %d time %d pv%s"
                    depth (scoreString score) nodes nps ms (getPvString pos st 24)
        if softMs > 0L && st.Sw.ElapsedMilliseconds * 2L > softMs then
            st.Stop.Value <- true
        depth <- depth + 1

    if helperThreads.Length > 0 then
        st.Stop.Value <- true
        // helpers observe the flag at every node — a hang here is a real bug, surface it
        for t in helperThreads do t.Join()
    lastBest
