module AlterEgo.Nnue

// NNUE evaluation: (768*K -> 256)x2 -> 1 with ClippedReLU, K = king buckets.
//
// Features are perspective-relative (black mirrors color+rank) and, for K > 1,
// conditioned on the perspective's OWN king: 4 king-zone buckets + horizontal
// mirroring (own king on files e-h flips the board for that perspective).
// A king move that changes bucket/mirror state rebuilds that perspective's
// accumulator; all other moves update incrementally. Unmake is a stack pop.
//
// File format: magic, inputSize, hiddenSize, B1, W1, W2, B2 (little-endian
// int16/int32). King buckets are derived from inputSize (768*K) — the same
// loader reads both legacy 768 nets and bucketed nets.
//
// THREAD-SAFE BY DESIGN: no mutable evaluation state in this module; every
// function operates on caller-owned accumulators (Position carries the stack).

open System.IO
open System.Numerics
open AlterEgo.Types

[<Literal>]
let HiddenSize = 256
[<Literal>]
let AccSize = 512        // 2 * HiddenSize: White perspective, then Black
[<Literal>]
let QA = 255
[<Literal>]
let QB = 64
[<Literal>]
let Scale = 400
[<Literal>]
let Magic = 0x4E4E4541   // "AENN" little-endian
[<Literal>]
let MaxBuckets = 4

type Network =
    { W1: int16[]   // [InputSize * HiddenSize], feature-major
      B1: int16[]   // [HiddenSize]
      W2: int16[]   // [2 * HiddenSize]: side-to-move half, then opponent half
      B2: int32
      Buckets: int  // 1 (legacy 768) or 4 (king-bucketed, with mirroring)
      Mirror: bool
      Dual: bool    // factorized dual-king: + enemy-king-bucketed block (input 6144)
      // WDL head (v2 nets; empty for v1): float32, consumed at probe-root
      // frequency only — never in the search hot path
      Wdl: float32[]      // [3 * AccSize]: (win,draw,loss) x (stm 256 | opp 256)
      WdlBias: float32[] } // [3]

let mutable net: Network option = None

/// true once a network is loaded — Positions maintain their accumulator stacks
let mutable active = false

/// true when king moves require an accumulator rebuild (bucketed nets)
let mutable kingSensitive = false

/// true when ANY king move rebuilds BOTH perspectives (dual-king nets)
let mutable dualRebuild = false

// king-zone buckets, indexed by own-perspective king square AFTER mirroring
// (king always on files a-d here): castled-zone splits on ranks 1-2, then bands
let private bucketTable =
    Array.init 64 (fun sq ->
        let r = sq >>> 3
        let f = sq &&& 7
        if r <= 1 then (if f < 2 then 0 else 1)
        elif r <= 3 then 2
        else 3)

// valid input layouts: 768 (flat), 3072 (own-king 4-bucket), 6144 (dual-king factorized)
let private layoutOk (inp: int) = inp = 768 || inp = 3072 || inp = 6144

let private install (b1: int16[]) (w1: int16[]) (w2: int16[]) (b2: int32) (inp: int) (wdl: float32[]) (wdlB: float32[]) =
    let dual = inp = 6144
    let buckets = if dual then 4 else inp / 768
    net <- Some { W1 = w1; B1 = b1; W2 = w2; B2 = b2; Buckets = buckets; Mirror = inp > 768
                  Dual = dual; Wdl = wdl; WdlBias = wdlB }
    active <- true
    kingSensitive <- inp > 768
    dualRebuild <- dual

/// Read a net from an open reader. Format v1: magic, inputSize, ... .
/// Format v2: magic, -2, inputSize, ... + float32 WDL head appended.
let private readNet (br: BinaryReader) : bool =
    if br.ReadInt32() <> Magic then false
    else
        let second = br.ReadInt32()
        let version = if second = -2 then 2 else 1
        let inp = if version = 2 then br.ReadInt32() else second
        let hid = br.ReadInt32()
        if hid <> HiddenSize || not (layoutOk inp) then false
        else
            let b1 = Array.init HiddenSize (fun _ -> br.ReadInt16())
            let w1 = Array.init (inp * HiddenSize) (fun _ -> br.ReadInt16())
            let w2 = Array.init (2 * HiddenSize) (fun _ -> br.ReadInt16())
            let b2 = br.ReadInt32()
            let wdl, wdlB =
                if version = 2 then
                    Array.init (3 * AccSize) (fun _ -> br.ReadSingle()),
                    Array.init 3 (fun _ -> br.ReadSingle())
                else [||], [||]
            install b1 w1 w2 b2 inp wdl wdlB
            true

let load (path: string) : bool =
    try
        use br = new BinaryReader(File.OpenRead path)
        readNet br
    with _ -> false

let save (path: string) (n: Network) =
    use bw = new BinaryWriter(File.Create path)
    bw.Write Magic
    if n.Wdl.Length > 0 then bw.Write -2   // v2 marker
    bw.Write (if n.Dual then 6144 else n.Buckets * 768)
    bw.Write HiddenSize
    for v in n.B1 do bw.Write v
    for v in n.W1 do bw.Write v
    for v in n.W2 do bw.Write v
    bw.Write n.B2
    if n.Wdl.Length > 0 then
        for v in n.Wdl do bw.Write v
        for v in n.WdlBias do bw.Write v

/// Load the network embedded in the executable (the zero-configuration default)
let loadEmbedded () : bool =
    try
        let asm = System.Reflection.Assembly.GetExecutingAssembly()
        let s = asm.GetManifestResourceStream "AlterEgo.DefaultNet"
        if isNull s then false
        else
            use br = new BinaryReader(s)
            readNet br
    with _ -> false

/// Feature index of piece pc (0..11) on sq, from `persp`'s point of view,
/// given that perspective's own (raw, unoriented) king square.
let featureIndexK (persp: int) (pc: int) (sq: int) (kingSq: int) (buckets: int) (mir: bool) =
    let mutable oSq = if persp = White then sq else sq ^^^ 56
    let oPc = if persp = White then pc else (if pc < 6 then pc + 6 else pc - 6)
    if buckets <= 1 then oPc * 64 + oSq
    else
        let mutable oK = if persp = White then kingSq else kingSq ^^^ 56
        if mir && (oK &&& 7) >= 4 then
            oSq <- oSq ^^^ 7
            oK <- oK ^^^ 7
        (bucketTable.[oK] * 12 + oPc) * 64 + oSq

/// Enemy-king-bucketed block (factorized dual-king nets): same scheme, bucketed
/// and mirrored by the perspective's OPPONENT king, offset past block A (3072).
let featureIndexE (persp: int) (pc: int) (sq: int) (enemyKingSq: int) =
    let mutable oSq = if persp = White then sq else sq ^^^ 56
    let oPc = if persp = White then pc else (if pc < 6 then pc + 6 else pc - 6)
    let mutable oK = if persp = White then enemyKingSq else enemyKingSq ^^^ 56
    if (oK &&& 7) >= 4 then
        oSq <- oSq ^^^ 7
        oK <- oK ^^^ 7
    3072 + (bucketTable.[oK] * 12 + oPc) * 64 + oSq

let inline private addColumn (acc: int16[]) (accOff: int) (w: int16[]) (wOff: int) =
    let vc = Vector<int16>.Count
    let mutable j = 0
    while j < HiddenSize do
        let va = Vector<int16>(acc, accOff + j)
        let vw = Vector<int16>(w, wOff + j)
        (va + vw).CopyTo(acc, accOff + j)
        j <- j + vc

let inline private subColumn (acc: int16[]) (accOff: int) (w: int16[]) (wOff: int) =
    let vc = Vector<int16>.Count
    let mutable j = 0
    while j < HiddenSize do
        let va = Vector<int16>(acc, accOff + j)
        let vw = Vector<int16>(w, wOff + j)
        (va - vw).CopyTo(acc, accOff + j)
        j <- j + vc

/// duplicate stack.[ply] into stack.[ply + 1] (called at the top of makeMove)
let inline pushCopy (stack: int16[][]) (ply: int) =
    System.Array.Copy(stack.[ply], stack.[ply + 1], AccSize)

/// apply "piece pc appears on sq" to ONE perspective of acc
let addFeatureTo (acc: int16[]) (persp: int) (pc: int) (sq: int) (ownK: int) (enemyK: int) =
    match net with
    | Some n ->
        let off = if persp = White then 0 else HiddenSize
        addColumn acc off n.W1 (featureIndexK persp pc sq ownK n.Buckets n.Mirror * HiddenSize)
        if n.Dual then
            addColumn acc off n.W1 (featureIndexE persp pc sq enemyK * HiddenSize)
    | None -> ()

let removeFeatureFrom (acc: int16[]) (persp: int) (pc: int) (sq: int) (ownK: int) (enemyK: int) =
    match net with
    | Some n ->
        let off = if persp = White then 0 else HiddenSize
        subColumn acc off n.W1 (featureIndexK persp pc sq ownK n.Buckets n.Mirror * HiddenSize)
        if n.Dual then
            subColumn acc off n.W1 (featureIndexE persp pc sq enemyK * HiddenSize)
    | None -> ()

/// rebuild ONE perspective of acc from raw piece bitboards
let rebuildPersp (acc: int16[]) (persp: int) (byPiece: uint64[]) =
    match net with
    | Some n ->
        let off = if persp = White then 0 else HiddenSize
        let kOwn = BitOperations.TrailingZeroCount byPiece.[persp * 6 + King]
        let kEnemy = BitOperations.TrailingZeroCount byPiece.[(persp ^^^ 1) * 6 + King]
        for j in 0 .. HiddenSize - 1 do
            acc.[off + j] <- n.B1.[j]
        for pc in 0 .. 11 do
            let mutable bb = byPiece.[pc]
            while bb <> 0UL do
                let sq = BitOperations.TrailingZeroCount bb
                bb <- bb &&& (bb - 1UL)
                addColumn acc off n.W1 (featureIndexK persp pc sq kOwn n.Buckets n.Mirror * HiddenSize)
                if n.Dual then
                    addColumn acc off n.W1 (featureIndexE persp pc sq kEnemy * HiddenSize)
    | None -> ()

/// rebuild both perspectives (setFen / verification)
let buildInto (acc: int16[]) (byPiece: uint64[]) =
    rebuildPersp acc White byPiece
    rebuildPersp acc Black byPiece

/// clamp to [0, QA], widening int16 -> int32 dot product (SIMD)
let private dotHalf (acc: int16[]) (accOff: int) (w2: int16[]) (w2Off: int) =
    let vc = Vector<int16>.Count
    let zero = Vector<int16>.Zero
    let qa = Vector<int16>(int16 QA)
    let mutable sumV = Vector<int32>.Zero
    let mutable j = 0
    while j < HiddenSize do
        let va = Vector<int16>(acc, accOff + j)
        let clamped = Vector.Min(Vector.Max(va, zero), qa)
        let vw = Vector<int16>(w2, w2Off + j)
        let mutable cLo = Vector<int32>.Zero
        let mutable cHi = Vector<int32>.Zero
        let mutable wLo = Vector<int32>.Zero
        let mutable wHi = Vector<int32>.Zero
        Vector.Widen(clamped, &cLo, &cHi)
        Vector.Widen(vw, &wLo, &wHi)
        sumV <- sumV + cLo * wLo + cHi * wHi
        j <- j + vc
    Vector.Sum sumV

/// NNUE evaluation in centipawns from `stm`'s point of view, reading acc
let evaluateAcc (acc: int16[]) (stm: int) : int =
    match net with
    | None -> 0
    | Some n ->
        let stmOff = if stm = White then 0 else HiddenSize
        let oppOff = HiddenSize - stmOff
        let sum = dotHalf acc stmOff n.W2 0 + dotHalf acc oppOff n.W2 HiddenSize
        (sum + n.B2) * Scale / (QA * QB)

/// WDL probabilities from `stm`'s point of view, or ValueNone for v1 nets.
/// Float math + softmax: probe-root frequency ONLY, never the search hot path.
let evaluateWdl (acc: int16[]) (stm: int) : struct (float32 * float32 * float32) voption =
    match net with
    | Some n when n.Wdl.Length > 0 ->
        let stmOff = if stm = White then 0 else HiddenSize
        let oppOff = HiddenSize - stmOff
        let inv = 1.0f / float32 QA
        let logits = Array.zeroCreate<float32> 3
        for k in 0 .. 2 do
            let b = k * AccSize
            let mutable s = n.WdlBias.[k]
            for j in 0 .. HiddenSize - 1 do
                let a = int acc.[stmOff + j]
                let c = if a < 0 then 0 elif a > QA then QA else a
                s <- s + float32 c * inv * n.Wdl.[b + j]
            for j in 0 .. HiddenSize - 1 do
                let a = int acc.[oppOff + j]
                let c = if a < 0 then 0 elif a > QA then QA else a
                s <- s + float32 c * inv * n.Wdl.[b + HiddenSize + j]
            logits.[k] <- s
        let m = max logits.[0] (max logits.[1] logits.[2])
        let e0 = exp (logits.[0] - m)
        let e1 = exp (logits.[1] - m)
        let e2 = exp (logits.[2] - m)
        let z = e0 + e1 + e2
        ValueSome (struct (e0 / z, e1 / z, e2 / z))
    | _ -> ValueNone
