module AlterEgo.Nnue

// NNUE evaluation, architecture (768 -> 256)x2 -> 1 with ClippedReLU.
// Feature: piece (12) x square (64), perspective-relative (black mirrors color+rank).
// Quantization: W1/B1 scaled by QA, W2 scaled by QB, B2 by QA*QB; cp = raw * Scale / (QA*QB).
//
// THREAD-SAFE BY DESIGN: this module holds no mutable evaluation state. Every
// function operates on caller-owned accumulator arrays (Position carries an
// accumulator stack indexed by its ply). The only globals are the loaded
// network (written once at load, read-only afterwards) and the `active` flag.

open System.IO
open System.Numerics
open AlterEgo.Types

[<Literal>]
let InputSize = 768
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

type Network =
    { W1: int16[]   // [InputSize * HiddenSize], feature-major
      B1: int16[]   // [HiddenSize]
      W2: int16[]   // [2 * HiddenSize]: side-to-move half, then opponent half
      B2: int32 }

let mutable net: Network option = None

/// true once a network is loaded — Positions maintain their accumulator stacks
let mutable active = false

let private readNet (br: BinaryReader) : bool =
    if br.ReadInt32() <> Magic then false
    elif br.ReadInt32() <> InputSize || br.ReadInt32() <> HiddenSize then false
    else
        let b1 = Array.init HiddenSize (fun _ -> br.ReadInt16())
        let w1 = Array.init (InputSize * HiddenSize) (fun _ -> br.ReadInt16())
        let w2 = Array.init (2 * HiddenSize) (fun _ -> br.ReadInt16())
        let b2 = br.ReadInt32()
        net <- Some { W1 = w1; B1 = b1; W2 = w2; B2 = b2 }
        active <- true
        true

let load (path: string) : bool =
    try
        use br = new BinaryReader(File.OpenRead path)
        readNet br
    with _ -> false

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

let save (path: string) (n: Network) =
    use bw = new BinaryWriter(File.Create path)
    bw.Write Magic
    bw.Write InputSize
    bw.Write HiddenSize
    for v in n.B1 do bw.Write v
    for v in n.W1 do bw.Write v
    for v in n.W2 do bw.Write v
    bw.Write n.B2

/// Feature index of piece pc (0..11) on sq, from `persp`'s point of view
let inline featureIndex (persp: int) (pc: int) (sq: int) =
    if persp = White then pc * 64 + sq
    else (if pc < 6 then pc + 6 else pc - 6) * 64 + (sq ^^^ 56)

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

/// apply "piece pc appears on sq" to both perspectives of acc
let addFeatureTo (acc: int16[]) (pc: int) (sq: int) =
    match net with
    | Some n ->
        addColumn acc 0 n.W1 (featureIndex White pc sq * HiddenSize)
        addColumn acc HiddenSize n.W1 (featureIndex Black pc sq * HiddenSize)
    | None -> ()

let removeFeatureFrom (acc: int16[]) (pc: int) (sq: int) =
    match net with
    | Some n ->
        subColumn acc 0 n.W1 (featureIndex White pc sq * HiddenSize)
        subColumn acc HiddenSize n.W1 (featureIndex Black pc sq * HiddenSize)
    | None -> ()

/// rebuild acc from raw piece bitboards (setFen / verification)
let buildInto (acc: int16[]) (byPiece: uint64[]) =
    match net with
    | Some n ->
        for j in 0 .. HiddenSize - 1 do
            acc.[j] <- n.B1.[j]
            acc.[HiddenSize + j] <- n.B1.[j]
        for pc in 0 .. 11 do
            let mutable bb = byPiece.[pc]
            while bb <> 0UL do
                let sq = BitOperations.TrailingZeroCount bb
                bb <- bb &&& (bb - 1UL)
                addColumn acc 0 n.W1 (featureIndex White pc sq * HiddenSize)
                addColumn acc HiddenSize n.W1 (featureIndex Black pc sq * HiddenSize)
    | None -> ()

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
