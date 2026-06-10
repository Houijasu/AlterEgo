module AlterEgo.TT

// Lock-free shared transposition table (lazy SMP). Each entry is two words:
// (key XOR data, data). A torn write from concurrent threads makes the XOR
// validation fail, so corrupted entries read as misses instead of lies.
// data layout: move(16) | score(16, biased int16) | depth(8) | bound(8)

open AlterEgo.Types

[<Literal>]
let BoundNone = 0uy
[<Literal>]
let BoundExact = 1uy
[<Literal>]
let BoundLower = 2uy   // fail-high: score is a lower bound
[<Literal>]
let BoundUpper = 3uy   // fail-low: score is an upper bound

[<Struct>]
type private Entry =
    { mutable KeyX: uint64
      mutable Data: uint64 }

[<Struct>]
type TTResult =
    { Hit: bool
      Move: Move
      Score: int
      Depth: int
      Bound: byte }

let inline private pack (m: Move) (score: int) (depth: int) (bound: byte) =
    uint64 m
    ||| (uint64 (uint16 (int16 score)) <<< 16)
    ||| (uint64 (byte (max 0 (min 255 depth))) <<< 32)
    ||| (uint64 bound <<< 40)

let inline private unpackMove (d: uint64) : Move = uint16 (d &&& 0xFFFFUL)
let inline private unpackScore (d: uint64) = int (int16 (uint16 ((d >>> 16) &&& 0xFFFFUL)))
let inline private unpackDepth (d: uint64) = int ((d >>> 32) &&& 0xFFUL)
let inline private unpackBound (d: uint64) = byte ((d >>> 40) &&& 0xFFUL)

type Table(sizeMb: int) =
    let count =
        let bytes = int64 sizeMb * 1024L * 1024L
        let n = bytes / 16L
        let mutable p = 1L
        while p * 2L <= n do p <- p * 2L
        int p
    let mask = uint64 (count - 1)
    let entries = Array.zeroCreate<Entry> count

    member _.Clear() = System.Array.Clear(entries, 0, count)

    member _.Probe(key: uint64) : TTResult =
        let e = entries.[int (key &&& mask)]
        if e.KeyX ^^^ e.Data = key then
            { Hit = true
              Move = unpackMove e.Data
              Score = unpackScore e.Data
              Depth = unpackDepth e.Data
              Bound = unpackBound e.Data }
        else
            { Hit = false; Move = NoMove; Score = 0; Depth = 0; Bound = BoundNone }

    member _.Store(key: uint64, m: Move, score: int, depth: int, bound: byte) =
        let idx = int (key &&& mask)
        let e = entries.[idx]
        let sameKey = e.KeyX ^^^ e.Data = key
        // replace if: different/garbage position, deeper, or exact bound
        if not sameKey || depth >= unpackDepth e.Data || bound = BoundExact then
            let mv = if m <> NoMove then m elif sameKey then unpackMove e.Data else NoMove
            let data = pack mv score depth bound
            entries.[idx].KeyX <- key ^^^ data
            entries.[idx].Data <- data
