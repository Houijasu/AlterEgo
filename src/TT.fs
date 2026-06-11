module AlterEgo.TT

// Lock-free shared transposition table (lazy SMP). Clustered: 4 entries per
// bucket, one 64-byte cache line. Each entry is two words: (key XOR data,
// data). A torn write from concurrent threads makes the XOR validation fail,
// so corrupted entries read as misses instead of lies.
// data layout: move(16) | score(16, biased int16) | depth(8) | bound(8) | gen(8)
//
// Replacement: a same-key entry is overwritten when the new fact is at least
// as deep or exact (otherwise only its generation is refreshed). On a fresh
// key the victim is the bucket's least valuable entry by depth - 8*age, so
// stale deep entries eventually yield to current shallow ones.

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

let inline private pack (m: Move) (score: int) (depth: int) (bound: byte) (gen: int) =
    uint64 m
    ||| (uint64 (uint16 (int16 score)) <<< 16)
    ||| (uint64 (byte (max 0 (min 255 depth))) <<< 32)
    ||| (uint64 bound <<< 40)
    ||| (uint64 (byte gen) <<< 48)

let inline private unpackMove (d: uint64) : Move = uint16 (d &&& 0xFFFFUL)
let inline private unpackScore (d: uint64) = int (int16 (uint16 ((d >>> 16) &&& 0xFFFFUL)))
let inline private unpackDepth (d: uint64) = int ((d >>> 32) &&& 0xFFUL)
let inline private unpackBound (d: uint64) = byte ((d >>> 40) &&& 0xFFUL)
let inline private unpackGen (d: uint64) = int ((d >>> 48) &&& 0xFFUL)

[<Literal>]
let private BucketSize = 4

type Table(sizeMb: int) =
    let bucketCount =
        let bytes = int64 sizeMb * 1024L * 1024L
        let n = bytes / 64L   // 4 entries x 16 bytes
        let mutable p = 1L
        while p * 2L <= n do p <- p * 2L
        int p
    let mask = uint64 (bucketCount - 1)
    let entries = Array.zeroCreate<Entry> (bucketCount * BucketSize)
    let mutable generation = 0

    /// Bump once per search so replacement can age out stale entries.
    member _.NewSearch() = generation <- (generation + 1) &&& 0xFF

    member _.Clear() =
        System.Array.Clear(entries, 0, entries.Length)
        generation <- 0

    member _.Probe(key: uint64) : TTResult =
        let b = int (key &&& mask) * BucketSize
        let mutable i = 0
        let mutable result = { Hit = false; Move = NoMove; Score = 0; Depth = 0; Bound = BoundNone }
        while i < BucketSize do
            let e = entries.[b + i]
            if e.KeyX ^^^ e.Data = key then
                // refresh generation so entries the search still touches survive
                if unpackGen e.Data <> generation then
                    let data = (e.Data &&& ~~~(0xFFUL <<< 48)) ||| (uint64 (byte generation) <<< 48)
                    entries.[b + i].KeyX <- key ^^^ data
                    entries.[b + i].Data <- data
                result <-
                    { Hit = true
                      Move = unpackMove e.Data
                      Score = unpackScore e.Data
                      Depth = unpackDepth e.Data
                      Bound = unpackBound e.Data }
                i <- BucketSize
            else i <- i + 1
        result

    member _.Store(key: uint64, m: Move, score: int, depth: int, bound: byte) =
        let b = int (key &&& mask) * BucketSize
        // find a same-key slot, else the least valuable victim (depth vs age)
        let mutable idx = -1
        let mutable sameKey = false
        let mutable victim = b
        let mutable victimValue = System.Int32.MaxValue
        let mutable i = 0
        while i < BucketSize do
            let e = entries.[b + i]
            if e.KeyX ^^^ e.Data = key then
                idx <- b + i
                sameKey <- true
                i <- BucketSize
            else
                let relAge = (256 + generation - unpackGen e.Data) &&& 0xFF
                let v = unpackDepth e.Data - 8 * relAge
                if v < victimValue then
                    victimValue <- v
                    victim <- b + i
                i <- i + 1
        let idx = if sameKey then idx else victim
        let old = entries.[idx]
        // same key: replace only with facts at least as deep, or exact bounds;
        // shallower facts just refresh the entry's generation
        if sameKey && depth < unpackDepth old.Data && bound <> BoundExact then
            let data = (old.Data &&& ~~~(0xFFUL <<< 48)) ||| (uint64 (byte generation) <<< 48)
            entries.[idx].KeyX <- key ^^^ data
            entries.[idx].Data <- data
        else
            let mv = if m <> NoMove then m elif sameKey then unpackMove old.Data else NoMove
            let data = pack mv score depth bound generation
            entries.[idx].KeyX <- key ^^^ data
            entries.[idx].Data <- data
