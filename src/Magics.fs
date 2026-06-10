module AlterEgo.Magics

open AlterEgo.Types
open AlterEgo.Bitboards

[<Struct>]
type Magic =
    { Mask: uint64
      Factor: uint64
      Shift: int
      Table: uint64[] }

let mutable private rngState = 0x9E3779B97F4A7C15UL

let private nextRand () =
    rngState <- rngState ^^^ (rngState <<< 13)
    rngState <- rngState ^^^ (rngState >>> 7)
    rngState <- rngState ^^^ (rngState <<< 17)
    rngState

let private sparseRand () = nextRand () &&& nextRand () &&& nextRand ()

/// Relevant-occupancy mask: ray squares excluding the board-edge terminus of each ray
let private relevantMask (deltas: (int * int) list) (sq: int) =
    let mutable bb = 0UL
    for (df, dr) in deltas do
        let mutable f = fileOf sq + df
        let mutable r = rankOf sq + dr
        while f + df >= 0 && f + df < 8 && r + dr >= 0 && r + dr < 8
              && f >= 0 && f < 8 && r >= 0 && r < 8 do
            bb <- bb ||| bit (mkSquare f r)
            f <- f + df
            r <- r + dr
    bb

/// Find a collision-free magic for one square; correctness guaranteed by construction
let private findMagic (deltas: (int * int) list) (sq: int) : Magic =
    let mask = relevantMask deltas sq
    let bits = popcount mask
    let size = 1 <<< bits
    let shift = 64 - bits
    let occs = Array.zeroCreate<uint64> size
    let refs = Array.zeroCreate<uint64> size
    // Carry-Rippler subset enumeration
    let mutable occ = 0UL
    for i in 0 .. size - 1 do
        occs.[i] <- occ
        refs.[i] <- slidingAttack deltas sq occ
        occ <- (occ - mask) &&& mask
    let table = Array.zeroCreate<uint64> size
    let used = Array.zeroCreate<bool> size
    let mutable factor = 0UL
    let mutable found = false
    while not found do
        factor <- sparseRand ()
        if popcount ((mask * factor) >>> 56) >= 6 then
            System.Array.Clear(used, 0, size)
            let mutable ok = true
            let mutable i = 0
            while ok && i < size do
                let idx = int ((occs.[i] * factor) >>> shift)
                if not used.[idx] then
                    used.[idx] <- true
                    table.[idx] <- refs.[i]
                elif table.[idx] <> refs.[i] then
                    ok <- false
                i <- i + 1
            found <- ok
    { Mask = mask; Factor = factor; Shift = shift; Table = table }

let rookMagics = Array.init 64 (findMagic rookDeltas)
let bishopMagics = Array.init 64 (findMagic bishopDeltas)

let inline rookAttacks (sq: int) (occ: uint64) =
    let m = rookMagics.[sq]
    m.Table.[int (((occ &&& m.Mask) * m.Factor) >>> m.Shift)]

let inline bishopAttacks (sq: int) (occ: uint64) =
    let m = bishopMagics.[sq]
    m.Table.[int (((occ &&& m.Mask) * m.Factor) >>> m.Shift)]

let inline queenAttacks (sq: int) (occ: uint64) =
    rookAttacks sq occ ||| bishopAttacks sq occ

/// Force initialization (call once at startup so timing runs are clean)
let init () =
    rookMagics.[0].Table.Length + bishopMagics.[0].Table.Length |> ignore
