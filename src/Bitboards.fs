module AlterEgo.Bitboards

open System.Numerics
open AlterEgo.Types

let inline popcount (b: uint64) = BitOperations.PopCount b
let inline lsb (b: uint64) = BitOperations.TrailingZeroCount b
let inline bit (sq: int) = 1UL <<< sq

[<Literal>]
let FileA = 0x0101010101010101UL
[<Literal>]
let FileH = 0x8080808080808080UL
[<Literal>]
let Rank1 = 0x00000000000000FFUL
[<Literal>]
let Rank3 = 0x0000000000FF0000UL
[<Literal>]
let Rank6 = 0x0000FF0000000000UL
[<Literal>]
let Rank8 = 0xFF00000000000000UL

let private leaperAttacks (deltas: (int * int) list) =
    Array.init 64 (fun sq ->
        let f = fileOf sq
        let r = rankOf sq
        let mutable bb = 0UL
        for (df, dr) in deltas do
            let nf = f + df
            let nr = r + dr
            if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 then
                bb <- bb ||| bit (mkSquare nf nr)
        bb)

let knightAttacks =
    leaperAttacks [ (1, 2); (2, 1); (2, -1); (1, -2); (-1, -2); (-2, -1); (-2, 1); (-1, 2) ]

let kingAttacks =
    leaperAttacks [ (1, 0); (1, 1); (0, 1); (-1, 1); (-1, 0); (-1, -1); (0, -1); (1, -1) ]

/// pawnAttacks.[color].[sq] = squares attacked by a pawn of `color` standing on `sq`
let pawnAttacks =
    [| leaperAttacks [ (-1, 1); (1, 1) ]      // White
       leaperAttacks [ (-1, -1); (1, -1) ] |] // Black

/// Slow ray attack computation (init + verification only, never in search)
let slidingAttack (deltas: (int * int) list) (sq: int) (occ: uint64) =
    let mutable bb = 0UL
    for (df, dr) in deltas do
        let mutable f = fileOf sq + df
        let mutable r = rankOf sq + dr
        let mutable stop = false
        while not stop && f >= 0 && f < 8 && r >= 0 && r < 8 do
            let s = mkSquare f r
            bb <- bb ||| bit s
            if occ &&& bit s <> 0UL then stop <- true
            f <- f + df
            r <- r + dr
    bb

let rookDeltas = [ (1, 0); (-1, 0); (0, 1); (0, -1) ]
let bishopDeltas = [ (1, 1); (1, -1); (-1, 1); (-1, -1) ]
