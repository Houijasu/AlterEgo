module AlterEgo.Perft

open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen

// Per-ply move buffers — zero allocation during the walk
let private buffers = Array.init 128 (fun _ -> Array.zeroCreate<Move> 256)

let rec perft (pos: Position) (depth: int) : uint64 =
    let moves = buffers.[depth]
    let n = generate pos moves
    if depth = 1 then
        // bulk-count legal moves at the frontier
        let mutable count = 0UL
        for i in 0 .. n - 1 do
            makeMove pos moves.[i]
            if wasLegal pos then count <- count + 1UL
            unmakeMove pos moves.[i]
        count
    else
        let mutable nodes = 0UL
        for i in 0 .. n - 1 do
            makeMove pos moves.[i]
            if wasLegal pos then nodes <- nodes + perft pos (depth - 1)
            unmakeMove pos moves.[i]
        nodes

let perftRoot (pos: Position) (depth: int) =
    if depth <= 0 then 1UL else perft pos depth

/// Per-root-move breakdown for debugging
let divide (pos: Position) (depth: int) =
    let moves = Array.zeroCreate<Move> 256
    let n = generate pos moves
    let mutable total = 0UL
    for i in 0 .. n - 1 do
        makeMove pos moves.[i]
        if wasLegal pos then
            let cnt = if depth <= 1 then 1UL else perft pos (depth - 1)
            printfn "%s: %d" (moveToUci moves.[i]) cnt
            total <- total + cnt
        unmakeMove pos moves.[i]
    printfn "total: %d" total
    total
