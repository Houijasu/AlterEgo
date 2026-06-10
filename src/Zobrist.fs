module AlterEgo.Zobrist

// splitmix64 from a fixed seed — deterministic keys across runs
let mutable private state = 0x106689D45497FDB5UL

let private next () =
    state <- state + 0x9E3779B97F4A7C15UL
    let mutable z = state
    z <- (z ^^^ (z >>> 30)) * 0xBF58476D1CE4E5B9UL
    z <- (z ^^^ (z >>> 27)) * 0x94D049BB133111EBUL
    z ^^^ (z >>> 31)

/// psq.[piece * 64 + square], piece 0..11
let psq = Array.init (12 * 64) (fun _ -> next ())
let castling = Array.init 16 (fun _ -> next ())
let epFile = Array.init 8 (fun _ -> next ())
let side = next ()
