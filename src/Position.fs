module AlterEgo.Position

open AlterEgo.Types
open AlterEgo.Bitboards
open AlterEgo.Magics

[<Literal>]
let StartFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

[<Struct>]
type Undo =
    { Captured: int
      Castling: int
      Ep: int
      Half: int
      Key: uint64 }

type Position =
    { ByPiece: uint64[]     // 12 — index color*6+type
      ByColor: uint64[]     // 2
      Mailbox: int[]        // 64 — piece index or -1
      mutable Stm: int
      mutable Castling: int // WK=1 WQ=2 BK=4 BQ=8
      mutable Ep: int       // -1 = none
      mutable Half: int
      mutable Full: int
      mutable Key: uint64
      mutable PawnKey: uint64   // Zobrist over pawns only (correction history)
      Undos: Undo[]
      mutable Ply: int
      // NNUE accumulator stack, indexed by Ply (per-Position => thread-safe)
      AccStack: int16[][]
      mutable AccUpdating: bool
      // set during makeMove when a king move forces a perspective rebuild
      mutable AccRebuildW: bool
      mutable AccRebuildB: bool }

let create () =
    { ByPiece = Array.zeroCreate 12
      ByColor = Array.zeroCreate 2
      Mailbox = Array.create 64 -1
      Stm = White
      Castling = 0
      Ep = -1
      Half = 0
      Full = 1
      Key = 0UL
      PawnKey = 0UL
      Undos = Array.zeroCreate 1024
      Ply = 0
      AccStack = Array.init 1024 (fun _ -> Array.zeroCreate<int16> Nnue.AccSize)
      AccUpdating = false
      AccRebuildW = false
      AccRebuildB = false }

/// Copy src's full game state into dst (helper threads reuse allocated Positions)
let copyInto (src: Position) (dst: Position) =
    Array.blit src.ByPiece 0 dst.ByPiece 0 12
    Array.blit src.ByColor 0 dst.ByColor 0 2
    Array.blit src.Mailbox 0 dst.Mailbox 0 64
    dst.Stm <- src.Stm
    dst.Castling <- src.Castling
    dst.Ep <- src.Ep
    dst.Half <- src.Half
    dst.Full <- src.Full
    dst.Key <- src.Key
    dst.PawnKey <- src.PawnKey
    dst.Ply <- src.Ply
    dst.AccUpdating <- false
    dst.AccRebuildW <- false
    dst.AccRebuildB <- false
    Array.blit src.Undos 0 dst.Undos 0 src.Ply
    if Nnue.active then
        Array.blit src.AccStack.[src.Ply] 0 dst.AccStack.[dst.Ply] 0 Nnue.AccSize

let inline occupancy (pos: Position) = pos.ByColor.[0] ||| pos.ByColor.[1]

let inline addPiece (pos: Position) (pc: int) (sq: int) =
    let bb = bit sq
    pos.ByPiece.[pc] <- pos.ByPiece.[pc] ||| bb
    pos.ByColor.[pc / 6] <- pos.ByColor.[pc / 6] ||| bb
    pos.Mailbox.[sq] <- pc
    pos.Key <- pos.Key ^^^ Zobrist.psq.[pc * 64 + sq]
    if pc % 6 = Pawn then pos.PawnKey <- pos.PawnKey ^^^ Zobrist.psq.[pc * 64 + sq]
    if pos.AccUpdating then
        let acc = pos.AccStack.[pos.Ply]
        let wk = lsb pos.ByPiece.[King]
        let bk = lsb pos.ByPiece.[6 + King]
        if not pos.AccRebuildW then Nnue.addFeatureTo acc White pc sq wk bk
        if not pos.AccRebuildB then Nnue.addFeatureTo acc Black pc sq bk wk

let inline removePiece (pos: Position) (pc: int) (sq: int) =
    let bb = bit sq
    pos.ByPiece.[pc] <- pos.ByPiece.[pc] &&& ~~~bb
    pos.ByColor.[pc / 6] <- pos.ByColor.[pc / 6] &&& ~~~bb
    pos.Mailbox.[sq] <- -1
    pos.Key <- pos.Key ^^^ Zobrist.psq.[pc * 64 + sq]
    if pc % 6 = Pawn then pos.PawnKey <- pos.PawnKey ^^^ Zobrist.psq.[pc * 64 + sq]
    if pos.AccUpdating then
        let acc = pos.AccStack.[pos.Ply]
        let wk = lsb pos.ByPiece.[King]
        let bk = lsb pos.ByPiece.[6 + King]
        if not pos.AccRebuildW then Nnue.removeFeatureFrom acc White pc sq wk bk
        if not pos.AccRebuildB then Nnue.removeFeatureFrom acc Black pc sq bk wk

let inline movePiece (pos: Position) (pc: int) (fromSq: int) (toSq: int) =
    let bb = bit fromSq ||| bit toSq
    pos.ByPiece.[pc] <- pos.ByPiece.[pc] ^^^ bb
    pos.ByColor.[pc / 6] <- pos.ByColor.[pc / 6] ^^^ bb
    pos.Mailbox.[fromSq] <- -1
    pos.Mailbox.[toSq] <- pc
    pos.Key <- pos.Key ^^^ Zobrist.psq.[pc * 64 + fromSq] ^^^ Zobrist.psq.[pc * 64 + toSq]
    if pc % 6 = Pawn then
        pos.PawnKey <- pos.PawnKey ^^^ Zobrist.psq.[pc * 64 + fromSq] ^^^ Zobrist.psq.[pc * 64 + toSq]
    if pos.AccUpdating then
        let acc = pos.AccStack.[pos.Ply]
        let wk = lsb pos.ByPiece.[King]
        let bk = lsb pos.ByPiece.[6 + King]
        if not pos.AccRebuildW then
            Nnue.removeFeatureFrom acc White pc fromSq wk bk
            Nnue.addFeatureTo acc White pc toSq wk bk
        if not pos.AccRebuildB then
            Nnue.removeFeatureFrom acc Black pc fromSq bk wk
            Nnue.addFeatureTo acc Black pc toSq bk wk

let inline kingSquare (pos: Position) (color: int) =
    lsb pos.ByPiece.[color * 6 + King]

/// Is `sq` attacked by any piece of `by`?
let isAttacked (pos: Position) (sq: int) (by: int) =
    let occ = occupancy pos
    (pawnAttacks.[by ^^^ 1].[sq] &&& pos.ByPiece.[by * 6 + Pawn] <> 0UL)
    || (knightAttacks.[sq] &&& pos.ByPiece.[by * 6 + Knight] <> 0UL)
    || (kingAttacks.[sq] &&& pos.ByPiece.[by * 6 + King] <> 0UL)
    || (bishopAttacks sq occ &&& (pos.ByPiece.[by * 6 + Bishop] ||| pos.ByPiece.[by * 6 + Queen]) <> 0UL)
    || (rookAttacks sq occ &&& (pos.ByPiece.[by * 6 + Rook] ||| pos.ByPiece.[by * 6 + Queen]) <> 0UL)

let inline inCheck (pos: Position) =
    isAttacked pos (kingSquare pos pos.Stm) (pos.Stm ^^^ 1)

/// Hash the ep square only when `capturer` has a pawn that can actually take it:
/// positions whose ep right is immaterial then share a key (better TT sharing and
/// repetition detection). Board state is identical at set- and clear-time, so the
/// recomputed test can never desync the key.
let inline private epHashable (pos: Position) (capturer: int) =
    pawnAttacks.[capturer ^^^ 1].[pos.Ep] &&& pos.ByPiece.[capturer * 6 + Pawn] <> 0UL

// Castling-rights mask per square: rights removed when a piece moves from/to the square
let castleMask =
    let m = Array.create 64 15
    m.[0] <- 15 &&& ~~~2    // a1 -> WQ
    m.[4] <- 15 &&& ~~~3    // e1 -> WK|WQ
    m.[7] <- 15 &&& ~~~1    // h1 -> WK
    m.[56] <- 15 &&& ~~~8   // a8 -> BQ
    m.[60] <- 15 &&& ~~~12  // e8 -> BK|BQ
    m.[63] <- 15 &&& ~~~4   // h8 -> BK
    m

let makeMove (pos: Position) (m: Move) =
    let fromSq = moveFrom m
    let toSq = moveTo m
    let flag = moveFlag m
    let us = pos.Stm
    let them = us ^^^ 1
    let piece = pos.Mailbox.[fromSq]
    if Nnue.active then
        Nnue.pushCopy pos.AccStack pos.Ply
        pos.AccUpdating <- true
        // a king move can change its perspective's bucket/mirror: rebuild after.
        // dual-king nets condition BOTH perspectives on both kings => rebuild both.
        if Nnue.kingSensitive && pieceType piece = King then
            if Nnue.dualRebuild then
                pos.AccRebuildW <- true
                pos.AccRebuildB <- true
            elif us = White then pos.AccRebuildW <- true
            else pos.AccRebuildB <- true
    let captured = if flag = FlagEnPassant then them * 6 + Pawn else pos.Mailbox.[toSq]

    pos.Undos.[pos.Ply] <-
        { Captured = captured
          Castling = pos.Castling
          Ep = pos.Ep
          Half = pos.Half
          Key = pos.Key }
    pos.Ply <- pos.Ply + 1

    if pos.Ep >= 0 && epHashable pos us then pos.Key <- pos.Key ^^^ Zobrist.epFile.[fileOf pos.Ep]
    pos.Ep <- -1

    if flag = FlagEnPassant then
        let capSq = if us = White then toSq - 8 else toSq + 8
        removePiece pos (them * 6 + Pawn) capSq
    elif captured >= 0 then
        removePiece pos captured toSq

    movePiece pos piece fromSq toSq

    if flag = FlagPromo then
        removePiece pos piece toSq
        addPiece pos (us * 6 + movePromo m) toSq
    elif flag = FlagCastle then
        match toSq with
        | 6 -> movePiece pos (us * 6 + Rook) 7 5
        | 2 -> movePiece pos (us * 6 + Rook) 0 3
        | 62 -> movePiece pos (us * 6 + Rook) 63 61
        | 58 -> movePiece pos (us * 6 + Rook) 56 59
        | _ -> ()

    let oldCastling = pos.Castling
    pos.Castling <- pos.Castling &&& castleMask.[fromSq] &&& castleMask.[toSq]
    if oldCastling <> pos.Castling then
        pos.Key <- pos.Key ^^^ Zobrist.castling.[oldCastling] ^^^ Zobrist.castling.[pos.Castling]

    let isPawn = pieceType piece = Pawn
    if isPawn && abs (toSq - fromSq) = 16 then
        pos.Ep <- (fromSq + toSq) / 2
        if epHashable pos them then pos.Key <- pos.Key ^^^ Zobrist.epFile.[fileOf pos.Ep]

    pos.Half <- if isPawn || captured >= 0 then 0 else pos.Half + 1
    if us = Black then pos.Full <- pos.Full + 1
    pos.Stm <- them
    pos.Key <- pos.Key ^^^ Zobrist.side
    if pos.AccRebuildW then
        Nnue.rebuildPersp pos.AccStack.[pos.Ply] White pos.ByPiece
        pos.AccRebuildW <- false
    if pos.AccRebuildB then
        Nnue.rebuildPersp pos.AccStack.[pos.Ply] Black pos.ByPiece
        pos.AccRebuildB <- false
    pos.AccUpdating <- false

let unmakeMove (pos: Position) (m: Move) =
    pos.Ply <- pos.Ply - 1
    let u = pos.Undos.[pos.Ply]
    let them = pos.Stm
    let us = them ^^^ 1
    pos.Stm <- us
    let fromSq = moveFrom m
    let toSq = moveTo m
    let flag = moveFlag m

    if flag = FlagPromo then
        // remove promoted piece without touching Key (restored from undo below)
        removePiece pos (us * 6 + movePromo m) toSq
        addPiece pos (us * 6 + Pawn) fromSq
    else
        movePiece pos pos.Mailbox.[toSq] toSq fromSq

    if flag = FlagCastle then
        match toSq with
        | 6 -> movePiece pos (us * 6 + Rook) 5 7
        | 2 -> movePiece pos (us * 6 + Rook) 3 0
        | 62 -> movePiece pos (us * 6 + Rook) 61 63
        | 58 -> movePiece pos (us * 6 + Rook) 59 56
        | _ -> ()

    if flag = FlagEnPassant then
        let capSq = if us = White then toSq - 8 else toSq + 8
        addPiece pos (them * 6 + Pawn) capSq
    elif u.Captured >= 0 then
        addPiece pos u.Captured toSq

    pos.Castling <- u.Castling
    pos.Ep <- u.Ep
    pos.Half <- u.Half
    pos.Key <- u.Key
    if us = Black then pos.Full <- pos.Full - 1

let makeNull (pos: Position) =
    if Nnue.active then Nnue.pushCopy pos.AccStack pos.Ply
    pos.Undos.[pos.Ply] <-
        { Captured = -1
          Castling = pos.Castling
          Ep = pos.Ep
          Half = pos.Half
          Key = pos.Key }
    pos.Ply <- pos.Ply + 1
    if pos.Ep >= 0 && epHashable pos pos.Stm then pos.Key <- pos.Key ^^^ Zobrist.epFile.[fileOf pos.Ep]
    pos.Ep <- -1
    pos.Half <- pos.Half + 1
    pos.Stm <- pos.Stm ^^^ 1
    pos.Key <- pos.Key ^^^ Zobrist.side

let unmakeNull (pos: Position) =
    pos.Ply <- pos.Ply - 1
    let u = pos.Undos.[pos.Ply]
    pos.Stm <- pos.Stm ^^^ 1
    pos.Castling <- u.Castling
    pos.Ep <- u.Ep
    pos.Half <- u.Half
    pos.Key <- u.Key

/// Has the current position occurred before in the make/unmake history?
let isRepetition (pos: Position) =
    let limit = max 0 (pos.Ply - pos.Half)
    let mutable i = pos.Ply - 2
    let mutable found = false
    while not found && i >= limit do
        if pos.Undos.[i].Key = pos.Key then found <- true
        i <- i - 1
    found

/// Side to move has at least one non-pawn, non-king piece (null-move guard)
let inline hasNonPawnMaterial (pos: Position) =
    let us = pos.Stm
    (pos.ByPiece.[us * 6 + Knight] ||| pos.ByPiece.[us * 6 + Bishop]
     ||| pos.ByPiece.[us * 6 + Rook] ||| pos.ByPiece.[us * 6 + Queen]) <> 0UL

let setFen (pos: Position) (fen: string) =
    Array.fill pos.ByPiece 0 12 0UL
    Array.fill pos.ByColor 0 2 0UL
    Array.fill pos.Mailbox 0 64 -1
    pos.Key <- 0UL
    pos.PawnKey <- 0UL
    pos.Ply <- 0
    let parts = fen.Split(' ') |> Array.filter (fun s -> s <> "")
    // board
    let mutable sq = 56 // a8
    for c in parts.[0] do
        match c with
        | '/' -> sq <- sq - 16
        | c when System.Char.IsDigit c -> sq <- sq + int c - int '0'
        | c ->
            let pc =
                match System.Char.ToLower c with
                | 'p' -> Pawn | 'n' -> Knight | 'b' -> Bishop
                | 'r' -> Rook | 'q' -> Queen | 'k' -> King
                | _ -> failwithf "bad FEN piece: %c" c
            let color = if System.Char.IsUpper c then White else Black
            addPiece pos (color * 6 + pc) sq
            sq <- sq + 1
    // side to move
    pos.Stm <- if parts.Length > 1 && parts.[1] = "b" then Black else White
    if pos.Stm = Black then pos.Key <- pos.Key ^^^ Zobrist.side
    // castling
    pos.Castling <- 0
    if parts.Length > 2 then
        for c in parts.[2] do
            match c with
            | 'K' -> pos.Castling <- pos.Castling ||| 1
            | 'Q' -> pos.Castling <- pos.Castling ||| 2
            | 'k' -> pos.Castling <- pos.Castling ||| 4
            | 'q' -> pos.Castling <- pos.Castling ||| 8
            | _ -> ()
    pos.Key <- pos.Key ^^^ Zobrist.castling.[pos.Castling]
    // en passant
    pos.Ep <- if parts.Length > 3 && parts.[3] <> "-" then parseSquare parts.[3] else -1
    if pos.Ep >= 0 && epHashable pos pos.Stm then pos.Key <- pos.Key ^^^ Zobrist.epFile.[fileOf pos.Ep]
    // clocks
    pos.Half <- if parts.Length > 4 then int parts.[4] else 0
    pos.Full <- if parts.Length > 5 then int parts.[5] else 1
    if Nnue.active then Nnue.buildInto pos.AccStack.[pos.Ply] pos.ByPiece

let fromFen (fen: string) =
    let pos = create ()
    setFen pos fen
    pos

let print (pos: Position) =
    let chars = "PNBRQKpnbrqk"
    for r in 7 .. -1 .. 0 do
        printf "%d  " (r + 1)
        for f in 0 .. 7 do
            let pc = pos.Mailbox.[mkSquare f r]
            printf "%c " (if pc < 0 then '.' else chars.[pc])
        printfn ""
    printfn "   a b c d e f g h"
    printfn "stm=%s castling=%d ep=%d key=%016X" (if pos.Stm = White then "w" else "b") pos.Castling pos.Ep pos.Key
