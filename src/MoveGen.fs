module AlterEgo.MoveGen

open AlterEgo.Types
open AlterEgo.Bitboards
open AlterEgo.Magics
open AlterEgo.Position

/// Generate pseudo-legal moves into `moves`, returning the count.
/// Castling is fully legality-checked here; everything else is filtered by
/// make + king-safety check at the call site.
let generate (pos: Position) (moves: Move[]) : int =
    let us = pos.Stm
    let them = us ^^^ 1
    let occ = occupancy pos
    let mine = pos.ByColor.[us]
    let theirs = pos.ByColor.[them]
    let empty = ~~~occ
    let mutable n = 0

    let inline push (m: Move) =
        moves.[n] <- m
        n <- n + 1

    let inline pushPromos (fromSq: int) (toSq: int) =
        push (mkMoveF fromSq toSq FlagPromo 3) // queen first
        push (mkMoveF fromSq toSq FlagPromo 0)
        push (mkMoveF fromSq toSq FlagPromo 1)
        push (mkMoveF fromSq toSq FlagPromo 2)

    // ---- pawns ----
    if us = White then
        let wp = pos.ByPiece.[Pawn]
        let mutable singles = (wp <<< 8) &&& empty
        let mutable doubles = ((singles &&& Rank3) <<< 8) &&& empty
        while singles <> 0UL do
            let t = lsb singles
            singles <- singles &&& (singles - 1UL)
            if t >= 56 then pushPromos (t - 8) t else push (mkMove (t - 8) t)
        while doubles <> 0UL do
            let t = lsb doubles
            doubles <- doubles &&& (doubles - 1UL)
            push (mkMove (t - 16) t)
        let mutable capL = ((wp &&& ~~~FileA) <<< 7) &&& theirs
        while capL <> 0UL do
            let t = lsb capL
            capL <- capL &&& (capL - 1UL)
            if t >= 56 then pushPromos (t - 7) t else push (mkMove (t - 7) t)
        let mutable capR = ((wp &&& ~~~FileH) <<< 9) &&& theirs
        while capR <> 0UL do
            let t = lsb capR
            capR <- capR &&& (capR - 1UL)
            if t >= 56 then pushPromos (t - 9) t else push (mkMove (t - 9) t)
    else
        let bp = pos.ByPiece.[6 + Pawn]
        let mutable singles = (bp >>> 8) &&& empty
        let mutable doubles = ((singles &&& Rank6) >>> 8) &&& empty
        while singles <> 0UL do
            let t = lsb singles
            singles <- singles &&& (singles - 1UL)
            if t <= 7 then pushPromos (t + 8) t else push (mkMove (t + 8) t)
        while doubles <> 0UL do
            let t = lsb doubles
            doubles <- doubles &&& (doubles - 1UL)
            push (mkMove (t + 16) t)
        let mutable capL = ((bp &&& ~~~FileH) >>> 7) &&& theirs
        while capL <> 0UL do
            let t = lsb capL
            capL <- capL &&& (capL - 1UL)
            if t <= 7 then pushPromos (t + 7) t else push (mkMove (t + 7) t)
        let mutable capR = ((bp &&& ~~~FileA) >>> 9) &&& theirs
        while capR <> 0UL do
            let t = lsb capR
            capR <- capR &&& (capR - 1UL)
            if t <= 7 then pushPromos (t + 9) t else push (mkMove (t + 9) t)

    // en passant
    if pos.Ep >= 0 then
        let mutable srcs = pawnAttacks.[them].[pos.Ep] &&& pos.ByPiece.[us * 6 + Pawn]
        while srcs <> 0UL do
            let s = lsb srcs
            srcs <- srcs &&& (srcs - 1UL)
            push (mkMoveF s pos.Ep FlagEnPassant 0)

    // ---- knights ----
    let mutable kn = pos.ByPiece.[us * 6 + Knight]
    while kn <> 0UL do
        let s = lsb kn
        kn <- kn &&& (kn - 1UL)
        let mutable att = knightAttacks.[s] &&& ~~~mine
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    // ---- bishops ----
    let mutable bs = pos.ByPiece.[us * 6 + Bishop]
    while bs <> 0UL do
        let s = lsb bs
        bs <- bs &&& (bs - 1UL)
        let mutable att = bishopAttacks s occ &&& ~~~mine
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    // ---- rooks ----
    let mutable rs = pos.ByPiece.[us * 6 + Rook]
    while rs <> 0UL do
        let s = lsb rs
        rs <- rs &&& (rs - 1UL)
        let mutable att = rookAttacks s occ &&& ~~~mine
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    // ---- queens ----
    let mutable qs = pos.ByPiece.[us * 6 + Queen]
    while qs <> 0UL do
        let s = lsb qs
        qs <- qs &&& (qs - 1UL)
        let mutable att = queenAttacks s occ &&& ~~~mine
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    // ---- king ----
    let ks = kingSquare pos us
    let mutable katt = kingAttacks.[ks] &&& ~~~mine
    while katt <> 0UL do
        let t = lsb katt
        katt <- katt &&& (katt - 1UL)
        push (mkMove ks t)

    // ---- castling (fully legal here) ----
    if us = White then
        if pos.Castling &&& 1 <> 0
           && occ &&& 0x60UL = 0UL
           && not (isAttacked pos 4 Black)
           && not (isAttacked pos 5 Black)
           && not (isAttacked pos 6 Black) then
            push (mkMoveF 4 6 FlagCastle 0)
        if pos.Castling &&& 2 <> 0
           && occ &&& 0xEUL = 0UL
           && not (isAttacked pos 4 Black)
           && not (isAttacked pos 3 Black)
           && not (isAttacked pos 2 Black) then
            push (mkMoveF 4 2 FlagCastle 0)
    else
        if pos.Castling &&& 4 <> 0
           && occ &&& 0x6000000000000000UL = 0UL
           && not (isAttacked pos 60 White)
           && not (isAttacked pos 61 White)
           && not (isAttacked pos 62 White) then
            push (mkMoveF 60 62 FlagCastle 0)
        if pos.Castling &&& 8 <> 0
           && occ &&& 0x0E00000000000000UL = 0UL
           && not (isAttacked pos 60 White)
           && not (isAttacked pos 59 White)
           && not (isAttacked pos 58 White) then
            push (mkMoveF 60 58 FlagCastle 0)

    n

/// After makeMove: was the move legal (mover's king not left in check)?
let inline wasLegal (pos: Position) =
    let mover = pos.Stm ^^^ 1
    not (isAttacked pos (kingSquare pos mover) pos.Stm)

/// Captures, en passant and queen promotions only (quiescence)
let generateCaptures (pos: Position) (moves: Move[]) : int =
    let us = pos.Stm
    let them = us ^^^ 1
    let occ = occupancy pos
    let mine = pos.ByColor.[us]
    let theirs = pos.ByColor.[them]
    let mutable n = 0

    let inline push (m: Move) =
        moves.[n] <- m
        n <- n + 1

    if us = White then
        let wp = pos.ByPiece.[Pawn]
        let mutable pushP = ((wp &&& 0x00FF000000000000UL) <<< 8) &&& ~~~occ
        while pushP <> 0UL do
            let t = lsb pushP
            pushP <- pushP &&& (pushP - 1UL)
            push (mkMoveF (t - 8) t FlagPromo 3)
        let mutable capL = ((wp &&& ~~~FileA) <<< 7) &&& theirs
        while capL <> 0UL do
            let t = lsb capL
            capL <- capL &&& (capL - 1UL)
            if t >= 56 then push (mkMoveF (t - 7) t FlagPromo 3) else push (mkMove (t - 7) t)
        let mutable capR = ((wp &&& ~~~FileH) <<< 9) &&& theirs
        while capR <> 0UL do
            let t = lsb capR
            capR <- capR &&& (capR - 1UL)
            if t >= 56 then push (mkMoveF (t - 9) t FlagPromo 3) else push (mkMove (t - 9) t)
    else
        let bp = pos.ByPiece.[6 + Pawn]
        let mutable pushP = ((bp &&& 0x000000000000FF00UL) >>> 8) &&& ~~~occ
        while pushP <> 0UL do
            let t = lsb pushP
            pushP <- pushP &&& (pushP - 1UL)
            push (mkMoveF (t + 8) t FlagPromo 3)
        let mutable capL = ((bp &&& ~~~FileH) >>> 7) &&& theirs
        while capL <> 0UL do
            let t = lsb capL
            capL <- capL &&& (capL - 1UL)
            if t <= 7 then push (mkMoveF (t + 7) t FlagPromo 3) else push (mkMove (t + 7) t)
        let mutable capR = ((bp &&& ~~~FileA) >>> 9) &&& theirs
        while capR <> 0UL do
            let t = lsb capR
            capR <- capR &&& (capR - 1UL)
            if t <= 7 then push (mkMoveF (t + 9) t FlagPromo 3) else push (mkMove (t + 9) t)

    if pos.Ep >= 0 then
        let mutable srcs = pawnAttacks.[them].[pos.Ep] &&& pos.ByPiece.[us * 6 + Pawn]
        while srcs <> 0UL do
            let s = lsb srcs
            srcs <- srcs &&& (srcs - 1UL)
            push (mkMoveF s pos.Ep FlagEnPassant 0)

    let mutable kn = pos.ByPiece.[us * 6 + Knight]
    while kn <> 0UL do
        let s = lsb kn
        kn <- kn &&& (kn - 1UL)
        let mutable att = knightAttacks.[s] &&& theirs
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    let mutable bs = pos.ByPiece.[us * 6 + Bishop]
    while bs <> 0UL do
        let s = lsb bs
        bs <- bs &&& (bs - 1UL)
        let mutable att = bishopAttacks s occ &&& theirs
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    let mutable rs = pos.ByPiece.[us * 6 + Rook]
    while rs <> 0UL do
        let s = lsb rs
        rs <- rs &&& (rs - 1UL)
        let mutable att = rookAttacks s occ &&& theirs
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    let mutable qs = pos.ByPiece.[us * 6 + Queen]
    while qs <> 0UL do
        let s = lsb qs
        qs <- qs &&& (qs - 1UL)
        let mutable att = queenAttacks s occ &&& theirs
        while att <> 0UL do
            let t = lsb att
            att <- att &&& (att - 1UL)
            push (mkMove s t)

    let ks = kingSquare pos us
    let mutable katt = kingAttacks.[ks] &&& theirs
    while katt <> 0UL do
        let t = lsb katt
        katt <- katt &&& (katt - 1UL)
        push (mkMove ks t)

    n

let private seeValue = [| 100; 320; 330; 500; 900; 20000 |]

/// All pieces of both colors attacking sq given occupancy
let attackersTo (pos: Position) (sq: int) (occ: uint64) =
    (pawnAttacks.[Black].[sq] &&& pos.ByPiece.[Pawn])
    ||| (pawnAttacks.[White].[sq] &&& pos.ByPiece.[6 + Pawn])
    ||| (knightAttacks.[sq] &&& (pos.ByPiece.[Knight] ||| pos.ByPiece.[6 + Knight]))
    ||| (kingAttacks.[sq] &&& (pos.ByPiece.[King] ||| pos.ByPiece.[6 + King]))
    ||| (bishopAttacks sq occ
         &&& (pos.ByPiece.[Bishop] ||| pos.ByPiece.[6 + Bishop]
              ||| pos.ByPiece.[Queen] ||| pos.ByPiece.[6 + Queen]))
    ||| (rookAttacks sq occ
         &&& (pos.ByPiece.[Rook] ||| pos.ByPiece.[6 + Rook]
              ||| pos.ByPiece.[Queen] ||| pos.ByPiece.[6 + Queen]))

/// Static exchange evaluation: does move m win at least `threshold` material?
/// Allocation-free swap algorithm (Stockfish-style).
let seeGe (pos: Position) (m: Move) (threshold: int) : bool =
    if moveFlag m <> FlagNormal then 0 >= threshold
    else
        let fromSq = moveFrom m
        let toSq = moveTo m
        let mutable swap =
            (if pos.Mailbox.[toSq] >= 0 then seeValue.[pieceType pos.Mailbox.[toSq]] else 0) - threshold
        if swap < 0 then false
        else
            swap <- seeValue.[pieceType pos.Mailbox.[fromSq]] - swap
            if swap <= 0 then true
            else
                let mutable occ = occupancy pos ^^^ bit fromSq ^^^ bit toSq
                let mutable stm = pos.Stm
                let mutable attackers = attackersTo pos toSq occ
                let mutable res = 1
                let mutable finished = false
                let bq = pos.ByPiece.[Bishop] ||| pos.ByPiece.[6 + Bishop] ||| pos.ByPiece.[Queen] ||| pos.ByPiece.[6 + Queen]
                let rq = pos.ByPiece.[Rook] ||| pos.ByPiece.[6 + Rook] ||| pos.ByPiece.[Queen] ||| pos.ByPiece.[6 + Queen]
                while not finished do
                    stm <- stm ^^^ 1
                    attackers <- attackers &&& occ
                    let stmAttackers = attackers &&& pos.ByColor.[stm]
                    if stmAttackers = 0UL then finished <- true
                    else
                        res <- res ^^^ 1
                        // least valuable attacker
                        let mutable pt = -1
                        let mutable ptBb = 0UL
                        let mutable t = Pawn
                        while pt < 0 && t <= King do
                            let bb = stmAttackers &&& pos.ByPiece.[stm * 6 + t]
                            if bb <> 0UL then
                                pt <- t
                                ptBb <- bb
                            t <- t + 1
                        if pt = King then
                            // king can capture only if no defenders remain
                            if attackers &&& occ &&& pos.ByColor.[stm ^^^ 1] <> 0UL then res <- res ^^^ 1
                            finished <- true
                        else
                            swap <- seeValue.[pt] - swap
                            if swap < res then finished <- true
                            else
                                occ <- occ ^^^ (ptBb &&& (0UL - ptBb))   // remove one attacker
                                // x-ray updates
                                if pt = Pawn || pt = Bishop || pt = Queen then
                                    attackers <- attackers ||| (bishopAttacks toSq occ &&& bq)
                                if pt = Rook || pt = Queen then
                                    attackers <- attackers ||| (rookAttacks toSq occ &&& rq)
                res <> 0
