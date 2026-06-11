module AlterEgo.Tests

open Xunit
open AlterEgo.Types
open AlterEgo.Position
open AlterEgo.MoveGen
open AlterEgo.Perft

// ---- Types: square parsing and move encoding ----

[<Fact>]
let ``parseSquare accepts valid squares`` () =
    Assert.Equal(0, parseSquare "a1")
    Assert.Equal(63, parseSquare "h8")
    Assert.Equal(28, parseSquare "e4")

[<Fact>]
let ``parseSquare rejects out-of-range input`` () =
    Assert.Equal(-1, parseSquare "i1")
    Assert.Equal(-1, parseSquare "a9")
    Assert.Equal(-1, parseSquare "z@")
    Assert.Equal(-1, parseSquare "a")
    Assert.Equal(-1, parseSquare "")

[<Fact>]
let ``move encoding roundtrips all fields`` () =
    let m = mkMoveF 12 28 FlagPromo 3
    Assert.Equal(12, moveFrom m)
    Assert.Equal(28, moveTo m)
    Assert.Equal(FlagPromo, moveFlag m)
    Assert.Equal(Queen, movePromo m)
    Assert.Equal("e2e4q", moveToUci m)
    let q = mkMove 6 21
    Assert.Equal(FlagNormal, moveFlag q)
    Assert.Equal("g1f3", moveToUci q)

// ---- TT: pack/unpack via public Store/Probe ----

[<Fact>]
let ``TT stores and retrieves entries exactly`` () =
    let tt = AlterEgo.TT.Table(1)
    let key = 0xDEADBEEFCAFE1234UL
    tt.Store(key, mkMove 12 28, -12345, 17, AlterEgo.TT.BoundLower)
    let e = tt.Probe key
    Assert.True e.Hit
    Assert.Equal(mkMove 12 28, e.Move)
    Assert.Equal(-12345, e.Score)
    Assert.Equal(17, e.Depth)
    Assert.Equal(AlterEgo.TT.BoundLower, e.Bound)

[<Fact>]
let ``TT misses on unknown key`` () =
    let tt = AlterEgo.TT.Table(1)
    tt.Store(0x1111UL, mkMove 0 1, 50, 5, AlterEgo.TT.BoundExact)
    let e = tt.Probe 0x2222UL
    Assert.False e.Hit

// ---- NNUE feature indexing: all three layouts ----

[<Fact>]
let ``flat feature indices stay in block bounds`` () =
    for pc in 0 .. 11 do
        for sq in [ 0; 7; 28; 56; 63 ] do
            let i = AlterEgo.Nnue.featureIndexK White pc sq 4 1 false
            Assert.InRange(i, 0, 767)
            let j = AlterEgo.Nnue.featureIndexK Black pc sq 60 1 false
            Assert.InRange(j, 0, 767)

[<Fact>]
let ``bucketed feature indices stay in block bounds`` () =
    for pc in 0 .. 11 do
        for k in [ 0; 4; 28; 36; 60; 63 ] do
            let i = AlterEgo.Nnue.featureIndexK White pc 28 k 4 true
            Assert.InRange(i, 0, 3071)

[<Fact>]
let ``enemy-king block indices live in the second block`` () =
    for pc in 0 .. 11 do
        for k in [ 0; 4; 36; 63 ] do
            let i = AlterEgo.Nnue.featureIndexE White pc 28 k
            Assert.InRange(i, 3072, 6143)

[<Fact>]
let ``perspective symmetry: black mirrors white`` () =
    // a white pawn on e2 from White's view = a black pawn on e7 from Black's view
    let w = AlterEgo.Nnue.featureIndexK White Pawn 12 4 4 true
    let b = AlterEgo.Nnue.featureIndexK Black (6 + Pawn) (12 ^^^ 56) (4 ^^^ 56) 4 true
    Assert.Equal(w, b)

// ---- Position / movegen: perft spot checks and make/unmake integrity ----

[<Fact>]
let ``perft startpos shallow depths`` () =
    let pos = fromFen StartFen
    Assert.Equal(20UL, perftRoot pos 1)
    Assert.Equal(400UL, perftRoot pos 2)
    Assert.Equal(8902UL, perftRoot pos 3)

[<Fact>]
let ``perft kiwipete depth 2`` () =
    let pos = fromFen "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    Assert.Equal(2039UL, perftRoot pos 2)

[<Fact>]
let ``make-unmake restores key and accumulators`` () =
    let pos = fromFen StartFen
    let key = pos.Key
    let mg = pos.Mg
    let m = mkMove 12 28   // e2e4
    makeMove pos m
    Assert.Equal(20, pos.Ep)    // e3
    Assert.NotEqual<uint64>(key, pos.Key)
    unmakeMove pos m
    Assert.Equal(key, pos.Key)
    Assert.Equal(mg, pos.Mg)
    Assert.Equal(0, pos.Ply)

[<Fact>]
let ``setFen rejects nothing silently: castling and ep parsed`` () =
    let pos = fromFen "r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 1"
    Assert.Equal(15, pos.Castling)
    Assert.Equal(parseSquare "e3", pos.Ep)
    Assert.Equal(Black, pos.Stm)

[<Fact>]
let ``illegal moves are filtered by wasLegal`` () =
    // white king on e1 pinned-rook scenario: moving the pinned piece is illegal
    let pos = fromFen "4r3/8/8/8/8/8/4R3/4K3 w - - 0 1"
    let buf = Array.zeroCreate<Move> 256
    let n = generate pos buf
    let mutable sawIllegal = false
    for i in 0 .. n - 1 do
        // rook leaving the e-file exposes the king: must be flagged illegal
        if moveFrom buf.[i] = 12 && fileOf (moveTo buf.[i]) <> 4 then
            makeMove pos buf.[i]
            if wasLegal pos then sawIllegal <- true
            unmakeMove pos buf.[i]
    Assert.False sawIllegal
