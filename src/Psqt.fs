module AlterEgo.Psqt

open AlterEgo.Types

// Tapered PST data (Michniewski-style) — replaced by NNUE in M5.
// Tables written visually: first row = rank 8. White lookup flips rank (sq ^ 56).

let pieceValue = [| 100; 320; 330; 500; 900; 0 |]
let phaseWeight = [| 0; 1; 1; 2; 4; 0 |]

let private pawnTable =
    [| 0;  0;  0;  0;  0;  0;  0;  0
       50; 50; 50; 50; 50; 50; 50; 50
       10; 10; 20; 30; 30; 20; 10; 10
       5;  5; 10; 25; 25; 10;  5;  5
       0;  0;  0; 20; 20;  0;  0;  0
       5; -5;-10;  0;  0;-10; -5;  5
       5; 10; 10;-20;-20; 10; 10;  5
       0;  0;  0;  0;  0;  0;  0;  0 |]

let private knightTable =
    [| -50;-40;-30;-30;-30;-30;-40;-50
       -40;-20;  0;  0;  0;  0;-20;-40
       -30;  0; 10; 15; 15; 10;  0;-30
       -30;  5; 15; 20; 20; 15;  5;-30
       -30;  0; 15; 20; 20; 15;  0;-30
       -30;  5; 10; 15; 15; 10;  5;-30
       -40;-20;  0;  5;  5;  0;-20;-40
       -50;-40;-30;-30;-30;-30;-40;-50 |]

let private bishopTable =
    [| -20;-10;-10;-10;-10;-10;-10;-20
       -10;  0;  0;  0;  0;  0;  0;-10
       -10;  0;  5; 10; 10;  5;  0;-10
       -10;  5;  5; 10; 10;  5;  5;-10
       -10;  0; 10; 10; 10; 10;  0;-10
       -10; 10; 10; 10; 10; 10; 10;-10
       -10;  5;  0;  0;  0;  0;  5;-10
       -20;-10;-10;-10;-10;-10;-10;-20 |]

let private rookTable =
    [| 0;  0;  0;  0;  0;  0;  0;  0
       5; 10; 10; 10; 10; 10; 10;  5
       -5;  0;  0;  0;  0;  0;  0; -5
       -5;  0;  0;  0;  0;  0;  0; -5
       -5;  0;  0;  0;  0;  0;  0; -5
       -5;  0;  0;  0;  0;  0;  0; -5
       -5;  0;  0;  0;  0;  0;  0; -5
       0;  0;  0;  5;  5;  0;  0;  0 |]

let private queenTable =
    [| -20;-10;-10; -5; -5;-10;-10;-20
       -10;  0;  0;  0;  0;  0;  0;-10
       -10;  0;  5;  5;  5;  5;  0;-10
       -5;  0;  5;  5;  5;  5;  0; -5
       0;  0;  5;  5;  5;  5;  0; -5
       -10;  5;  5;  5;  5;  5;  0;-10
       -10;  0;  5;  0;  0;  0;  0;-10
       -20;-10;-10; -5; -5;-10;-10;-20 |]

let private kingTableMg =
    [| -30;-40;-40;-50;-50;-40;-40;-30
       -30;-40;-40;-50;-50;-40;-40;-30
       -30;-40;-40;-50;-50;-40;-40;-30
       -30;-40;-40;-50;-50;-40;-40;-30
       -20;-30;-30;-40;-40;-30;-30;-20
       -10;-20;-20;-20;-20;-20;-20;-10
       20; 20;  0;  0;  0;  0; 20; 20
       20; 30; 10;  0;  0; 10; 30; 20 |]

let private kingTableEg =
    [| -50;-40;-30;-20;-20;-30;-40;-50
       -30;-20;-10;  0;  0;-10;-20;-30
       -30;-10; 20; 30; 30; 20;-10;-30
       -30;-10; 30; 40; 40; 30;-10;-30
       -30;-10; 30; 40; 40; 30;-10;-30
       -30;-10; 20; 30; 30; 20;-10;-30
       -30;-30;  0;  0;  0;  0;-30;-30
       -50;-30;-30;-30;-30;-30;-30;-50 |]

let private mgTables = [| pawnTable; knightTable; bishopTable; rookTable; queenTable; kingTableMg |]
let private egTables = [| pawnTable; knightTable; bishopTable; rookTable; queenTable; kingTableEg |]

/// Signed White-POV contribution of piece index pc (0..11) on sq.
/// mg.[pc].[sq], eg.[pc].[sq]; phase.[pc]
let mg =
    Array.init 12 (fun pc ->
        Array.init 64 (fun sq ->
            let pt = pc % 6
            if pc < 6 then pieceValue.[pt] + mgTables.[pt].[sq ^^^ 56]
            else -(pieceValue.[pt] + mgTables.[pt].[sq])))

let eg =
    Array.init 12 (fun pc ->
        Array.init 64 (fun sq ->
            let pt = pc % 6
            if pc < 6 then pieceValue.[pt] + egTables.[pt].[sq ^^^ 56]
            else -(pieceValue.[pt] + egTables.[pt].[sq])))

let phase = Array.init 12 (fun pc -> phaseWeight.[pc % 6])
