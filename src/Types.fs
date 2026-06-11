module AlterEgo.Types

// Colors
[<Literal>]
let White = 0
[<Literal>]
let Black = 1

// Piece types 0..5; piece index = color * 6 + type (0..11), -1 = empty
[<Literal>]
let Pawn = 0
[<Literal>]
let Knight = 1
[<Literal>]
let Bishop = 2
[<Literal>]
let Rook = 3
[<Literal>]
let Queen = 4
[<Literal>]
let King = 5

let inline pieceType (pc: int) = pc % 6
let inline pieceColor (pc: int) = pc / 6

// Squares: 0 = a1 .. 63 = h8 (rank * 8 + file)
let inline fileOf (sq: int) = sq &&& 7
let inline rankOf (sq: int) = sq >>> 3
let inline mkSquare (file: int) (rank: int) = rank * 8 + file

let squareName (sq: int) =
    string (char (int 'a' + fileOf sq)) + string (char (int '1' + rankOf sq))

let parseSquare (s: string) =
    if s.Length < 2 then -1
    else
        let f = int s.[0] - int 'a'
        let r = int s.[1] - int '1'
        if f < 0 || f > 7 || r < 0 || r > 7 then -1 else mkSquare f r

// Move encoding (uint16):
//   bits 0-5   from square
//   bits 6-11  to square
//   bits 12-13 promotion piece - 1 (0=N, 1=B, 2=R, 3=Q), valid only with FlagPromo
//   bits 14-15 flag
[<Literal>]
let FlagNormal = 0
[<Literal>]
let FlagPromo = 1
[<Literal>]
let FlagEnPassant = 2
[<Literal>]
let FlagCastle = 3

type Move = uint16

[<Literal>]
let NoMove : Move = 0us

let inline mkMove (from: int) (dest: int) : Move =
    uint16 (from ||| (dest <<< 6))

let inline mkMoveF (from: int) (dest: int) (flag: int) (promo: int) : Move =
    uint16 (from ||| (dest <<< 6) ||| (promo <<< 12) ||| (flag <<< 14))

let inline moveFrom (m: Move) = int m &&& 63
let inline moveTo (m: Move) = (int m >>> 6) &&& 63
let inline movePromo (m: Move) = ((int m >>> 12) &&& 3) + Knight  // piece type
let inline moveFlag (m: Move) = int m >>> 14

/// Append diagnostics to alterego-crash.log next to the exe (never throws)
let logCrash (context: string) (ex: exn) =
    try
        let log = System.IO.Path.Combine(System.AppContext.BaseDirectory, "alterego-crash.log")
        let msg =
            sprintf "[%s] %s%s%O%s"
                (System.DateTime.Now.ToString "s") context
                System.Environment.NewLine ex System.Environment.NewLine
        System.IO.File.AppendAllText(log, msg)
    with _ -> ()

/// UCI string for a move, e.g. "e2e4", "e7e8q"
let moveToUci (m: Move) =
    if m = NoMove then "0000"
    else
        let s = squareName (moveFrom m) + squareName (moveTo m)
        if moveFlag m = FlagPromo then
            s + string "nbrq".[movePromo m - Knight]
        else s
