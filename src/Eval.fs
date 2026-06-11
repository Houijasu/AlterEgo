module AlterEgo.Eval

open AlterEgo.Position

/// Static evaluation in centipawns from the side-to-move's point of view.
/// NNUE-only: every entry point loads a network (embedded by default) before
/// searching; without one, Nnue.evaluateAcc returns 0 for every position.
let inline evaluate (pos: Position) =
    Nnue.evaluateAcc pos.AccStack.[pos.Ply] pos.Stm
