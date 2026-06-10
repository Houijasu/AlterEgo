module AlterEgo.Eval

open AlterEgo.Types
open AlterEgo.Bitboards
open AlterEgo.Position

let pieceValue = Psqt.pieceValue

/// PST path: O(1), reads incremental accumulators maintained by Position
let inline evaluatePst (pos: Position) =
    let phase = min pos.Phase 24
    let score = (pos.Mg * phase + pos.Eg * (24 - phase)) / 24
    if pos.Stm = White then score else -score

/// Static evaluation in centipawns from the side-to-move's point of view.
/// Uses NNUE when a network is loaded (and the position doesn't force PST —
/// a per-Position flag so parallel A/B game lanes can't race).
let evaluate (pos: Position) =
    match Nnue.net with
    | Some _ when not pos.ForcePst -> Nnue.evaluateAcc pos.AccStack.[pos.Ply] pos.Stm
    | _ -> evaluatePst pos

/// Full recomputation — used only to verify the incremental accumulators.
let evaluateSlow (pos: Position) =
    let mutable mg = 0
    let mutable eg = 0
    let mutable phase = 0
    for pc in 0 .. 11 do
        let mutable bb = pos.ByPiece.[pc]
        while bb <> 0UL do
            let sq = lsb bb
            bb <- bb &&& (bb - 1UL)
            mg <- mg + Psqt.mg.[pc].[sq]
            eg <- eg + Psqt.eg.[pc].[sq]
            phase <- phase + Psqt.phase.[pc]
    let phase = min phase 24
    let score = (mg * phase + eg * (24 - phase)) / 24
    if pos.Stm = White then score else -score
