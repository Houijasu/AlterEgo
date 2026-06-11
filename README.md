# AlterEgo

AlterEgo is a UCI chess engine written in F#. It combines a conventional
alpha-beta search core with an embedded NNUE evaluator, perft tooling, self-play
data generation, NNUE training, and match harnesses for engine-vs-engine
experiments.

The engine starts in UCI mode by default and identifies itself as
`AlterEgo 0.3`.

## Features

- UCI engine interface for chess GUIs and engine tournaments.
- Bitboard board representation with magic move generation.
- Perft and divide commands for move-generation verification.
- Iterative deepening alpha-beta search with transposition table, move ordering,
  pruning, quiescence search, and lazy SMP helper threads.
- Embedded `default.nnue` network with PST evaluation fallback.
- `MACHINE` timed-search layer for root verification over the Ego search core.
- Self-play data generation and in-repo NNUE trainer.
- In-process A/B match runner and external UCI gauntlet support.

## Requirements

- .NET SDK `10.0.300` or a compatible later feature band.
- A UCI-compatible chess GUI or tournament manager if you want to run AlterEgo
  as an engine.
- Stockfish or another external UCI engine only for `cage` gauntlets.

The SDK version is pinned in [global.json](global.json).

## Quick Start

Build the engine:

```powershell
dotnet restore
dotnet build -c Release
```

Run it in UCI mode:

```powershell
dotnet run -c Release -- uci
```

With no arguments, AlterEgo also enters UCI mode:

```powershell
dotnet run -c Release --
```

After a release build, the executable is written under
`bin/Release/net10.0/`. Add that executable to your chess GUI as a UCI engine.

## UCI Options

AlterEgo exposes these UCI options:

| Option | Type | Default | Range | Description |
|---|---:|---:|---:|---|
| `Hash` | spin | `64` | `1..8192` | Transposition table size in MB. |
| `Threads` | spin | `1` | `1..256` | Search threads, including helper threads. |
| `EvalFile` | string | `<embedded>` | n/a | Optional NNUE file path. Uses embedded `default.nnue` by default. |

Clocked UCI searches such as `go movetime`, `go wtime`, and `go btime` use the
`MACHINE` timed-search layer. Budgetless searches such as `go depth`,
`go nodes`, and `go infinite` run the Ego alpha-beta core directly.

## Commands

All commands below are run through `dotnet run -c Release -- ...`.

| Command | Purpose |
|---|---|
| `uci` | Start the UCI engine loop. This is also the default. |
| `perft <depth>` | Run perft from the standard starting position. |
| `perft <depth> "<fen>"` | Run perft from a custom FEN. |
| `divide <depth> "<fen>"` | Print per-move perft counts for a position. |
| `suite` | Run the built-in perft suite through depth 5. |
| `suite deep` | Run the built-in perft suite through depth 6 where available. |
| `bench [depth]` | Search benchmark over the built-in position suite. Default depth is 9. |
| `machine <ms> ["<fen>"]` | Run one timed `MACHINE` search and print `bestmove`. |
| `match <games> <ms> [lanes]` | Run `MACHINE` vs Ego self-play matches. |
| `matchnnue <games> <ms> <net> [lanes]` | Run NNUE Ego vs PST Ego matches. |
| `datagen <games> <nodes> <out> [net] [lanes]` | Generate binary self-play samples. |
| `train <data> <epochs> <out> [kingBuckets]` | Train and export an NNUE file. |
| `scrub <in> <out>` | Recover valid 100-byte records from interrupted datagen output. |
| `evalcheck <net>` | Compare NNUE and PST scores on reference positions. |
| `nnuetest <net>` | Verify incremental NNUE accumulators against full rebuilds. |
| `cage <games> <ms> <net> <engine> [elo] [lanes]` | Run a gauntlet against an external UCI engine. |

Examples:

```powershell
dotnet run -c Release -- suite
dotnet run -c Release -- bench 9
dotnet run -c Release -- perft 5
dotnet run -c Release -- machine 1000
dotnet run -c Release -- datagen 1000 10000 data\samples.bin default.nnue 8
dotnet run -c Release -- train data\samples.bin 12 data\net.nnue 4
dotnet run -c Release -- cage 64 500 default.nnue tools\stockfish.exe 2900 8
```

## Verification

There is no separate test project in this repository. The main verification
commands are:

```powershell
dotnet build -c Release
dotnet run -c Release -- suite
dotnet run -c Release -- bench 9
dotnet run -c Release -- nnuetest default.nnue
```

Use `suite deep` for a slower perft check.

## Training Data

Datagen writes fixed-size binary records. Each record is 100 bytes:

- `12 x uint64` bitboards.
- Side-to-move byte.
- White-perspective centipawn score as `int16`.
- White-perspective result as `sbyte` (`-1`, `0`, or `1`).

Generated data, trained networks, logs, and external engines are intentionally
ignored by Git. Keep them under `data/` and `tools/` unless you intentionally
want to track a small artifact.

## Search Feature Toggles

Search experiments can be controlled with environment variables:

| Variable | Example | Purpose |
|---|---|---|
| `ALTEREGO_ENABLE` | `probcut,corrhist` | Opt in to experimental features. |
| `ALTEREGO_DISABLE` | `lmp,improving` | Disable promoted features for A/B runs. |
| `ALTEREGO_TUNE` | `lmpbase=2,sbetamult=3` | Override tuning constants for sweeps. |

PowerShell example:

```powershell
$env:ALTEREGO_DISABLE = "qstt,qschecks"
dotnet run -c Release -- bench 9
Remove-Item Env:\ALTEREGO_DISABLE
```

## Project Layout

```text
.
|-- AlterEgo.fsproj       F# project file
|-- Program.fs            CLI dispatcher and entry point
|-- default.nnue          Embedded default evaluation network
|-- docs/
|   |-- MACHINE.md        Search architecture notes
|   `-- LADDER.md         Strength measurement notes
`-- src/
    |-- Arena.fs          Match and gauntlet harness
    |-- Datagen.fs        Self-play sample generation
    |-- Machine.fs        Timed MACHINE search layer
    |-- MoveGen.fs        Legal move generation
    |-- Nnue.fs           NNUE load, save, and inference
    |-- Search.fs         Alpha-beta search core
    |-- Train.fs          NNUE trainer
    |-- Uci.fs            UCI protocol loop
    `-- ...
```

## Documentation

- [MACHINE](docs/MACHINE.md) describes the intended search architecture and
  build order.
- [LADDER](docs/LADDER.md) records Stockfish ladder anchors and experiment
  notes.

## License

No license file is currently included in this repository.
