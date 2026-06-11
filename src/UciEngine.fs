module AlterEgo.UciEngine

// UCI client for external engines (M7 cage matches): spawns the engine process,
// handles the handshake, and exchanges position/go/bestmove.

open System
open System.Diagnostics

type Engine(path: string, options: (string * string) list) =
    let psi = ProcessStartInfo(path)
    do
        psi.RedirectStandardInput <- true
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true
        // reference engines must run their compiled defaults: never inherit the
        // parent's feature flags (same-binary A/B tests would silently match)
        for k in [ "ALTEREGO_ENABLE"; "ALTEREGO_DISABLE"; "ALTEREGO_TUNE" ] do
            psi.EnvironmentVariables.Remove k |> ignore
    let proc = Process.Start psi

    // drain stderr asynchronously: a noisy engine must not fill the pipe and
    // deadlock our stdout reads (audit finding)
    do
        proc.ErrorDataReceived.Add(fun _ -> ())
        proc.BeginErrorReadLine()

    let send (s: string) =
        proc.StandardInput.WriteLine s
        proc.StandardInput.Flush()

    let rec waitFor (prefix: string) =
        let line = proc.StandardOutput.ReadLine()
        if line = null then failwithf "engine %s terminated unexpectedly" path
        elif line.StartsWith prefix then line
        else waitFor prefix

    do
        send "uci"
        waitFor "uciok" |> ignore
        for (n, v) in options do
            send (sprintf "setoption name %s value %s" n v)
        send "isready"
        waitFor "readyok" |> ignore

    member _.Name = path

    member _.NewGame() =
        send "ucinewgame"
        send "isready"
        waitFor "readyok" |> ignore

    /// moves: space-separated UCI moves from startpos ("" for the initial position)
    member _.BestMove(moves: string, moveMs: int64) =
        send (if moves = "" then "position startpos" else "position startpos moves " + moves)
        send (sprintf "go movetime %d" moveMs)
        let line = waitFor "bestmove"
        let parts = line.Split(' ')
        if parts.Length >= 2 then parts.[1] else "0000"

    member _.Quit() =
        try
            send "quit"
            if not (proc.WaitForExit 2000) then proc.Kill()
        with _ ->
            (try proc.Kill() with _ -> ())

    interface IDisposable with
        member this.Dispose() = this.Quit()
