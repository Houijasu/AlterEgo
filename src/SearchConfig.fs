module AlterEgo.SearchConfig

// Single owner of the experiment environment: feature flags and tuning knobs.
// ALTEREGO_ENABLE / ALTEREGO_DISABLE / ALTEREGO_TUNE are read ONCE at startup;
// Search and MACHINE bind their hot-path statics from here, so defaults live
// in exactly one place and same-binary A/B configs cannot drift apart.

let private parseSet (envVar: string) =
    match System.Environment.GetEnvironmentVariable envVar with
    | null -> Set.empty
    | s -> s.Split(',') |> Array.map (fun x -> x.Trim().ToLowerInvariant()) |> Set.ofArray

let private enabled = parseSet "ALTEREGO_ENABLE"
let private disabled = parseSet "ALTEREGO_DISABLE"

let private tune =
    match System.Environment.GetEnvironmentVariable "ALTEREGO_TUNE" with
    | null -> Map.empty
    | s ->
        s.Split(',')
        |> Array.choose (fun kv ->
            match kv.Split('=') with
            | [| k; v |] ->
                match System.Int32.TryParse(v.Trim()) with
                | true, n -> Some (k.Trim().ToLowerInvariant(), n)
                | _ -> None
            | _ -> None)
        |> Map.ofArray

/// Tuning knob: ALTEREGO_TUNE override or the given default.
let tuned (key: string) (dflt: int) = tune |> Map.tryFind key |> Option.defaultValue dflt

/// Experimental feature: off unless opted in via ALTEREGO_ENABLE.
let optIn (name: string) = enabled.Contains name

/// Promoted feature: on unless opted out via ALTEREGO_DISABLE (A/B runs).
let promoted (name: string) = not (disabled.Contains name)
