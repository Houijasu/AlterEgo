# Solo feature A/B queue: each ALTEREGO_ENABLE flag plays a matched-net cage
# match against bin\dev compiled defaults, results appended to docs\EVIDENCE.md.
# Resumable: features already logged for the current commit are skipped.
#
#   .\scripts\sprt-queue.ps1                       # the 11 pending features, 192g
#   .\scripts\sprt-queue.ps1 -Features iir -Games 64
#   .\scripts\sprt-queue.ps1 -Features histlmr -Tune "histlmrdiv=8192"

param(
    [string[]]$Features = @("iir","histlmr","nmp2","sext2","countermove","seecapt","evalcache",
                            "probcut","corrhist","conthist","seequiet"),
    [int]$Games = 192,
    [int]$Ms = 500,
    [int]$Lanes = 12,
    [string]$Net = "data\net10a.bin",
    [string]$Opponent = "bin\dev\AlterEgo.exe",
    [string]$Tune = "",
    [string]$Exe = "bin\Release\net10.0\AlterEgo.exe"   # snapshot path: rebuilds must not touch a running queue
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$exe = Join-Path $root $Exe
$evidence = Join-Path $root "docs\EVIDENCE.md"
$commit = (git rev-parse --short HEAD).Trim()

if (-not (Test-Path $evidence)) {
    @"
# Feature evidence log

Solo A/B protocol: the in-process side carries ``ALTEREGO_ENABLE=<flag>`` (plus
optional ``ALTEREGO_TUNE``), the opponent is ``bin\dev`` compiled defaults, nets
matched on both sides. 192 games at 500ms resolve roughly +-20 Elo (1 sigma);
promote only clear positives, one feature at a time (M9 lesson).

| date | commit | flag | tune | games | W-D-L | score | elo | LLR | bench(flag-on) |
|---|---|---|---|---|---|---|---|---|---|
"@ | Set-Content $evidence
}

$qStart = Get-Date
$qDone = 0
$idx = 0
foreach ($f in $Features) {
    $idx++
    if (Select-String -Path $evidence -Pattern "\| $commit \| $f \|" -Quiet) {
        Write-Host "skip $f (already logged for $commit)"
        continue
    }
    Write-Host "=== [$idx/$($Features.Count)] $f ($Games games) ==="
    $env:ALTEREGO_ENABLE = $f
    if ($Tune) { $env:ALTEREGO_TUNE = $Tune }
    else { Remove-Item Env:\ALTEREGO_TUNE -ErrorAction SilentlyContinue }

    # flag-on bench signature, pinned (single-threaded runs crash unpinned on this box)
    New-Item -ItemType Directory -Force (Join-Path $root "log") | Out-Null
    $blog = Join-Path $root "log\bench_tmp.log"
    $p = Start-Process -FilePath $exe -ArgumentList "bench" -NoNewWindow -PassThru -RedirectStandardOutput $blog
    $p.ProcessorAffinity = 0x10
    $p.WaitForExit()
    $sig = if ((Get-Content $blog -Tail 1) -match 'bench: (\d+) nodes') { $Matches[1] } else { "?" }
    Remove-Item $blog

    # stream the gauntlet live — per-game lines carry [%, ETA] from the arena
    $res = $null
    & $exe cage $Games $Ms $Net $Opponent $Lanes 2>&1 | ForEach-Object {
        if ($_ -match '^cage result:') { $res = "$_" }
        Write-Host "  $_"
    }
    $date = Get-Date -Format yyyy-MM-dd
    if ($res -match 'cage result: \+(\d+) =(\d+) -(\d+) \((\d+) games\)\s+score ([\d.]+)%\s+elo ([+-]?\d+)\s+LLR ([+-]?[\d.]+)') {
        $row = "| $date | $commit | $f | $Tune | $($Matches[4]) | +$($Matches[1])=$($Matches[2])-$($Matches[3]) | $($Matches[5])% | $($Matches[6]) | $($Matches[7]) | $sig |"
    } else {
        $row = "| $date | $commit | $f | $Tune | ERROR | $res |  |  |  | $sig |"
    }
    Add-Content $evidence $row
    Write-Host $row
    $qDone++
    $remaining = $Features.Count - $idx
    if ($remaining -gt 0) {
        $perFeature = ((Get-Date) - $qStart).TotalMinutes / $qDone
        Write-Host ("queue: {0}/{1} done  [{2}%  ETA {3:N0}m]" -f $idx, $Features.Count,
            [int](100 * $idx / $Features.Count), ($perFeature * $remaining))
    }
}
Remove-Item Env:\ALTEREGO_ENABLE -ErrorAction SilentlyContinue
Remove-Item Env:\ALTEREGO_TUNE -ErrorAction SilentlyContinue
Write-Host "queue done -> $evidence"
