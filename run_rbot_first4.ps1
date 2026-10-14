# ASCII-only so this entry point also works in Windows PowerShell 5.1.
[CmdletBinding()]
param(
    [ValidateSet(
        'all', 'ape', 'bakingsoda', 'benchviseblue', 'broccolisoup',
        'cam', 'can', 'cat', 'clown', 'cube', 'driller', 'duck',
        'eggbox', 'glue', 'iron', 'koalacandy', 'lamp', 'phone',
        'squirrel'
    )]
    [string]$Body = 'all',

    [ValidateRange(1, 256)]
    [int]$Threads = [Math]::Min(8, [Environment]::ProcessorCount),

    [switch]$Visualize,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$runner = Join-Path $PSScriptRoot 'run_rbot_parallel.ps1'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Required runner not found: $runner"
}

# A fresh experiment keeps the 72 configurations and their overall mean
# together, without overwriting earlier results or including modeled occlusion.
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$suffix = [Guid]::NewGuid().ToString('N').Substring(0, 8)
$experiment = "first4_baseline_${Body}_${stamp}_${suffix}"
Write-Host 'Sequences: a_regular, b_dynamiclight, c_noisy, d_occlusion'
Write-Host 'Algorithm: baseline (same tracking options as run_rbot_parallel.ps1)'
Write-Host "Experiment: $experiment"

& $runner -Body $Body -Sequence first4 -Threads $Threads `
    -ExperimentName $experiment -Visualize:$Visualize -SkipBuild:$SkipBuild
