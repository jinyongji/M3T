# Four user controls: Body, Sequence, Variant, Visualize.
# ASCII-only for Windows PowerShell 5.1 compatibility.
[CmdletBinding()]
param(
    [ValidateSet('all','ape','bakingsoda','benchviseblue','broccolisoup','cam',
        'can','cat','clown','cube','driller','duck','eggbox','glue','iron',
        'koalacandy','lamp','phone','squirrel')]
    [string]$Body = 'all',

    [ValidateSet('first4','a_regular','b_dynamiclight','c_noisy','d_occlusion')]
    [string]$Sequence = 'first4',

    [ValidateSet('final','release-v1','visibility-only','combined-experimental')]
    [string]$Variant = 'final',

    [switch]$Visualize
)

$ErrorActionPreference = 'Stop'
$intermediateName = -join ([char[]]@(0x4E2D,0x95F4,0x7ED3,0x679C,0x6587,0x4EF6))
$outputRoot = Join-Path 'D:\workspace\_cpp\3DObjectTracking' $intermediateName
$buildRoot = Join-Path $outputRoot 'M3T-offline-build'
$exe = Join-Path $buildRoot 'examples\Release\evaluate_rbot_dataset.exe'
$dataset = 'D:\datasets\RBOT_dataset'
$diskModel = 'D:\workspace\_env\m3t_lightglue\disk_512.onnx'
$cachePath = Join-Path $buildRoot 'CMakeCache.txt'
if (-not (Test-Path -LiteralPath $cachePath -PathType Leaf)) { throw "Build cache missing: $cachePath" }
# Use the configuring CMake version: mixing VS CMake modules with vcpkg's
# newer executable can fail when a source change triggers regeneration.
$cmake = (Select-String -LiteralPath $cachePath -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$').Matches.Groups[1].Value
if (-not $cmake) { throw 'CMAKE_COMMAND was not found in the build cache.' }
foreach ($path in @($cmake,$cachePath,$diskModel,(Join-Path $dataset 'poses_first.txt'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file not found: $path" }
}
if (-not (Select-String -LiteralPath $cachePath -Pattern '^USE_ONNXRUNTIME:BOOL=ON$' -Quiet)) {
    throw 'The build must have USE_ONNXRUNTIME=ON for these DISK variants.'
}
$cachedSource = (Select-String -LiteralPath $cachePath -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$').Matches.Groups[1].Value
if (-not $cachedSource -or [IO.Path]::GetFullPath($cachedSource) -ne [IO.Path]::GetFullPath($PSScriptRoot)) {
    throw "Build cache belongs to a different source tree: $cachedSource"
}

$allBodies = @('ape','bakingsoda','benchviseblue','broccolisoup','cam','can',
    'cat','clown','cube','driller','duck','eggbox','glue','iron','koalacandy',
    'lamp','phone','squirrel')
$selectedBodies = if ($Body -eq 'all') { $allBodies } else { @($Body) }
$selectedSequences = if ($Sequence -eq 'first4') {
    @('a_regular','b_dynamiclight','c_noisy','d_occlusion')
} else { @($Sequence) }
foreach ($object in $selectedBodies) {
    foreach ($seq in $selectedSequences) {
        foreach ($frame in @('0000','1000')) {
            $path = Join-Path $dataset ("$object\frames\${seq}${frame}.png")
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Dataset frame missing: $path" }
        }
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$suffix = [Guid]::NewGuid().ToString('N').Substring(0,8)
$experiment = "N_final_${Variant}_${Body}_${Sequence}_${stamp}_${suffix}"
$resultRoot = Join-Path (Join-Path $outputRoot 'RBOT_paper_optimization') $experiment
$logs = Join-Path $resultRoot 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

function Invoke-LoggedNativeCommand {
    param([string]$Executable, [string[]]$Arguments, [string]$LogPath)
    $savedPreference = $ErrorActionPreference
    try {
        # PS 5.1: retain stderr as text instead of terminating on warnings.
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $LogPath -ErrorAction Stop
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($nativeExitCode -ne 0) { throw "Native command failed (exit $nativeExitCode): $Executable. See $LogPath" }
}

$arguments = @($Body,$Sequence,'--experiment',$experiment,'--diagnostics',
    '--sequential','--constant-velocity','--reliability-fusion',
    '--directional-temporal-prior','--model-disk','--disk-ransac-threshold','3')
if ($Variant -in @('final','combined-experimental')) { $arguments += '--noise-region-only' }
if ($Variant -in @('combined-experimental','visibility-only')) { $arguments += '--visibility-memory-local' }
if ($Visualize) { $arguments += 'visualize' }

Write-Host "Variant: $Variant; body: $Body; sequence: $Sequence" -ForegroundColor Cyan
Write-Host 'Sequential GPU evaluation; modeled occlusion and oracle resets are excluded.'
Write-Host "Output: $resultRoot"
Write-Host '[1/2] Building Release...'
Invoke-LoggedNativeCommand -Executable $cmake -Arguments @('--build',$buildRoot,
    '--config','Release','--target','evaluate_rbot_dataset','--parallel','8') `
    -LogPath (Join-Path $logs 'build.log')
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Executable missing: $exe" }

$gitCommit = (& git -C $PSScriptRoot rev-parse HEAD 2>$null | Out-String).Trim()
$gitStatus = @(& git -C $PSScriptRoot status --short 2>$null)
$patchPath = Join-Path $resultRoot 'source_changes.patch'
& git -C $PSScriptRoot diff HEAD --binary "--output=$patchPath"
if ($LASTEXITCODE -ne 0) { throw 'Could not save the source diff' }
foreach ($relativePath in @(& git -C $PSScriptRoot ls-files --others --exclude-standard)) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $relativePath))
    if (-not $sourcePath.StartsWith($PSScriptRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Untracked source path is outside the project: $relativePath"
    }
    $snapshotPath = Join-Path (Join-Path $resultRoot 'source_untracked') $relativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $snapshotPath) | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $snapshotPath
}
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $resultRoot 'runner_snapshot.ps1')
Copy-Item -LiteralPath $cachePath -Destination (Join-Path $resultRoot 'CMakeCache_snapshot.txt')
$manifest = [ordered]@{
    Status = 'running'; Variant = $Variant; Body = $Body; Sequence = $Sequence
    Arguments = $arguments; Executable = $exe
    ExecutableSha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
    ModelSha256 = (Get-FileHash -LiteralPath $diskModel -Algorithm SHA256).Hash
    GitCommit = $gitCommit; GitStatus = $gitStatus
    StartedAt = (Get-Date).ToString('o'); ExpectedConfigurations = @($selectedBodies).Count * @($selectedSequences).Count
    OMP_NUM_THREADS = 1; OPENCV_FOR_THREADS_NUM = 1
}
$manifestPath = Join-Path $resultRoot 'run_manifest.json'
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
$savedOmp = $env:OMP_NUM_THREADS
$savedDynamic = $env:OMP_DYNAMIC
$savedCvThreads = $env:OPENCV_FOR_THREADS_NUM
$timer = [Diagnostics.Stopwatch]::StartNew()
try {
    $env:OMP_NUM_THREADS = '1'
    $env:OMP_DYNAMIC = 'FALSE'
    $env:OPENCV_FOR_THREADS_NUM = '1'
    Write-Host '[2/2] Evaluating...'
    Write-Host ($exe + ' ' + ($arguments -join ' '))
    Invoke-LoggedNativeCommand -Executable $exe -Arguments $arguments -LogPath (Join-Path $logs 'evaluation.log')
    $timer.Stop()
    $culture = [Globalization.CultureInfo]::InvariantCulture
    $rows = @(foreach ($seq in $selectedSequences) {
        $successSum = 0.0
        $timeSum = 0.0
        foreach ($object in $selectedBodies) {
            $path = Join-Path $resultRoot "results\results_${seq}_${object}.txt"
            $values = (Get-Content -LiteralPath $path -Raw).Trim().Split(',')
            if ($values.Count -ne 6) { throw "Invalid result: $path" }
            $success = [double]::Parse($values[0],$culture)
            $cycle = [double]::Parse($values[1],$culture)
            if ([double]::IsNaN($success) -or $success -lt 0 -or $success -gt 1 -or
                [double]::IsNaN($cycle) -or [double]::IsInfinity($cycle) -or $cycle -le 0) {
                throw "Non-finite or out-of-range result: $path"
            }
            $diagnosticPath = Join-Path $resultRoot "diagnostics\diagnostics_${seq}_${object}.csv"
            $frames = @(Import-Csv -LiteralPath $diagnosticPath)
            if ($frames.Count -ne 1000) { throw "Incomplete diagnostics: $diagnosticPath" }
            $frameSuccess = 0
            for ($frameIndex = 0; $frameIndex -lt 1000; ++$frameIndex) {
                if ([int]$frames[$frameIndex].frame -ne $frameIndex -or
                    $frames[$frameIndex].tracking_success -notin @('0','1')) {
                    throw "Invalid diagnostic frame: $diagnosticPath row $frameIndex"
                }
                $frameSuccess += [int]$frames[$frameIndex].tracking_success
            }
            if ([Math]::Abs($success - $frameSuccess/1000.0) -gt 0.00001) {
                throw "Summary/diagnostic success mismatch: $path"
            }
            $successSum += $success
            $timeSum += $cycle
        }
        $count = @($selectedBodies).Count
        [pscustomobject]@{
            Sequence = $seq; Objects = $count
            SuccessPercent = [Math]::Round(100*$successSum/$count,4)
            TrackingCycleMs = [Math]::Round($timeSum/$count/1000,4)
            TrackingCycleFPS = [Math]::Round(1000000*$count/$timeSum,2)
        }
    })
    $rows | Format-Table -AutoSize
    $rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resultRoot 'summary.json') -Encoding UTF8
    $manifest.Status = 'complete'
    $manifest.Add('EvaluationWallSeconds',$timer.Elapsed.TotalSeconds)
    $manifest.Add('WholeJobFramesPerSecond',1000*$manifest.ExpectedConfigurations/$timer.Elapsed.TotalSeconds)
} catch {
    $manifest.Status = 'failed'
    $manifest.Add('Error',$_.Exception.Message)
    throw
} finally {
    $env:OMP_NUM_THREADS = $savedOmp
    $env:OMP_DYNAMIC = $savedDynamic
    $env:OPENCV_FOR_THREADS_NUM = $savedCvThreads
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}
Write-Host "Complete. Results, diagnostics, logs and summary: $resultRoot" -ForegroundColor Green
