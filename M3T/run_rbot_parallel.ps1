[CmdletBinding()]
param(
    [ValidateSet(
        'all', 'ape', 'bakingsoda', 'benchviseblue', 'broccolisoup',
        'cam', 'can', 'cat', 'clown', 'cube', 'driller', 'duck',
        'eggbox', 'glue', 'iron', 'koalacandy', 'lamp', 'phone',
        'squirrel'
    )]
    [string]$Body = 'all',

    [ValidateSet(
        'all', 'a_regular', 'b_dynamiclight', 'c_noisy',
        'd_occlusion', 'd_occlusion_modeled'
    )]
    [string]$Sequence = 'all',

    [ValidateRange(1, 256)]
    [int]$Threads = [Math]::Min(8, [Environment]::ProcessorCount),

    [switch]$Visualize,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$datasetRoot = 'D:\datasets\RBOT_dataset'
$environmentRoot = 'D:\workspace\_env'
$outputRoot = 'D:\workspace\_cpp\3DObjectTracking\中间结果文件'
$buildDirectory = Join-Path $outputRoot 'M3T-offline-build'
$resultRoot = Join-Path $outputRoot 'RBOT'
$logDirectory = Join-Path $resultRoot 'logs'
$executable = Join-Path $buildDirectory 'examples\Release\evaluate_rbot_dataset.exe'
$cmakeExecutable = Join-Path $environmentRoot `
    'vcpkg\downloads\tools\cmake-4.4.2-windows\cmake-4.4.2-windows-x86_64\bin\cmake.exe'

function Assert-PathExists {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [Parameter(Mandatory)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Description 不存在：$Path"
    }
}

Assert-PathExists -Path $datasetRoot -Description 'RBOT 数据集目录'
Assert-PathExists -Path (Join-Path $datasetRoot 'poses_first.txt') `
    -Description 'RBOT 主物体位姿文件'
Assert-PathExists -Path (Join-Path $datasetRoot 'poses_second.txt') `
    -Description 'RBOT 遮挡物位姿文件'
Assert-PathExists -Path $cmakeExecutable -Description 'CMake 可执行文件'

Write-Host '=== M3T RBOT 并行评测 ===' -ForegroundColor Cyan
Write-Host "项目目录 : $projectRoot"
Write-Host "数据集   : $datasetRoot"
Write-Host "构建目录 : $buildDirectory"
Write-Host "评测范围 : body=$Body, sequence=$Sequence"

if (-not $SkipBuild) {
    Assert-PathExists -Path (Join-Path $buildDirectory 'CMakeCache.txt') `
        -Description '已配置的 CMake 构建目录'

    Write-Host ''
    Write-Host '[1/2] 正在编译 evaluate_rbot_dataset (Release)...' `
        -ForegroundColor Yellow
    & $cmakeExecutable --build $buildDirectory --config Release `
        --target evaluate_rbot_dataset --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "编译失败，CMake 退出码：$LASTEXITCODE"
    }
} else {
    Write-Host ''
    Write-Host '[1/2] 已跳过编译。' -ForegroundColor DarkYellow
}

Assert-PathExists -Path $executable -Description 'RBOT 评测程序'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logDirectory `
    "rbot_${Body}_${Sequence}_${timestamp}.log"
$arguments = @($Body, $Sequence)
if ($Visualize) {
    $arguments += 'visualize'
    Write-Warning '可视化模式会强制串行评测，Threads 参数在该模式下不生效。'
}

$previousThreadCount = $env:OMP_NUM_THREADS
$previousDynamicSetting = $env:OMP_DYNAMIC
$env:OMP_NUM_THREADS = [string]$Threads
$env:OMP_DYNAMIC = 'FALSE'

Write-Host ''
Write-Host '[2/2] 正在运行 RBOT 评测...' -ForegroundColor Yellow
Write-Host "OpenMP 线程数: $Threads"
Write-Host "日志文件     : $logPath"
Write-Host ''

try {
    & $executable @arguments 2>&1 | Tee-Object -FilePath $logPath
    $runExitCode = $LASTEXITCODE
} finally {
    $env:OMP_NUM_THREADS = $previousThreadCount
    $env:OMP_DYNAMIC = $previousDynamicSetting
}

if ($runExitCode -ne 0) {
    throw "RBOT 评测失败，程序退出码：$runExitCode"
}

Write-Host ''
Write-Host 'RBOT 评测完成。' -ForegroundColor Green
Write-Host "日志: $logPath"
Write-Host "结果: $(Join-Path $resultRoot 'results')"
