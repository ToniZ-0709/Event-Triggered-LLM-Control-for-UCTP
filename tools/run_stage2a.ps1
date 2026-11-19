[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string] $Batch,
    [switch] $DryRun,
    [switch] $ConfirmFreeTier,
    [ValidateRange(1, 15)]
    [int] $Workers = 1,
    [string] $Python
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Python) {
    $candidates = @(
        (Join-Path $repo '.venv/Scripts/python.exe')
        (Join-Path $env:USERPROFILE 'venvs/itc2019-nsga3/Scripts/python.exe')
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if (-not $Python) {
        $Python = (Get-Command python -CommandType Application).Source
    }
}

$stageArgs = @('--batch', $Batch, '--workers', $Workers)
if ($DryRun) { $stageArgs += '--dry-run' }
if ($ConfirmFreeTier) { $stageArgs += '--confirm-free-tier' }

& $Python -B -u (Join-Path $PSScriptRoot 'run_stage2a.py') @stageArgs
exit $LASTEXITCODE
