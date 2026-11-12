# Finishes the setup once the downloads are complete. Safe to run again: each step is skipped when done
# and reports what is still missing.
#   powershell -ExecutionPolicy Bypass -File tools\finish_setup.ps1
$ErrorActionPreference = 'Stop'
$root   = Split-Path -Parent $PSScriptRoot
$py     = Join-Path $root '.venv\Scripts\python.exe'
$ollama = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
$blobs  = Join-Path $env:USERPROFILE '.ollama\models\blobs'
$model  = 'gemma4:12b-it-qat'
$modelBlob = 'faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1'
$projBlob  = 'e70b0e5cd80323d5d588b4ed06780356b7b1ba03995a4b8164c6ae9db0ff5989'
Set-Location $root

function Step($name) { Write-Host "`n== $name" -ForegroundColor Cyan }

# 1. Ollama: join the downloaded pieces, check the publisher's signature, install silently
Step 'Ollama'
if (Test-Path $ollama) { 'already installed' }
else {
    $parts = Join-Path $root '.cache\ollama_parts'
    $setup = Join-Path $root '.cache\OllamaSetup.exe'
    $total = [int64]((Get-Content (Join-Path $parts 'total.txt')) -replace '[^\d]', '')
    $have  = (Get-ChildItem $parts -Filter 'part*' | Measure-Object Length -Sum).Sum
    if ($have -ne $total) { "installer not complete yet ($([int]($have/1MB)) of $([int]($total/1MB)) MB)" }
    else {
        $out = [IO.File]::Create($setup)
        Get-ChildItem $parts -Filter 'part*' | Sort-Object Name | ForEach-Object {
            $in = [IO.File]::OpenRead($_.FullName); $in.CopyTo($out); $in.Close() }
        $out.Close()
        $sig = Get-AuthenticodeSignature $setup
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Ollama') {
            throw "installer signature is not valid ($($sig.Status)) - not running it" }
        "signature valid: $($sig.SignerCertificate.Subject)"
        # no -Wait: the installer starts the Ollama tray app and would never return
        Start-Process $setup -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART'
        foreach ($i in 1..120) { if (Test-Path $ollama) { break }; Start-Sleep 2 }
        Start-Sleep 5
        "installed: $(Test-Path $ollama)"
    }
}

# 2. Reply model: verify the downloaded weights; fall back to a normal pull if anything is off
Step "Reply model $model"
if (Test-Path $ollama) {
    try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null }
    catch { Start-Process $ollama -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 5 }
    $ok = $true
    foreach ($h in $modelBlob, $projBlob) {
        $f = Join-Path $blobs "sha256-$h"
        if ((Test-Path "$f.blocks") -or -not (Test-Path $f)) { $ok = $false; "weights still downloading ($h)"; continue }
        if ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -ne $h) { $ok = $false; "checksum mismatch: $h" }
    }
    if ($ok) { 'weights verified' } else { 'pulling through Ollama instead (resumes what is missing)'; & $ollama pull $model }
    & $ollama list
} else { 'skipped: Ollama is not installed yet' }

# 3. Bangla model from the Kaggle notebook: convert for the GPU, then measure it against the other one
Step 'Bangla model (tugstugi whisper-medium)'
$src = Join-Path $root '.cache\src\tugstugi-medium'
$dst = Join-Path $root 'models\bn-tugstugi-medium'
if (Test-Path (Join-Path $dst 'model.bin')) { 'already converted' }
elseif ((Get-Item (Join-Path $src 'model.safetensors') -ErrorAction SilentlyContinue).Length -eq 3055544304) {
    & 'D:\arc\venv-convert\Scripts\python.exe' tools\convert_model.py $src models\bn-tugstugi-medium
} else { 'weights still downloading' }
if (Test-Path (Join-Path $dst 'model.bin')) {
    & $py tools\bench.py --lang bn --n 100 --show 3 tugstugi-medium=models/bn-tugstugi-medium
}

Step 'Done'
'Start the app with the "Voice Chat" shortcut on the desktop (or start.bat).'
