# Builds the separate GPU environment for the expressive OmniVoice voice (D:\voice\.venv-tts) and downloads the model.
# Safe to re-run: finished downloads are reused. Wheels are fetched with tools\fetch.py (parallel, resumable) and
# their sha256 checked, because a plain pip download from download.pytorch.org stalled on this connection.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py   = Join-Path $root '.venv\Scripts\python.exe'          # main venv: runs fetch.py
$tpy  = Join-Path $root '.venv-tts\Scripts\python.exe'
$dl   = Join-Path $root '.cache\wheels'
New-Item -ItemType Directory -Force $dl | Out-Null

$wheels = @(
  @{ url = 'https://download-r2.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl';
     sha = '0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff'; file = 'torch-2.8.0+cu128-cp312-cp312-win_amd64.whl' },
  @{ url = 'https://download-r2.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl';
     sha = 'cce3a60cd9a97f7360c8f95504ac349311fb7d6b9b826135936764f4de5f782d'; file = 'torchaudio-2.8.0+cu128-cp312-cp312-win_amd64.whl' }
)
foreach ($w in $wheels) {
  $f = Join-Path $dl $w.file
  $ok = (Test-Path $f) -and -not (Test-Path "$f.blocks") -and ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -eq $w.sha)
  if (-not $ok) {
    "downloading $($w.file)"
    & $py tools\fetch.py $w.url $f 16
    if ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -ne $w.sha) { throw "checksum mismatch: $($w.file)" }
  }
  "verified $($w.file)"
}

if (-not (Test-Path $tpy)) { py -3.12 -m venv .venv-tts }
& $tpy -m pip install --quiet --upgrade pip
& $tpy -m pip install --quiet (Join-Path $dl $wheels[0].file) (Join-Path $dl $wheels[1].file)
& $tpy -m pip install --quiet omnivoice soundfile fastapi "uvicorn[standard]"
& $tpy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

$env:HF_HUB_DISABLE_XET = '1'                                # the xet transfer stalled at 0 bytes on this PC
& (Join-Path $root '.venv-tts\Scripts\hf.exe') download k2-fsa/OmniVoice --local-dir models\omnivoice
Get-ChildItem models\omnivoice -Recurse -File | Select-Object @{n='MB';e={[math]::Round($_.Length/1MB,1)}}, Name
'done'
