# Instala aforo-vision en Windows sin Docker (5 pasos). Se puede repetir: no rompe lo ya hecho.
# Uso:  Set-ExecutionPolicy -Scope Process Bypass; .\scripts\windows\install.ps1 [-Full]
#   -Full  fuerza requirements.txt (con torch/ultralytics) aunque exista requirements-slim.txt

param([switch]$Full)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPython = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$EnvFile = Join-Path $RepoRoot '.env'
$Total = 5

function Step([int]$n, [string]$text) { Write-Host "`nPaso $n de $Total - $text" -ForegroundColor Cyan }
function Fail([string]$msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Confirm([string]$q) { (Read-Host "$q (s/N)") -match '^[sSyY]' }

# Devuelve el comando de Python 3.14.2+ o $null
function Find-Python {
    foreach ($candidate in @(@('py', '-3.14'), @('python'))) {
        try {
            $cmdArgs = @($candidate[1..($candidate.Length)] | Where-Object { $_ }) + @('-c', 'import sys;print(sys.version_info[0],sys.version_info[1],sys.version_info[2])')
            $out = & $candidate[0] @cmdArgs 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) {
                $v = $out.Trim().Split(' ') | ForEach-Object { [int]$_ }
                if ($v[0] -eq 3 -and $v[1] -eq 14 -and $v[2] -ge 2) { return $candidate }
            }
        } catch { }
    }
    return $null
}

# Valida un valor de .env sin mostrarlo: ASCII, sin espacios, sin '<'
function Test-EnvValue([string]$v) { return ($v -match '^[\x21-\x7E]+$') -and (-not $v.StartsWith('<')) }

function ConvertFrom-Secure([System.Security.SecureString]$s) {
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

# --- Paso 1: Windows y Python ---
Step 1 'Revisando Windows y Python 3.14'
if ($env:OS -ne 'Windows_NT') { Fail 'Este instalador es solo para Windows 10/11.' }
if (-not [Environment]::Is64BitOperatingSystem) { Fail 'Se necesita Windows de 64 bits (x64).' }
$py = Find-Python
if (-not $py) {
    Write-Host 'No encontre Python 3.14.2 o superior (3.14.1 no sirve).' -ForegroundColor Yellow
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    $idOk = $false
    if ($winget) { & winget show --id Python.Python.3.14 --exact *> $null; $idOk = ($LASTEXITCODE -eq 0) }
    if ($idOk -and (Confirm 'Instalar Python 3.14 con winget ahora?')) {
        & winget install -e --id Python.Python.3.14
        if ($LASTEXITCODE -ne 0) { Fail 'winget no pudo instalar Python. Instalalo desde https://www.python.org/downloads/ y vuelve a correr este script.' }
        Write-Host 'Python instalado. Cierra y abre PowerShell y vuelve a correr .\scripts\windows\install.ps1 (para que reconozca el nuevo Python).' -ForegroundColor Green
        exit 0
    }
    if (-not $idOk) { Write-Host 'winget no esta disponible o no encuentra Python.Python.3.14.' }
    Fail 'Instala Python 3.14 desde https://www.python.org/downloads/ (marca "Add python.exe to PATH") y vuelve a correr este script.'
}
Write-Host "Python encontrado: $($py -join ' ')" -ForegroundColor Green

# --- Paso 2: entorno virtual ---
Step 2 'Creando el entorno virtual (.venv)'
if (Test-Path -LiteralPath $VenvPython) {
    Write-Host '.venv ya existe, la reutilizo.'
} else {
    $cmdArgs = @($py[1..($py.Length)] | Where-Object { $_ }) + @('-m', 'venv', (Join-Path $RepoRoot '.venv'))
    & $py[0] @cmdArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) { Fail 'No se pudo crear .venv. Revisa el espacio en disco y que Python este completo.' }
    Write-Host '.venv creada.' -ForegroundColor Green
}

# --- Paso 3: dependencias ---
Step 3 'Instalando dependencias (puede tardar varios minutos y bajar cientos de MB)'
$slim = Join-Path $RepoRoot 'requirements-slim.txt'
$req = if ((Test-Path -LiteralPath $slim) -and -not $Full) { $slim } else { Join-Path $RepoRoot 'requirements.txt' }
Write-Host "Usando $(Split-Path -Leaf $req)"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r $req
if ($LASTEXITCODE -ne 0) {
    Fail 'Fallo la instalacion de dependencias. Revisa tu internet y que haya al menos 5 GB libres, y vuelve a correr este script (retoma donde quedo).'
}
Write-Host 'Dependencias instaladas.' -ForegroundColor Green

# --- Paso 4: .env ---
Step 4 'Configurando el archivo .env (URL y llaves)'
$write = $true
if ((Test-Path -LiteralPath $EnvFile) -and (Confirm '.env ya existe. Conservarlo?')) { $write = $false }
if ($write) {
    Write-Host 'Pide a un admin la URL del proyecto, la publishable key y la llave del salon. Nada de esto se muestra ni se guarda fuera de .env.'
    do {
        $url = (Read-Host 'URL de Supabase (https://xxxx.supabase.co)').Trim()
        $okUrl = $url.StartsWith('https://') -and (Test-EnvValue $url)
        if (-not $okUrl) { Write-Host 'La URL debe empezar con https://, sin espacios, tildes ni "<". Intenta de nuevo.' -ForegroundColor Yellow }
    } until ($okUrl)
    do {
        $pub = (Read-Host 'Publishable key (sb_publishable_...)').Trim()
        $okPub = Test-EnvValue $pub
        if (-not $okPub) { Write-Host 'La llave solo puede tener caracteres ASCII, sin espacios ni "<". Intenta de nuevo.' -ForegroundColor Yellow }
    } until ($okPub)
    do {
        $secure = Read-Host 'Llave del salon (no se muestra al escribir)' -AsSecureString
        $dev = (ConvertFrom-Secure $secure).Trim()
        $okDev = Test-EnvValue $dev
        if (-not $okDev) { Write-Host 'La llave solo puede tener caracteres ASCII, sin espacios ni "<". Intenta de nuevo.' -ForegroundColor Yellow }
    } until ($okDev)
    $content = "AFORO_BACKEND_URL=$url`r`nAFORO_SUPABASE_KEY=$pub`r`nAFORO_DEVICE_KEY=$dev`r`n"
    [IO.File]::WriteAllText($EnvFile, $content, (New-Object Text.UTF8Encoding($false)))
    $dev = $null; $pub = $null; $content = $null
    Write-Host '.env creado (solo vive en esta carpeta; no lo compartas ni lo subas a git).' -ForegroundColor Green
} else {
    Write-Host 'Conservo el .env existente.'
}

# --- Paso 5: diagnostico ---
Step 5 'Revisando que todo funcione (python -m tools.doctor)'
Push-Location $RepoRoot
try { & $VenvPython -m tools.doctor; $doctorExit = $LASTEXITCODE } finally { Pop-Location }
if ($doctorExit -ne 0) {
    Write-Host "`nEl diagnostico encontro problemas. Corrige lo marcado con X y repite: .\.venv\Scripts\python.exe -m tools.doctor" -ForegroundColor Yellow
} else {
    Write-Host "`nTodo en orden." -ForegroundColor Green
}

Write-Host "`nArranque manual:       .\scripts\windows\start.ps1"
Write-Host 'Diagnostico cuando falle algo:  .\.venv\Scripts\python.exe -m tools.doctor'
if (Confirm 'Registrar el arranque automatico al iniciar sesion?') {
    & (Join-Path $PSScriptRoot 'install_autostart.ps1')
} else {
    Write-Host 'Sin arranque automatico. Puedes activarlo despues con .\scripts\windows\install_autostart.ps1'
}
