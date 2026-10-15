[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$InstallRoot)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd("\")
$Python = Join-Path $InstallRoot "venv\Scripts\python.exe"
$WorkDir = Join-Path ([System.IO.Path]::GetTempPath()) ("spacr-uninstall-" + [guid]::NewGuid().ToString("N"))
$Created = $false
try {
    $null = New-Item -ItemType Directory -Path $WorkDir
    $Created = $true
    $Acl = [System.Security.AccessControl.DirectorySecurity]::new()
    $Acl.SetAccessRuleProtection($true, $false)
    $Sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
    $Rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $Sid, [System.Security.AccessControl.FileSystemRights]::FullControl,
        [System.Security.AccessControl.InheritanceFlags]"ContainerInherit,ObjectInherit",
        [System.Security.AccessControl.PropagationFlags]::None,
        [System.Security.AccessControl.AccessControlType]::Allow)
    $Acl.AddAccessRule($Rule)
    Set-Acl -LiteralPath $WorkDir -AclObject $Acl
    & $Python -I -m spacr.install_cleanup prepare-windows-uninstall --install-root $InstallRoot --workdir $WorkDir
    if ($LASTEXITCODE -ne 0) { throw "The private uninstall interpreter could not be prepared." }
    $OutsidePython = Join-Path $WorkDir "runtime\python.exe"
    $Cleanup = Join-Path $WorkDir "install_cleanup.py"
    & $OutsidePython -I $Cleanup remove --only-root $InstallRoot --require-outside
    if ($LASTEXITCODE -ne 0) { throw "Application removal did not finish; retained paths are listed above." }
} finally {
    if ($Created) { Remove-Item -LiteralPath $WorkDir -Recurse -Force }
}
