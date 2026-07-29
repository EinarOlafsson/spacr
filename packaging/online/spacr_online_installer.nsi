!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif

Name "spaCR"
OutFile "..\..\dist\online\SpaCR-${VERSION}-Windows-Online-Setup.exe"
InstallDir "$LOCALAPPDATA\SpaCR"
InstallDirRegKey HKCU "Software\spaCR" "InstallRoot"
RequestExecutionLevel user
Unicode True
SetCompressor /SOLID lzma

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Launch spaCR"
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchSpaCR"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Section "spaCR" SecSpaCR
  SetOutPath "$TEMP\spaCR-online-installer"
  File "install_spacr_windows.ps1"

  DetailPrint "Downloading Python, Qt, PyTorch and spaCR. This can take several minutes."
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TEMP\spaCR-online-installer\install_spacr_windows.ps1" -InstallRoot "$INSTDIR" -Version "${VERSION}"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "spaCR installation failed with exit code $0. The existing installation, if any, was preserved."
    Abort
  ${EndIf}

  WriteRegStr HKCU "Software\spaCR" "InstallRoot" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "DisplayName" "spaCR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "Publisher" "Olafsson Lab"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "UninstallString" '"$INSTDIR\Uninstall.exe"'

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\spaCR"
  CreateShortcut "$SMPROGRAMS\spaCR\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"'
  CreateShortcut "$SMPROGRAMS\spaCR\Uninstall spaCR.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"'

  Delete "$TEMP\spaCR-online-installer\install_spacr_windows.ps1"
  RMDir "$TEMP\spaCR-online-installer"
SectionEnd

Function LaunchSpaCR
  ExecShell "open" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"'
FunctionEnd

Section "Uninstall"
  Delete "$DESKTOP\spaCR.lnk"
  RMDir /r "$SMPROGRAMS\spaCR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR"
  DeleteRegKey HKCU "Software\spaCR"
  RMDir /r "$INSTDIR"
SectionEnd
