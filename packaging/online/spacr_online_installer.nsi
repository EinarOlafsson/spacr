!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "Sections.nsh"
!include "nsDialogs.nsh"
!include "FileFunc.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif

!define SPACR_ICON "..\..\spacr\resources\icons\app_icon.ico"
!define MUI_ICON "${SPACR_ICON}"
!define MUI_UNICON "${SPACR_ICON}"

Name "spaCR"
OutFile "..\..\dist\online\spaCR-${VERSION}-Windows-Online-Setup.exe"
Icon "${SPACR_ICON}"
InstallDir "$LOCALAPPDATA\SpaCR"
InstallDirRegKey HKCU "Software\spaCR" "InstallRoot"
RequestExecutionLevel user
Unicode True
SetCompressor /SOLID lzma

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "$(SPACR_NSIS_LAUNCH)"
!define MUI_FINISHPAGE_RUN_FUNCTION "LaunchSpaCR"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
Page custom ConsentPage ConsentPageLeave
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!include "generated\installer_messages.nsh"

Var ConsentShare
Var ConsentIssues
Var ConsentShareValue
Var ConsentIssuesValue
Var ConsentCollectedValue

Section /o "$(SPACR_NSIS_GPU)" SecGpu
SectionEnd

; Off by default: wgpu draws the True 3D view through Vulkan or Direct3D 12.
; Ticked, spaCR is installed without wgpu and draws 3D through OpenGL or the CPU.
Section /o "$(SPACR_WITHOUT_WGPU)" SecWithoutWgpu
SectionEnd

Function .onInit
  !insertmacro MUI_LANGDLL_DISPLAY
  ; Automatic acceleration is the default. uv selects CUDA when compatible
  ; NVIDIA hardware is present and falls back safely on other systems. The
  ; user can still untick this component to require the smaller CPU wheel.
  SectionSetFlags ${SecGpu} ${SF_SELECTED}
  SectionSetFlags ${SecWithoutWgpu} 0
  StrCpy $ConsentShareValue 1
  StrCpy $ConsentIssuesValue 1
  StrCpy $ConsentCollectedValue 0
FunctionEnd

Function ConsentPage
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 12u "Privacy choices"
  Pop $0
  ${NSD_CreateCheckbox} 0 20u 100% 12u "Include redacted diagnostic logs in report previews"
  Pop $ConsentShare
  ${NSD_Check} $ConsentShare
  ${NSD_CreateCheckbox} 0 38u 100% 12u "Enable the public GitHub issue-report action"
  Pop $ConsentIssues
  ${NSD_Check} $ConsentIssues
  nsDialogs::Show
FunctionEnd

Function ConsentPageLeave
  ${NSD_GetState} $ConsentShare $ConsentShareValue
  ${NSD_GetState} $ConsentIssues $ConsentIssuesValue
  StrCpy $ConsentCollectedValue 1
FunctionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecGpu} "Automatic acceleration (recommended): uses CUDA on compatible NVIDIA hardware and falls back safely elsewhere. Untick only for a smaller CPU-only installation."
  !insertmacro MUI_DESCRIPTION_TEXT ${SecWithoutWgpu} "$(SPACR_WITHOUT_WGPU_DESCRIPTION)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "$(SPACR_NSIS_APPLICATION)" SecSpaCR
  SectionIn RO
  InitPluginsDir
  SetOutPath "$INSTDIR"
  File /oname=spacr.ico "${SPACR_ICON}"
  SetOutPath "$PLUGINSDIR\spaCR-online-installer"
  File "install_spacr_windows.ps1"
  File /r "generated"
  ; The finder and remover the bootstrap runs before it installs: the same
  ; module the in-app update uses.
  File "..\..\spacr\install_cleanup.py"
  File "..\..\spacr\install_profile.py"
  File "windows_crt.py"
  File "windows_crt.json"

  StrCpy $1 "cpu"
  SectionGetFlags ${SecGpu} $2
  IntOp $2 $2 & ${SF_SELECTED}
  ${If} $2 != 0
    StrCpy $1 "auto"
  ${EndIf}

  StrCpy $5 ""
  SectionGetFlags ${SecWithoutWgpu} $2
  IntOp $2 $2 & ${SF_SELECTED}
  ${If} $2 != 0
    StrCpy $5 " -WithoutWgpu"
  ${EndIf}

  ; Keep the bootstrap phase in the language selected in MUI_LANGDLL_DISPLAY,
  ; even when it differs from the account's Windows display language.
  StrCpy $3 "en"
  ${If} $LANGUAGE == ${LANG_SWEDISH}
    StrCpy $3 "sv"
  ${ElseIf} $LANGUAGE == ${LANG_GERMAN}
    StrCpy $3 "de"
  ${ElseIf} $LANGUAGE == ${LANG_SPANISH}
    StrCpy $3 "es"
  ${ElseIf} $LANGUAGE == ${LANG_SIMPCHINESE}
    StrCpy $3 "zh_CN"
  ${ElseIf} $LANGUAGE == ${LANG_PORTUGUESE}
    StrCpy $3 "pt"
  ${ElseIf} $LANGUAGE == ${LANG_HINDI}
    StrCpy $3 "hi"
  ${ElseIf} $LANGUAGE == ${LANG_KOREAN}
    StrCpy $3 "ko"
  ${ElseIf} $LANGUAGE == ${LANG_ICELANDIC}
    StrCpy $3 "is"
  ${ElseIf} $LANGUAGE == ${LANG_FRENCH}
    StrCpy $3 "fr"
  ${EndIf}

  ; Keep a tiny wrapper-level trace even when PowerShell fails before its own
  ; transcript starts. This makes silent enterprise/CI installs diagnosable.
  FileOpen $4 "$INSTDIR\nsis-bootstrap-status.txt" w
  FileWrite $4 "starting backend=$1 language=$3 consent=$ConsentCollectedValue$5$\r$\n"
  FileClose $4
  DetailPrint "$(SPACR_NSIS_DOWNLOADING)"
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\spaCR-online-installer\install_spacr_windows.ps1" -InstallRoot "$INSTDIR" -Version "${VERSION}" -TorchBackend "$1" -Language "$3" -ConsentCollected $ConsentCollectedValue -ShareDiagnostics $ConsentShareValue -ReportIssues $ConsentIssuesValue$5'
  Pop $0
  FileOpen $4 "$INSTDIR\nsis-bootstrap-status.txt" a
  FileWrite $4 "exit=$0$\r$\n"
  FileClose $4
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "$(SPACR_NSIS_FAILED)"
    SetErrorLevel $0
    Abort
  ${EndIf}
  Delete "$INSTDIR\nsis-bootstrap-status.txt"

  WriteRegStr HKCU "Software\spaCR" "InstallRoot" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "DisplayName" "spaCR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "Publisher" "Olafsson Lab"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR" "UninstallString" '"$INSTDIR\Uninstall.exe"'

  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\spaCR"
  CreateShortcut "$SMPROGRAMS\spaCR\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '-I "$INSTDIR\launch_spacr.pyw"' "$INSTDIR\spacr.ico" 0
  CreateShortcut "$SMPROGRAMS\spaCR\Uninstall spaCR.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '-I "$INSTDIR\launch_spacr.pyw"' "$INSTDIR\spacr.ico" 0

  Delete "$PLUGINSDIR\spaCR-online-installer\install_spacr_windows.ps1"
  Delete "$PLUGINSDIR\spaCR-online-installer\install_cleanup.py"
  Delete "$PLUGINSDIR\spaCR-online-installer\install_profile.py"
  Delete "$PLUGINSDIR\spaCR-online-installer\windows_crt.py"
  Delete "$PLUGINSDIR\spaCR-online-installer\windows_crt.json"
  Delete "$PLUGINSDIR\spaCR-online-installer\generated\installer_messages.ps1"
  RMDir "$PLUGINSDIR\spaCR-online-installer\generated"
  RMDir "$PLUGINSDIR\spaCR-online-installer"
SectionEnd

Function LaunchSpaCR
  ExecShell "open" "$INSTDIR\venv\Scripts\pythonw.exe" '-I "$INSTDIR\launch_spacr.pyw"'
FunctionEnd

; NSIS identifies the uninstaller by this exact sentinel name. Localising it
; turns it into a normal install section, which then deletes the fresh install.
Section "Uninstall"
  ; Uninstall.exe /PURGE also deletes spaCR's caches, backends and user data.
  ; The purge lists them in a console and asks before it deletes anything.
  ${GetParameters} $R0
  ClearErrors
  ${GetOptions} $R0 "/PURGE" $R1
  ${IfNot} ${Errors}
    ExecWait '"$INSTDIR\venv\Scripts\python.exe" -m spacr.install_cleanup purge'
  ${EndIf}
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  File "uninstall_spacr_windows.ps1"
  nsExec::ExecToLog 'powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PLUGINSDIR\uninstall_spacr_windows.ps1" -InstallRoot "$INSTDIR"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP "$(SPACR_NSIS_UNINSTALL_FAILED)"
    SetErrorLevel $0
    Abort
  ${EndIf}
  ; The shared remover handles only matching shortcuts/registrations and keeps
  ; user/cache/model/unknown folders. Remove the root only when it is empty.
  RMDir "$INSTDIR"
SectionEnd
