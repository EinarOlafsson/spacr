!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "Sections.nsh"
!include "nsDialogs.nsh"

!ifndef VERSION
  !define VERSION "0.0.0"
!endif

!define SPACR_ICON "..\..\spacr\resources\icons\app_icon.ico"
!define MUI_ICON "${SPACR_ICON}"
!define MUI_UNICON "${SPACR_ICON}"

Name "spaCR"
OutFile "..\..\dist\online\SpaCR-${VERSION}-Windows-Online-Setup.exe"
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
Var ConsentSignIn
Var ConsentShareValue
Var ConsentIssuesValue
Var ConsentSignInValue
Var ConsentCollectedValue

Section /o "$(SPACR_NSIS_GPU)" SecGpu
SectionEnd

Function .onInit
  !insertmacro MUI_LANGDLL_DISPLAY
  ; Match the bootstrap's default: select acceleration only when a working
  ; NVIDIA driver identifies a card. The user can still untick the component.
  nsExec::ExecToStack 'cmd.exe /D /C nvidia-smi -L'
  Pop $0
  Pop $1
  ${If} $0 == 0
    SectionSetFlags ${SecGpu} ${SF_SELECTED}
  ${EndIf}
  StrCpy $ConsentShareValue 0
  StrCpy $ConsentIssuesValue 0
  StrCpy $ConsentSignInValue 0
  StrCpy $ConsentCollectedValue 0
FunctionEnd

Function ConsentPage
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 58u "Privacy and optional account setup$\r$\n$\r$\nCrash reports go to the PUBLIC spaCR GitHub repository. They are world-readable, indexed, and cannot be reliably unpublished. Every report is redacted, shown in an editable preview, and sent only when you press Send for that report. These choices are optional and revocable in Preferences."
  Pop $0
  ${NSD_CreateCheckbox} 0 66u 100% 12u "Include redacted diagnostic logs in report previews (off by default)"
  Pop $ConsentShare
  ${NSD_CreateCheckbox} 0 84u 100% 12u "Enable the public GitHub issue-report action (off by default)"
  Pop $ConsentIssues
  ${NSD_CreateCheckbox} 0 102u 100% 22u "Set up GitHub, Claude, GPT/Codex, and Gemini on first launch; official CLIs own credentials (off by default)"
  Pop $ConsentSignIn
  nsDialogs::Show
FunctionEnd

Function ConsentPageLeave
  ${NSD_GetState} $ConsentShare $ConsentShareValue
  ${NSD_GetState} $ConsentIssues $ConsentIssuesValue
  ${NSD_GetState} $ConsentSignIn $ConsentSignInValue
  StrCpy $ConsentCollectedValue 1
FunctionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecGpu} "GPU acceleration: measured 13x faster Cellpose segmentation and 20x faster ResNet classification than CPU on an RTX 3090; hardware varies."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "$(SPACR_NSIS_APPLICATION)" SecSpaCR
  SectionIn RO
  SetOutPath "$INSTDIR"
  File /oname=spacr.ico "${SPACR_ICON}"
  SetOutPath "$TEMP\spaCR-online-installer"
  File "install_spacr_windows.ps1"
  File /r "generated"

  StrCpy $1 "cpu"
  SectionGetFlags ${SecGpu} $2
  IntOp $2 $2 & ${SF_SELECTED}
  ${If} $2 != 0
    StrCpy $1 "auto"
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
  FileWrite $4 "starting backend=$1 language=$3 consent=$ConsentCollectedValue$\r$\n"
  FileClose $4
  DetailPrint "$(SPACR_NSIS_DOWNLOADING)"
  nsExec::ExecToLog 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$TEMP\spaCR-online-installer\install_spacr_windows.ps1" -InstallRoot "$INSTDIR" -Version "${VERSION}" -TorchBackend "$1" -Language "$3" -ConsentCollected $ConsentCollectedValue -ShareDiagnostics $ConsentShareValue -ReportIssues $ConsentIssuesValue -SignInNow $ConsentSignInValue'
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
  CreateShortcut "$SMPROGRAMS\spaCR\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"' "$INSTDIR\spacr.ico" 0
  CreateShortcut "$SMPROGRAMS\spaCR\Uninstall spaCR.lnk" "$INSTDIR\Uninstall.exe"
  CreateShortcut "$DESKTOP\spaCR.lnk" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"' "$INSTDIR\spacr.ico" 0

  Delete "$TEMP\spaCR-online-installer\install_spacr_windows.ps1"
  Delete "$TEMP\spaCR-online-installer\generated\installer_messages.ps1"
  RMDir "$TEMP\spaCR-online-installer\generated"
  RMDir "$TEMP\spaCR-online-installer"
SectionEnd

Function LaunchSpaCR
  ExecShell "open" "$INSTDIR\venv\Scripts\pythonw.exe" '"$INSTDIR\launch_spacr.pyw"'
FunctionEnd

; NSIS identifies the uninstaller by this exact sentinel name. Localising it
; turns it into a normal install section, which then deletes the fresh install.
Section "Uninstall"
  Delete "$DESKTOP\spaCR.lnk"
  RMDir /r "$SMPROGRAMS\spaCR"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\spaCR"
  DeleteRegKey HKCU "Software\spaCR"
  RMDir /r "$INSTDIR"
SectionEnd
