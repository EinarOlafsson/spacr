"""Terms-of-use acceptance and version records for a spaCR profile.

spaCR presents its terms during setup and records the accepted
:data:`TERMS_VERSION` with a UTC timestamp. A profile must accept a revised
version explicitly; acceptance of an earlier version is not carried forward.
The record is stored with the other setup values under the ``onboarding/``
prefix.
"""
from __future__ import annotations

from typing import Dict, Tuple

#: The version of the terms below.
#:
#: BUMP IT WHENEVER THE WORDS CHANGE. Nothing else makes a rewrite a new
#: agreement: the recorded version is compared against this one, so terms
#: edited without bumping it are terms nobody was asked about.
#:
#: 2.0 is the first version that states how the developers may use what a
#: profile sends them, so a profile that accepted 1.0 accepted a document
#: that did not say it and is asked again.
TERMS_VERSION = "3.0"

#: What the licence is called, and where the whole of it can be read.
LICENSE_NAME = "PolyForm Noncommercial License 1.0.0"
LICENSE_URL = "https://polyformproject.org/licenses/noncommercial/1.0.0"

#: The notice the licence itself requires be carried with the software.
REQUIRED_NOTICE = "Copyright 2025-2026 Einar Birnir Olafsson."

#: The agreement, in the words it is accepted in.
#:
#: WRITTEN AS AN END USER LICENCE AGREEMENT, because that is the form a
#: reader already knows how to read. The earlier draft explained itself in
#: an essayist's voice -- "this section is the one worth reading twice" --
#: which reads as someone talking ABOUT terms rather than as terms, and
#: leaves a reader unsure whether they have agreed to anything. The shape
#: here is the one Apple, Microsoft and Google all use: an acceptance
#: paragraph, defined terms, a licence grant, restrictions, ownership,
#: data, warranty and liability in the capitals those two sections are
#: always set in, then term, changes and general provisions.
#:
#: Section 2 summarises the PolyForm Noncommercial License 1.0.0 and says
#: so: the licence itself governs, and its own section names -- Acceptance,
#: Copyright License, Noncommercial Purposes, Violations, No Liability --
#: are what Sections 2, 3 and 8 are written against.
#:
#: LONG ENOUGH TO BE TERMS. A four-sentence summary is a thing a reader
#: takes in without noticing they agreed to anything, and the clauses that
#: matter here -- the ones about what the developers may do with what you
#: send them -- are exactly the clauses a summary drops. They are set out in
#: numbered sections, in the order a reader wants them: what the licence is,
#: what they may do, what there is no promise about, what leaves the
#: machine, what may be done with it, what is never made public, and how to
#: say no.
#:
#: THE LENGTH IS LOAD-BEARING, not padding: the acceptance on the setup
#: slide is disabled until the end of this text has been on screen, so the
#: document has to be long enough that reaching its end is an act.
TERMS: Tuple[str, ...] = (
    "END USER LICENCE AGREEMENT",
    "PLEASE READ THIS AGREEMENT CAREFULLY. By selecting \u201cI have read "
    "and agree to these terms\u201d, or by installing, copying or otherwise "
    "using spaCR, You agree to be bound by the terms of this Agreement. If "
    "You do not agree, do not install or use spaCR.",

    "1. DEFINITIONS",
    "1.1 \u201cAgreement\u201d means this End User Licence Agreement.",
    "1.2 \u201cLicensor\u201d means the copyright holder identified in the "
    "notice at the end of this Agreement.",
    "1.3 \u201cSoftware\u201d means the spaCR application, its source code, "
    "its documentation, and any updates the Licensor makes available.",
    "1.4 \u201cYou\u201d means the individual or entity accepting this "
    "Agreement.",
    "1.5 \u201cYour Content\u201d means the images, measurements, "
    "annotations, models, figures and other data You supply to the Software "
    "or that the Software produces from them on Your equipment.",
    "1.6 \u201cDiagnostic Data\u201d means logs, error reports, stack "
    "traces, configuration values, software and hardware version "
    "information, and timing measurements describing how the Software ran.",
    "1.7 \u201cShared Resource\u201d means any model hub, public dataset, "
    "community repository or similar service to which You choose to upload "
    "material.",

    "2. LICENCE GRANT",
    "2.1 The Software is licensed, not sold. Subject to Your compliance "
    "with this Agreement, the Licensor grants You a worldwide, "
    "royalty-free, non-exclusive licence to use, copy, modify and "
    "distribute the Software for any Noncommercial Purpose.",
    "2.2 This licence is granted under the terms of the PolyForm "
    "Noncommercial License 1.0.0, the full text of which is linked below "
    "and which governs in the event of any conflict with this summary.",
    "2.3 \u201cNoncommercial Purpose\u201d means any purpose other than "
    "commercial advantage or monetary compensation. Use by a university, a "
    "public research organisation, a charity, a hospital, a health or "
    "environmental body, or a government institution is a Noncommercial "
    "Purpose regardless of the source of its funding, including funding "
    "received from a commercial entity.",
    "2.4 Selling the Software, selling a service built upon it, or using it "
    "in the internal business operations of a commercial entity requires a "
    "separate licence from the Licensor.",

    "3. RESTRICTIONS",
    "3.1 You shall retain all copyright and licence notices in any copy of "
    "the Software You distribute, and shall supply this Agreement with it.",
    "3.2 You shall not use the Software for any Noncommercial Purpose "
    "exception not granted in Section 2, nor sublicense it on terms "
    "inconsistent with this Agreement.",
    "3.3 If You breach this Agreement, Your licence terminates. It is "
    "reinstated if You cure the breach within thirty-two (32) days of "
    "becoming aware of it.",

    "4. OWNERSHIP OF YOUR CONTENT",
    "4.1 As between You and the Licensor, You retain all right, title and "
    "interest in and to Your Content. This Agreement grants the Licensor no "
    "ownership of it and makes no claim of authorship over any result the "
    "Software produces from it.",
    "4.2 The Licensor retains all right, title and interest in and to the "
    "Software, subject to the licence granted in Section 2.",

    "5. DIAGNOSTIC DATA AND HOW IT MAY BE USED",
    "5.1 The Software does not collect or transmit Diagnostic Data "
    "automatically. It contains no telemetry, no background upload and no "
    "analytics. Data is transmitted only as the result of an action You "
    "take.",
    "5.2 Where You elect to send a bug report, and do not clear the setting "
    "\u201cInclude recent logs in a report\u201d, You grant the Licensor a "
    "perpetual, worldwide, "
    "royalty-free licence to use the Diagnostic Data contained in that "
    "report for the purposes of diagnosing faults, improving the Software, "
    "and any other purpose connected with its development.",
    "5.3 Where You upload image data or trained models to a Shared "
    "Resource, You grant the Licensor a perpetual, worldwide, "
    "royalty-free licence to use that material to develop the Software "
    "further and to produce community resources, including but not limited "
    "to object detection and segmentation models made available to other "
    "users.",
    "5.4 The licences in Sections 5.2 and 5.3 are non-exclusive. They do "
    "not transfer ownership, and they do not limit what You may do with the "
    "same material.",
    "5.5 Diagnostic Data is not published. A bug report filed through the "
    "Software opens a public issue that contains the fault description and "
    "software versions only; any log accompanying it is written to a file "
    "on Your equipment, and the issue records the path rather than the "
    "contents. Sending that file is a separate act You take.",
    "5.6 You are responsible for the content of anything You elect to send. "
    "The Software redacts credential-shaped values and file paths on a best "
    "efforts basis; it cannot identify material that is confidential for "
    "reasons particular to Your work.",

    "6. THIRD PARTY SERVICES",
    "6.1 The Software can be configured to send a question to a third party "
    "coding assistant. Such requests are made through that vendor's own "
    "command line tool under Your own account and are governed by that "
    "vendor's terms. The Licensor does not receive Your credentials and is "
    "not a party to those communications.",
    "6.2 The Software queries public package indexes to determine whether a "
    "newer release exists. Such queries transmit no information about You "
    "or Your Content.",

    "7. DISCLAIMER OF WARRANTIES",
    "7.1 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THE SOFTWARE IS "
    "PROVIDED \u201cAS IS\u201d AND \u201cAS AVAILABLE\u201d, WITH ALL "
    "FAULTS AND WITHOUT WARRANTY OR CONDITION OF ANY KIND, WHETHER EXPRESS, "
    "IMPLIED OR STATUTORY, INCLUDING ANY IMPLIED WARRANTY OF "
    "MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, ACCURACY OR "
    "NON-INFRINGEMENT.",
    "7.2 The Software is research software. It is written to be correct and "
    "it is tested, and it remains capable of producing results that are "
    "wrong for Your data in ways not previously encountered. You are "
    "responsible for validating any result on which You rely, and no result "
    "produced by the Software constitutes scientific, medical, diagnostic "
    "or clinical advice.",
    "7.3 The Software writes to the directories You direct it to. You are "
    "responsible for maintaining backups of Your Content.",

    "8. LIMITATION OF LIABILITY",
    "8.1 TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT "
    "SHALL THE LICENSOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, "
    "SPECIAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES, INCLUDING WITHOUT "
    "LIMITATION LOSS OF DATA, LOSS OF PROFITS, OR COST OF SUBSTITUTE "
    "SOFTWARE, ARISING OUT OF OR RELATING TO THIS AGREEMENT OR THE USE OF "
    "OR INABILITY TO USE THE SOFTWARE, UNDER ANY THEORY OF LIABILITY, EVEN "
    "IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.",
    "8.2 Nothing in this Agreement excludes or limits liability that cannot "
    "be excluded or limited under applicable law.",

    "9. TERM AND TERMINATION",
    "9.1 This Agreement takes effect when You accept it and continues until "
    "terminated.",
    "9.2 You may terminate it at any time by ceasing all use of the "
    "Software and destroying Your copies of it.",
    "9.3 Sections 4, 5, 7, 8 and 10 survive termination.",

    "10. CHANGES TO THIS AGREEMENT",
    "10.1 The Licensor may revise this Agreement. A revised Agreement "
    "carries a new version number and is presented to You for acceptance "
    "before the Software next runs.",
    "10.2 Your acceptance applies to the version You were shown. Continuing "
    "to use an earlier version of the Software does not bind You to a "
    "later Agreement.",

    "11. GENERAL",
    "11.1 If any provision of this Agreement is held unenforceable, the "
    "remaining provisions continue in full force.",
    "11.2 A failure to enforce any provision is not a waiver of it.",
    "11.3 This Agreement, together with the licence referenced in Section "
    "2.2, is the entire agreement between You and the Licensor concerning "
    "the Software.",
)

#: The line shown on the agreement control itself.
AGREE_LABEL = "I have read and agree to these terms"

#: What the screen says while the end of the terms has not been reached.
#:
#: THE GATE HAS TO EXPLAIN ITSELF. A control that is greyed with nothing
#: beside it is a control the reader has to guess at, and "scroll further"
#: is not a guess anyone makes about a switch. It is shown from the moment
#: the slide opens, not only after a press, because the reader meets the
#: greyed switch before they meet the button.
SCROLL_HINT = (
    "Scroll to the end of the terms to enable the acceptance checkbox.")

#: What the screen says when the user asks to move on without agreeing.
#:
#: NOT A DEAD BUTTON. A Next that is greyed with nothing beside it tells the
#: reader neither what is missing nor where to look for it, so the button
#: stays live and answers the press with the reason.
WHY_NOT_YET = (
    "Accept the terms of use to complete setup. If you close this window "
    "without accepting, spaCR will present the terms again at the next "
    "startup.")

#: Where the accepted version is remembered, and when it was accepted.
_KEY_VERSION = "onboarding/terms_agreed_version"
_KEY_WHEN = "onboarding/terms_agreed_at"


def _settings():
    from .preferences import _settings as store

    return store()


def terms_text() -> str:
    """Return the terms with a blank line between clauses."""
    return "\n\n".join(TERMS)


def agreed_version() -> str:
    """The terms version this profile accepted, or ``""`` if none."""
    try:
        return str(_settings().value(_KEY_VERSION, "") or "")
    except Exception:                                        # noqa: BLE001
        # A PROFILE THAT CANNOT BE READ HAS NOT AGREED. Answering "yes" when
        # the store is unreachable would turn a broken settings file into a
        # silent acceptance.
        return ""


def agreed_at() -> str:
    """When this profile accepted, in UTC ISO-8601, or ``""`` if never."""
    try:
        return str(_settings().value(_KEY_WHEN, "") or "")
    except Exception:                                        # noqa: BLE001
        return ""


def agreement_record() -> Dict[str, str]:
    """Return the recorded and current agreement metadata.

    :returns: ``{'version', 'accepted_at', 'current_version', 'license'}``.
        ``version`` is empty for a profile that has never accepted, which is
        a different answer from having accepted an older version and the
        reason both are reported rather than one boolean.
    """
    return {
        "version": agreed_version(),
        "accepted_at": agreed_at(),
        "current_version": TERMS_VERSION,
        "license": LICENSE_NAME,
    }


def record_agreement(version: str = "") -> str:
    """Record acceptance of a terms version and return the stored version.

    :param version: the terms version accepted. Defaults to
        :data:`TERMS_VERSION`, which is what the screen shows.
    """
    stamped = str(version or TERMS_VERSION)
    from datetime import datetime, timezone

    when = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    store = _settings()
    store.setValue(_KEY_VERSION, stamped)
    store.setValue(_KEY_WHEN, when)
    try:
        # WRITTEN THROUGH IMMEDIATELY. QSettings flushes lazily, and an
        # acceptance still sitting in a buffer when the process is killed is
        # an acceptance the user gave and would be asked for again.
        store.sync()
    except Exception:                                        # noqa: BLE001
        pass
    return stamped


def needs_agreement(version: str = "") -> bool:
    """Return whether the profile must accept the requested terms version.

    :param version: the version to check against. Defaults to
        :data:`TERMS_VERSION`.

    Returns ``True`` when no acceptance is recorded or when the recorded
    version differs from the requested version.
    """
    return agreed_version() != str(version or TERMS_VERSION)


#: The slide's own captions, translated. ``(source, (sv, de, es, zh_CN, pt,
#: hi, ko, is, fr))`` -- :data:`spacr.qt.i18n.LANGUAGES` order after English.
#:
#: THE CHROME IS TRANSLATED AND THE DOCUMENT IS NOT. The title, the sentence
#: introducing the terms, the acceptance and the reason for refusing are the
#: screen talking, and they are translated like everything else on it. The
#: TERMS THEMSELVES are not: a translated licence summary is not the licence,
#: and offering one as though it were would be the screen making a promise
#: the document does not. They are shown in the language the licence is
#: written in, with its name and its URL beside them.
TRANSLATIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Terms of use", (
        "Användningsvillkor", "Nutzungsbedingungen", "Condiciones de uso",
        "使用条款", "Termos de uso", "उपयोग की शर्तें", "이용 약관",
        "Notkunarskilmálar", "Conditions d’utilisation")),
    ("Review the terms of use and scroll to the end to enable acceptance. "
     "Use the license link to read the full PolyForm Noncommercial License "
     "1.0.0.", (
        "Läs igenom användningsvillkoren och rulla till slutet för att "
        "aktivera godkännandet. Använd licenslänken för att läsa hela "
        "PolyForm Noncommercial License 1.0.0.",
        "Lesen Sie die Nutzungsbedingungen und scrollen Sie bis zum Ende, "
        "um die Zustimmung zu aktivieren. Über den Lizenzlink können Sie die "
        "vollständige PolyForm Noncommercial License 1.0.0 lesen.",
        "Revise las condiciones de uso y desplácese hasta el final para "
        "habilitar la aceptación. Utilice el enlace de la licencia para leer "
        "la PolyForm Noncommercial License 1.0.0 completa.",
        "请查看使用条款并滚动到末尾以启用接受选项。使用许可证链接可阅读完整的 "
        "PolyForm Noncommercial License 1.0.0。",
        "Revise os termos de uso e role até o final para habilitar a "
        "aceitação. Use o link da licença para ler a PolyForm Noncommercial "
        "License 1.0.0 completa.",
        "उपयोग की शर्तों की समीक्षा करें और स्वीकृति सक्षम करने के लिए अंत तक स्क्रॉल करें। "
        "पूर्ण PolyForm Noncommercial License 1.0.0 पढ़ने के लिए लाइसेंस लिंक का उपयोग करें।",
        "이용 약관을 검토하고 끝까지 스크롤하여 동의 항목을 활성화하십시오. "
        "라이선스 링크에서 전체 PolyForm Noncommercial License 1.0.0을 확인할 수 있습니다.",
        "Farðu yfir notkunarskilmálana og skrunaðu til enda til að "
        "virkja samþykki. Notaðu leyfistengilinn til að lesa PolyForm "
        "Noncommercial License 1.0.0 í heild.",
        "Consultez les conditions d’utilisation et faites défiler jusqu’à la "
        "fin pour activer l’acceptation. Utilisez le lien de licence pour lire "
        "l’intégralité de la PolyForm Noncommercial License 1.0.0.")),
    (AGREE_LABEL, (
        "Jag har läst och godkänner dessa villkor",
        "Ich habe diese Bedingungen gelesen und stimme ihnen zu",
        "He leído y acepto estos términos",
        "我已阅读并同意这些条款",
        "Li e aceito estes termos",
        "मैंने इन शर्तों को पढ़ लिया है और मैं इन्हें स्वीकार करता हूँ",
        "이 약관을 읽었으며 이에 동의합니다",
        "Ég hef lesið og samþykki þessa skilmála",
        "J’ai lu et j’accepte ces conditions")),
    (SCROLL_HINT, (
        "Rulla till slutet av villkoren för att aktivera kryssrutan för "
        "godkännande.",
        "Scrollen Sie bis zum Ende der Nutzungsbedingungen, um das "
        "Kontrollkästchen zur Zustimmung zu aktivieren.",
        "Desplácese hasta el final de las condiciones para habilitar la "
        "casilla de aceptación.",
        "滚动到条款末尾以启用接受复选框。",
        "Role até o final dos termos para habilitar a caixa de seleção de "
        "aceitação.",
        "स्वीकृति चेकबॉक्स सक्षम करने के लिए शर्तों के अंत तक स्क्रॉल करें।",
        "약관 끝까지 스크롤하여 동의 확인란을 활성화하십시오.",
        "Skrunaðu til enda skilmálanna til að virkja samþykkisreitinn.",
        "Faites défiler jusqu’à la fin des conditions pour activer la case "
        "d’acceptation.")),
    (WHY_NOT_YET, (
        "Godkänn användningsvillkoren för att slutföra konfigurationen. Om "
        "du stänger fönstret utan att godkänna dem visar spaCR villkoren igen "
        "vid nästa start.",
        "Akzeptieren Sie die Nutzungsbedingungen, um die Einrichtung "
        "abzuschließen. Wenn Sie dieses Fenster ohne Zustimmung schließen, "
        "zeigt spaCR die Bedingungen beim nächsten Start erneut an.",
        "Acepte las condiciones de uso para completar la configuración. Si "
        "cierra esta ventana sin aceptarlas, spaCR volverá a mostrar las "
        "condiciones en el próximo inicio.",
        "接受使用条款以完成设置。如果未接受就关闭此窗口，spaCR 将在下次启动时再次显示这些条款。",
        "Aceite os termos de uso para concluir a configuração. Se fechar "
        "esta janela sem aceitá-los, o spaCR apresentará os termos novamente "
        "na próxima inicialização.",
        "सेटअप पूरा करने के लिए उपयोग की शर्तें स्वीकार करें। यदि आप बिना स्वीकार किए यह विंडो "
        "बंद करते हैं, तो spaCR अगली बार शुरू होने पर शर्तें फिर दिखाएगा।",
        "설정을 완료하려면 이용 약관에 동의하십시오. 동의하지 않고 이 창을 닫으면 spaCR가 "
        "다음 시작 시 약관을 다시 표시합니다.",
        "Samþykktu notkunarskilmálana til að ljúka uppsetningu. Ef þú "
        "lokar þessum glugga án þess að samþykkja birtir spaCR skilmálana "
        "aftur við næstu ræsingu.",
        "Acceptez les conditions d’utilisation pour terminer la configuration. "
        "Si vous fermez cette fenêtre sans les accepter, spaCR les présentera "
        "de nouveau au prochain démarrage.")),
)


def register_translations() -> int:
    """Register this screen's localized captions.

    :returns: number of rows added; repeated registration returns ``0``.
    """
    try:
        from .i18n import add_translation
    except Exception:                                        # noqa: BLE001
        # A SCREEN WITH NO CATALOG IS STILL A SCREEN. Every caption falls
        # back to the English it was written in.
        return 0
    added = 0
    for source, values in TRANSLATIONS:
        try:
            added += bool(add_translation(source, values))
        except ValueError:
            # A row that does not fit the catalog's shape is skipped rather
            # than allowed to stop the rest of the screen being catalogued.
            continue
    return added
