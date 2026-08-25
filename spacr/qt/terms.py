"""The terms spaCR asks a profile to accept, and the record that it did.

spaCR is distributed under the PolyForm Noncommercial License 1.0.0, whose
first clause is that the licence exists only once the terms are agreed to.
That makes acceptance a condition of use rather than a courtesy, which is
why it is asked on the way in and why the answer is kept.

WHAT IS KEPT IS THE VERSION, NOT A BARE YES. An acceptance of terms that
have since been rewritten is an acceptance of a different document, so
:data:`TERMS_VERSION` is stored beside the other setup answers and a newer
version asks again instead of inheriting the old answer.

The record lives in the same store the setup screen writes its answers to,
under the same ``onboarding/`` prefix, so "what did this profile agree to,
and when" is one lookup and not an inference.
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

#: The terms, in the words the user is asked to accept them in.
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
    "Scroll to the end of the terms. The acceptance below stays greyed "
    "until you have reached the bottom.")

#: What the screen says when the user asks to move on without agreeing.
#:
#: NOT A DEAD BUTTON. A Next that is greyed with nothing beside it tells the
#: reader neither what is missing nor where to look for it, so the button
#: stays live and answers the press with the reason.
WHY_NOT_YET = (
    "spaCR is licensed on the condition that you accept its terms, so setup "
    "cannot finish until the box above is ticked. Closing this window leaves "
    "them unaccepted and asks again next time.")

#: Where the accepted version is remembered, and when it was accepted.
_KEY_VERSION = "onboarding/terms_agreed_version"
_KEY_WHEN = "onboarding/terms_agreed_at"


def _settings():
    from .preferences import _settings as store

    return store()


def terms_text() -> str:
    """The terms as one block of text, one point per paragraph."""
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
    """What was accepted and when, as one mapping.

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
    """Record that these terms were accepted. Returns the version stored.

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
    """Whether this profile still has to accept the terms.

    :param version: the version to check against. Defaults to
        :data:`TERMS_VERSION`.

    True when nothing was ever recorded AND when what was recorded is a
    different version from the one now shipped -- rewritten terms are a new
    document, and inheriting the old answer would accept them on the user's
    behalf.
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
        "Användarvillkor", "Nutzungsbedingungen", "Términos de uso",
        "使用条款", "Termos de uso", "उपयोग की शर्तें", "이용 약관",
        "Notkunarskilmálar", "Conditions d'utilisation")),
    ("spaCR is licensed on the condition that you accept these terms. Read "
     "to the end -- the acceptance below stays greyed until you do -- and "
     "the whole licence is one click away.", (
        "spaCR licensieras på villkor att du godkänner dessa villkor. Läs "
        "till slutet -- godkännandet nedan är gråtonat tills du har gjort "
        "det -- och hela licensen är ett klick bort.",
        "spaCR wird unter der Bedingung lizenziert, dass Sie diese "
        "Bedingungen akzeptieren. Lesen Sie bis zum Ende -- die Zustimmung "
        "unten bleibt bis dahin ausgegraut -- und die vollständige Lizenz "
        "ist einen Klick entfernt.",
        "spaCR se licencia con la condición de que aceptes estos términos. "
        "Léelos hasta el final -- la aceptación de abajo permanece atenuada "
        "hasta entonces -- y la licencia completa está a un clic.",
        "spaCR 的授权以您接受这些条款为条件。请读到最后——在此之前下方的接受控件保持"
        "灰显——完整许可证只需点击一次即可查看。",
        "O spaCR é licenciado sob a condição de que você aceite estes "
        "termos. Leia até o fim -- a aceitação abaixo fica esmaecida até lá "
        "-- e a licença completa está a um clique.",
        "spaCR का लाइसेंस इस शर्त पर है कि आप इन शर्तों को स्वीकार करें। इन्हें अंत तक "
        "पढ़ें -- तब तक नीचे दी गई स्वीकृति धूसर रहती है -- और पूरा लाइसेंस एक क्लिक दूर है।",
        "spaCR는 이 약관에 동의하는 것을 조건으로 사용이 허가됩니다. 끝까지 읽어 "
        "주세요. 그때까지 아래 동의 항목은 흐리게 남아 있으며, 전체 라이선스는 "
        "클릭 한 번으로 볼 수 있습니다.",
        "spaCR er veitt með því skilyrði að þú samþykkir þessa skilmála. "
        "Lestu til enda -- samþykkið hér að neðan er grátt þar til þá -- og "
        "allt leyfið er einum smelli í burtu.",
        "spaCR est concédé sous licence à condition que vous acceptiez ces "
        "conditions. Lisez-les jusqu'au bout -- l'acceptation ci-dessous "
        "reste grisée jusque-là -- et la licence complète est à un clic.")),
    (AGREE_LABEL, (
        "Jag har läst och godkänner dessa villkor",
        "Ich habe diese Bedingungen gelesen und stimme ihnen zu",
        "He leído y acepto estos términos",
        "我已阅读并同意这些条款",
        "Li e aceito estes termos",
        "मैंने इन शर्तों को पढ़ लिया है और मैं इन्हें स्वीकार करता हूँ",
        "이 약관을 읽었으며 이에 동의합니다",
        "Ég hef lesið og samþykki þessa skilmála",
        "J'ai lu et j'accepte ces conditions")),
    (SCROLL_HINT, (
        "Bläddra till slutet av villkoren. Godkännandet nedan förblir "
        "gråtonat tills du har nått botten.",
        "Scrollen Sie bis zum Ende der Bedingungen. Die Zustimmung unten "
        "bleibt ausgegraut, bis Sie unten angekommen sind.",
        "Desplázate hasta el final de los términos. La aceptación de abajo "
        "permanece atenuada hasta que llegues al final.",
        "请滚动到条款末尾。在您到达底部之前，下方的接受控件保持灰显。",
        "Role até o fim dos termos. A aceitação abaixo fica esmaecida até "
        "você chegar ao final.",
        "शर्तों के अंत तक स्क्रॉल करें। जब तक आप नीचे तक नहीं पहुँचते, नीचे दी गई स्वीकृति "
        "धूसर रहती है।",
        "약관 끝까지 스크롤하세요. 맨 아래에 도달할 때까지 아래 동의 항목은 흐리게 "
        "표시됩니다.",
        "Skrunaðu að enda skilmálanna. Samþykkið hér að neðan helst grátt "
        "þar til þú kemst neðst.",
        "Faites défiler jusqu'à la fin des conditions. L'acceptation "
        "ci-dessous reste grisée tant que vous n'avez pas atteint le bas.")),
    (WHY_NOT_YET, (
        "spaCR licensieras på villkor att du godkänner villkoren, så "
        "installationen kan inte slutföras förrän rutan ovan är ikryssad. "
        "Om du stänger fönstret förblir de ogodkända och du får frågan igen "
        "nästa gång.",
        "spaCR wird unter der Bedingung lizenziert, dass Sie die Bedingungen "
        "akzeptieren; die Einrichtung kann daher erst abgeschlossen werden, "
        "wenn das Kästchen oben angekreuzt ist. Wenn Sie dieses Fenster "
        "schließen, bleiben sie unakzeptiert und werden beim nächsten Mal "
        "erneut abgefragt.",
        "spaCR se licencia con la condición de que aceptes sus términos, así "
        "que la configuración no puede terminar hasta que marques la casilla "
        "de arriba. Si cierras esta ventana quedarán sin aceptar y se "
        "preguntará de nuevo la próxima vez.",
        "spaCR 的授权以您接受其条款为条件，因此在勾选上面的复选框之前无法完成设置。"
        "关闭此窗口将使条款未被接受，下次仍会再次询问。",
        "O spaCR é licenciado sob a condição de que você aceite os seus "
        "termos, portanto a configuração não pode terminar até que a caixa "
        "acima seja marcada. Fechar esta janela deixa-os sem aceitação e "
        "pergunta de novo na próxima vez.",
        "spaCR का लाइसेंस इस शर्त पर है कि आप इसकी शर्तें स्वीकार करें, इसलिए ऊपर का "
        "बॉक्स चुने बिना सेटअप पूरा नहीं हो सकता। इस विंडो को बंद करने पर शर्तें अस्वीकृत "
        "रहती हैं और अगली बार फिर पूछा जाएगा।",
        "spaCR는 약관에 동의하는 것을 조건으로 사용이 허가되므로 위 상자를 선택하기 "
        "전에는 설정을 마칠 수 없습니다. 이 창을 닫으면 약관은 동의되지 않은 상태로 "
        "남으며 다음에 다시 묻습니다.",
        "spaCR er veitt með því skilyrði að þú samþykkir skilmálana, svo "
        "uppsetningin getur ekki lokið fyrr en hakað er í reitinn að ofan. "
        "Ef þú lokar þessum glugga verða þeir ósamþykktir og spurt verður "
        "aftur næst.",
        "spaCR est concédé sous licence à condition que vous acceptiez ses "
        "conditions ; la configuration ne peut donc pas se terminer tant que "
        "la case ci-dessus n'est pas cochée. Fermer cette fenêtre les laisse "
        "non acceptées et la question sera reposée la prochaine fois.")),
)


def register_translations() -> int:
    """Put this screen's captions in the translation catalogs.

    :returns: how many rows were added; 0 when they are already there, which
        is what a second call answers rather than an error.

    THROUGH THE REGISTRATION SEAM, not by hand-editing nine catalogs. A
    caption that lives with the module it is shown by cannot fall out of step
    with it, and :func:`spacr.qt.i18n.add_translation` is the same door the
    app registry uses.
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
