"""Terms-of-use acceptance and version records for a spaCR profile.

spaCR presents its terms during setup and records the accepted
:data:`TERMS_VERSION` with a UTC timestamp. A profile must accept a revised
version explicitly; acceptance of an earlier version is not carried forward.
The record is stored with the other setup values under the ``onboarding/``
prefix.

Version 4.3 rewrites Sections 5 and 6 for item 700: anonymised error reports
are on by default and filed by spaCR's relay with no GitHub account, spaCR AI
(free) answers through that relay, and a user's own Claude, GPT or Gemini is
chosen in Preferences, so a profile that accepted 4.2 is asked again.
"""
from __future__ import annotations

import logging

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
#:
#: 4.0 is the version that RELICENSES spaCR from PolyForm Noncommercial to
#: BSD 3-Clause. That is a widening -- everything a profile could do under
#: 3.0 it may still do -- but it is not a wording change: the licence a
#: profile accepted is a different licence, the noncommercial restriction
#: is gone, and the warranty and liability terms are now the licence's own
#: rather than this document's. A profile that accepted 3.0 is asked again.
#: 4.1 adds Section 11.4, the governing-language clause, when the
#: agreement began being presented in nine languages on 2026-09-02. A
#: translation is a convenience and the English governs; that is a term
#: of the agreement rather than a note about it, so it is IN the
#: agreement and a profile that accepted 4.0 is asked again.
#: 4.2 adds Section 5.6, AUTOMATIC ERROR REPORTS: automatic filing is the
#: default, and a user agrees to it in the agreement when the mode is
#: 'always'. Sections 5.1 and 5.5 said nothing was ever transmitted
#: automatically and that Diagnostic Data was not published, and neither is
#: true with 'always'. A profile that accepted 4.1 accepted neither, and is
#: asked again.
LOG = logging.getLogger(__name__)

from ..crashreport import _REPORTING_TERMS_VERSION

TERMS_VERSION = _REPORTING_TERMS_VERSION

LICENSE_NAME = "spaCR Noncommercial License 1.0"
LICENSE_URL = "https://github.com/EinarOlafsson/spacr/blob/nightly/LICENSE"
REQUIRED_NOTICE = ("Copyright (c) 2025-2026, Einar Birnir Olafsson "
                   "and the spaCR contributors.")

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
    "2.1 New original spaCR material first distributed with Agreement version "
    "4.4 is licensed under the spaCR Noncommercial License 1.0 supplied in "
    "LICENSE. That licence governs where this summary conflicts with it. "
    "This is source-available software, not OSI-approved open source.",
    "2.2 Noncommercial use, copying, modification and distribution of the "
    "covered material are permitted subject to that licence. Noncommercial "
    "research, education, teaching and personal experiments are permitted "
    "without a fee, account approval or material-transfer agreement.",
    "2.3 Commercial use requires separate written permission from an "
    "authorized licensor. Commercial use means use directed toward commercial "
    "advantage or monetary compensation, including paid services, commercial "
    "hosting or products and internal research or operations for commercial "
    "advantage. Organizational status alone creates no exception. Contact "
    "olafsson@med.umich.com. This Agreement grants no commercial permission "
    "and specifies no fee or commercial licence terms.",
    "2.4 Previously BSD-licensed spaCR material retains its BSD 3-Clause "
    "permissions, including commercial permissions, and its notices in "
    "LICENSE-BSD-3-Clause. This Agreement does not withdraw earlier grants "
    "or change the licences of prior releases or third-party material.",
    "2.5 Third-party libraries, contributions, datasets and model weights "
    "remain under their own licences. Their rights to inspect, modify, "
    "replace, relink, obtain source, redistribute or debug by reverse "
    "engineering, where granted, are not restricted by this Agreement. "
    "GPL-covered combined-work obligations are not overridden; this Agreement "
    "does not certify every dependency combination or binary distribution as "
    "licence-compatible.",

    "3. WHAT THE LICENCE ASKS IN RETURN",
    "3.1 Distribute the applicable licence and copyright notice with copies "
    "of covered material or modifications, and identify modifications. The "
    "noncommercial licence does not grant commercial permission for changes "
    "or new works; obtain separate written permission where required.",
    "3.2 Retain the BSD copyright notice, conditions and disclaimer for "
    "previously BSD-licensed material in source and binary distributions. "
    "Keep all applicable third-party licences and notices.",
    "3.3 A copyright holder's or contributor's name may not be used to "
    "endorse a derived product without prior written permission where the "
    "applicable licence prohibits it. No trademark rights are granted here.",
    "3.4 If You publish work that relied on the Software, a citation is "
    "asked for and is not a condition of the licence. See CITATION.cff in "
    "the Software for the form it should take.",

    "4. OWNERSHIP OF YOUR CONTENT",
    "4.1 As between You and the Licensor, You retain all right, title and "
    "interest in and to Your Content. This Agreement grants the Licensor no "
    "ownership of it and makes no claim of authorship over any result the "
    "Software produces from it.",
    "4.2 The Licensor retains all right, title and interest in and to the "
    "Software, subject to the licence granted in Section 2.",

    "5. DIAGNOSTIC DATA AND HOW IT MAY BE USED",
    "5.1 The Software contains no analytics and never uploads Your "
    "Content. Apart from the anonymised error reports described in Section "
    "5.6 and the spaCR AI requests described in Section 6, Diagnostic Data "
    "is transmitted only as the result of an action You take.",
    "5.2 Where You elect to send a bug report, or an anonymised error "
    "report is sent under Section 5.6, You grant the Licensor a perpetual, "
    "worldwide, royalty-free licence to use the Diagnostic Data contained "
    "in that report for the purposes of diagnosing faults, improving the "
    "Software, and any other purpose connected with its development.",
    "5.3 Where You upload image data or trained models to a Shared "
    "Resource, You grant the Licensor a perpetual, worldwide, "
    "royalty-free licence to use that material to develop the Software "
    "further and to produce community resources, including but not limited "
    "to object detection and segmentation models made available to other "
    "users.",
    "5.4 The licences in Sections 5.2 and 5.3 are non-exclusive. They do "
    "not transfer ownership, and they do not limit what You may do with the "
    "same material.",
    "5.5 A bug report You choose to file yourself under Your own GitHub "
    "account is PUBLISHED as an issue on the public spaCR repository, "
    "github.com/EinarOlafsson/spacr. Before it is sent, credential-shaped "
    "values, database file names, and this computer's login and host names "
    "are replaced with placeholders, and so are file and folder names "
    "unless You restore them in a preview. Public issues are "
    "world-readable, indexed and mirrored, and cannot be reliably "
    "unpublished. The log is never published: any log accompanying a "
    "report is written to a file on Your equipment, and sending that file "
    "is a separate act You take.",
    "5.6 ANONYMISED ERROR REPORTS ARE ON BY DEFAULT. When a run fails, is "
    "refused for its settings, or the Software crashes, the Software sends "
    "an anonymised summary of the error so that it can be fixed without "
    "You having to report it. The summary holds the versions of the "
    "Software, Python and its principal libraries, the operating system "
    "family, the type of GPU, the screen and module in use, the error type, "
    "the error message with personal details removed, a call stack that "
    "names files without their folders, an AI-written summary of the "
    "likely cause, and a random install identifier that is not derived "
    "from Your equipment and is used only to share the free allowance "
    "fairly. The following are NEVER sent: file and folder paths, file "
    "names, user, login and host names, IP addresses, e-mail addresses, "
    "tokens and other credentials, logs, settings values, environment "
    "variables, images, and any other part of Your Content. The summary is "
    "sent over an encrypted connection to spaCR's relay, which files it as "
    "a PUBLIC issue on the GitHub repository EinarOlafsson/spacr, or adds "
    "it as a comment to an open issue for the same error. The relay files "
    "it under its own name, so You need no GitHub account. Public issues "
    "are world-readable, indexed and mirrored, and cannot be reliably "
    "unpublished.",
    "5.7 Under Preferences → AI, the setting “AI Error "
    "report” chooses what happens to an error: summarize and send all "
    "errors (the default), with or without code context from the "
    "Software's own source; summarize all errors and send only faults in "
    "the Software; summarize all errors and send nothing; send without an "
    "AI summary; or “Off”. Its “What is sent…” "
    "button shows an example report. To stop all error reports, choose "
    "“Off” or “Summarize all errors, send nothing” "
    "there, or turn off “Send anonymised error reports” in the "
    "notice shown when spaCR first starts. BY ACCEPTING THIS AGREEMENT "
    "WITHOUT TURNING REPORTS OFF, YOU AGREE TO THE SENDING OF THESE "
    "ANONYMISED REPORTS.",
    "5.8 You are responsible for the content of anything You elect to send "
    "or allow to be sent automatically. The Software removes personal "
    "details, credential-shaped values and file paths on a best efforts "
    "basis; it cannot identify material that is confidential for reasons "
    "particular to Your work.",

    "6. spaCR AI AND THIRD PARTY SERVICES",
    "6.1 spaCR AI (free) is ON BY DEFAULT. It explains errors in the "
    "console and answers questions You ask the assistant, and it needs no "
    "account. Your question, the error summary described in Section 5.6 "
    "and the name of the screen in use are sent to spaCR's relay, which "
    "has them answered by a model running on Cloudflare Workers AI. "
    "Cloudflare processes them under its own terms.",
    "6.2 spaCR AI is shared among all its users. Each install receives a "
    "daily fair share of the free capacity: after it is used, answers come "
    "from a lighter model, and then stop for the rest of the day. An "
    "answer to the same error or to the same short question may be saved "
    "and given again to any user, without asking the model again.",
    "6.3 An answer from spaCR AI, or from any coding assistant, is "
    "machine-generated advice. It may be incomplete or wrong, and You are "
    "responsible for checking it before You act on it.",
    "6.4 The relay keeps usage counters per random install identifier, "
    "saved answers and links to the issues it has filed. It keeps no "
    "personal data: no names, e-mail addresses, IP addresses, file paths "
    "or Your Content. Saved answers expire after 30 days.",
    "6.5 You may use Your own Claude, GPT (Codex) or Gemini assistant "
    "instead: choose it under Preferences → AI → Provider, and "
    "use the Providers… button there to install its command line tool "
    "and sign in. Requests are then made through that vendor's own command "
    "line tool under Your own account and are governed by that vendor's "
    "terms; they do not pass through spaCR's relay. The Licensor does not "
    "receive Your credentials and is not a party to those communications.",
    "6.6 The Software queries public package indexes to determine whether a "
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
    "9.3 Sections 3, 4, 5, 7, 8 and 10 survive termination.",

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
    "11.3 This Agreement and the applicable licences described in Section "
    "2 govern the Software. Where they differ, the applicable licence "
    "governs. Separately licensed material retains its own terms.",
    "11.4 LANGUAGE. This Agreement is written in English. spaCR may present a translation of it for convenience, and a translation is not a separate agreement: if a translated version differs from the English in any respect, THE ENGLISH VERSION GOVERNS. You accept the English Agreement whichever language You read it in.",
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
_KEY_LICENSE = "onboarding/terms_agreed_license"


def _settings():
    """Open spaCR's ``QSettings``.

    :returns: the settings store.
    """
    from .preferences import _settings as store

    return store()


def terms_text(language: str | None = None) -> str:
    """Return the terms with a blank line between clauses.

    :param language: language code to present the agreement in. ``None`` uses
        the profile's current language. A locale with no translated
        presentation gets the English, which is always correct here.

    The English text is the agreement: section 11.4 states that a translation
    is a convenience and the English governs. So a missing or partial
    translation is never an error condition -- it falls back and the document a
    profile accepts is unchanged, which is why `record_agreement` stores
    `TERMS_VERSION` and nothing about the language on screen.
    """
    if language is None:
        try:
            from .i18n import current_language

            language = current_language()
        except Exception:                                    # noqa: BLE001
            language = "en"
    if str(language) != "en":
        try:
            from . import terms_i18n

            translated = terms_i18n.paragraphs(str(language))
            if translated:
                return "\n\n".join(translated)
        except Exception:                                    # noqa: BLE001
            LOG.debug("no translated agreement for %s", language,
                      exc_info=True)
    return "\n\n".join(TERMS)


def agreed_version() -> str:
    """The terms version this profile accepted, or ``""`` if none."""
    try:
        return str(_settings().value(_KEY_VERSION, "") or "")
    except Exception:                                        # noqa: BLE001
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
    version = agreed_version()
    try:
        accepted_license = str(_settings().value(_KEY_LICENSE, "") or "")
    except Exception:
        accepted_license = ""
    if not accepted_license and version in ("4.0", "4.1", "4.2", "4.3"):
        accepted_license = "BSD 3-Clause License"
    return {
        "version": version,
        "accepted_at": agreed_at(),
        "current_version": TERMS_VERSION,
        "license": accepted_license,
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
    store.setValue(_KEY_LICENSE, LICENSE_NAME if stamped == TERMS_VERSION else
                   "BSD 3-Clause License" if stamped in ("4.0", "4.1", "4.2", "4.3") else "")
    try:
        store.sync()
    except Exception:                                        # noqa: BLE001
        pass
    from .ai.issue_report import _sync_automatic_failure_policy
    _sync_automatic_failure_policy()
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
    ("Review the terms of use and full license. Use is noncommercial by default; commercial use requires separate written permission.", (
        'Läs användningsvillkoren och hela licensen. Användning är som standard icke-kommersiell; kommersiell användning kräver separat skriftligt tillstånd.',
        'Lesen Sie die Nutzungsbedingungen und die vollständige Lizenz. Standardmäßig ist die Nutzung nichtkommerziell; kommerzielle Nutzung erfordert eine gesonderte schriftliche Erlaubnis.',
        'Revise las condiciones de uso y la licencia completa. El uso es no comercial por defecto; el uso comercial requiere autorización escrita por separado.',
        '请阅读使用条款和完整许可证。默认仅供非商业用途；商业使用须另行获得书面许可。',
        'Leia os termos de uso e a licença completa. Por predefinição, o uso é não comercial; o uso comercial exige autorização escrita separada.',
        'उपयोग की शर्तें और पूरा लाइसेंस पढ़ें। डिफ़ॉल्ट रूप से उपयोग गैर-व्यावसायिक है; व्यावसायिक उपयोग के लिए अलग से लिखित अनुमति आवश्यक है।',
        '이용 약관과 전체 라이선스를 읽으세요. 기본적으로 비상업적 사용이 허용되며, 상업적 사용에는 별도의 서면 허가가 필요합니다.',
        'Lesið notkunarskilmálana og leyfið í heild. Notkun er sjálfgefið ekki í viðskiptaskyni; notkun í viðskiptaskyni krefst sérstaks skriflegs leyfis.',
        'Lisez les conditions d’utilisation et la licence complète. L’utilisation est non commerciale par défaut ; toute utilisation commerciale nécessite une autorisation écrite distincte.',
    )),
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
        return 0
    added = 0
    for source, values in TRANSLATIONS:
        try:
            added += bool(add_translation(source, values))
        except ValueError:
            continue
    return added
