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
TERMS_VERSION = "1.0"

#: What the licence is called, and where the whole of it can be read.
LICENSE_NAME = "PolyForm Noncommercial License 1.0.0"
LICENSE_URL = "https://polyformproject.org/licenses/noncommercial/1.0.0"

#: The notice the licence itself requires be carried with the software.
REQUIRED_NOTICE = "Copyright 2025-2026 Einar Birnir Olafsson."

#: The terms, in the words the user is asked to accept them in.
#:
#: A SUMMARY THE READER CAN ACTUALLY READ, with the whole licence one click
#: away. A hundred lines of licence text on a setup slide is a scroll bar
#: nobody moves, and an agreement nobody read is the thing this is meant to
#: avoid rather than the thing it produces.
TERMS: Tuple[str, ...] = (
    "spaCR is free to use for any noncommercial purpose. Use by a university, "
    "a public research organisation, a charity, a health or environmental "
    "body or a government institution counts as noncommercial whatever the "
    "funding behind it. Commercial use needs a separate licence from the "
    "author.",
    "You may copy it, change it, and share your changes, as long as anyone "
    "you give it to gets these terms and the copyright notice with it.",
    "It comes as is, with no warranty or condition of any kind, and as far as "
    "the law allows the author is not liable for any damage arising from it. "
    "Keep your own backups.",
    "spaCR sends nothing anywhere on its own. Diagnostics, issue reports and "
    "assistant queries leave this machine only when you press send, and only "
    "through tools you have already signed in to.",
)

#: The line shown on the agreement control itself.
AGREE_LABEL = "I have read and agree to these terms"

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
    ("spaCR is licensed on the condition that you accept these terms. They "
     "are short, and the whole licence is a click away.", (
        "spaCR licensieras på villkor att du godkänner dessa villkor. De är "
        "korta, och hela licensen är ett klick bort.",
        "spaCR wird unter der Bedingung lizenziert, dass Sie diese "
        "Bedingungen akzeptieren. Sie sind kurz, und die vollständige Lizenz "
        "ist einen Klick entfernt.",
        "spaCR se licencia con la condición de que aceptes estos términos. "
        "Son breves, y la licencia completa está a un clic.",
        "spaCR 的授权以您接受这些条款为条件。条款很短，完整许可证只需点击一次即可查看。",
        "O spaCR é licenciado sob a condição de que você aceite estes "
        "termos. Eles são curtos, e a licença completa está a um clique.",
        "spaCR का लाइसेंस इस शर्त पर है कि आप इन शर्तों को स्वीकार करें। ये संक्षिप्त हैं, "
        "और पूरा लाइसेंस एक क्लिक दूर है।",
        "spaCR는 이 약관에 동의하는 것을 조건으로 사용이 허가됩니다. 약관은 짧으며 "
        "전체 라이선스는 클릭 한 번으로 볼 수 있습니다.",
        "spaCR er veitt með því skilyrði að þú samþykkir þessa skilmála. "
        "Þeir eru stuttir og allt leyfið er einum smelli í burtu.",
        "spaCR est concédé sous licence à condition que vous acceptiez ces "
        "conditions. Elles sont courtes, et la licence complète est à un "
        "clic.")),
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
