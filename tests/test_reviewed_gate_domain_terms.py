"""The sense gates accept spaCR's correct domain words and still catch errors.

The reviewed-translation gates reject a target word when the English uses a
scientific or software sense the word would get wrong. Five rules rejected
correct domain words in batches 3 and 4 of item 678 (2026-10-20):

* "screen" meaning a genetic or image-based screen, read as a GUI screen
  because the same sentence also mentions wells, guides or the GUI;
* Portuguese "poder estatístico", the standard term for statistical power;
* Portuguese "bem" and Spanish/French "bien" for the adverb in "well-tested"
  and in fixed expressions such as "si bien que" and "bem como";
* Chinese 门限 (threshold) for a distance gate;
* Hindi द्वार written with the "(gate)" gloss.

Each case below is a real API source block; the first group must pass and the
second group must still be rejected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_i18n_catalogs import _semantic_false_friends  # noqa: E402

POWER_DESIGN = ("Power / Design — estimate how many cells per well, and how "
                "many wells, a screen needs.")
IMAGE_SCREEN = ("The **Map Barcodes** module connects a sequencing run to an "
                "image-based screen.")
PLATES = ("A well that was never measured is drawn as a neutral wash and left "
          "out of the scale; drawing it as zero would, on a screen with 155 of "
          "384 wells used, fill more than half the panel.")
SCREEN_LEVEL = ("When omitted, the function uses the screen-level ``results`` "
                "folder derived from ``count_data``.")
LEVENE = ("On three replicates Levene has almost no power, so a large p-value "
          "means we could not tell.")
WELL_TESTED = ("Keeping that well-tested implementation and adapting one "
               "settings view at its boundary prevents four copies.")
DISTANCE_GATE = ("gate for the distance-based backends, in xy pixels, with z "
                 "scaled by ``anisotropy``.")
HOME_SCREEN = "Open the Home screen and choose a module."
EACH_WELL = "Each well receives one dye stain."
EACH_WELL_WHILE = "Each well receives one dye stain, so no well is left empty."

ACCEPTED = [
    ("pt", POWER_DESIGN, "Power / Design — estima quantas células por poço, "
     "e quantos poços, uma triagem precisa."),
    ("zh_CN", POWER_DESIGN, "Power / Design — 估算一次筛选需要每孔多少细胞以及多少个孔。"),
    ("hi", POWER_DESIGN, "Power / Design — अनुमान लगाता है कि एक स्क्रीनिंग को "
     "प्रति वेल कितनी कोशिकाएँ और कितने वेल चाहिए।"),
    ("zh_CN", IMAGE_SCREEN, "**Map Barcodes** 模块将一次测序运行与基于图像的筛选关联起来。"),
    ("pt", PLATES, "Um poço nunca medido é desenhado como uma lavagem neutra e "
     "fica fora da escala; desenhá-lo como zero, numa triagem com 155 de 384 "
     "poços usados, preencheria mais da metade do painel."),
    ("pt", SCREEN_LEVEL, "Quando omitido, a função usa a pasta ``results`` da "
     "triagem, derivada de ``count_data``."),
    ("pt", LEVENE, "Com três réplicas, o teste de Levene quase não tem poder "
     "estatístico, então um valor p alto significa que não foi possível dizer."),
    ("pt", WELL_TESTED, "Manter essa implementação bem testada e adaptar uma "
     "visão de configurações no seu limite evita quatro cópias."),
    ("es", WELL_TESTED, "Mantener esa implementación bien probada y adaptar una "
     "vista de ajustes en su límite evita cuatro copias."),
    ("fr", WELL_TESTED, "Conserver cette implémentation bien éprouvée et adapter "
     "une vue de paramètres à sa frontière évite quatre copies."),
    ("fr", EACH_WELL_WHILE, "Chaque puits reçoit un colorant, si bien qu'aucun "
     "puits ne reste vide."),
    ("pt", EACH_WELL_WHILE, "Cada poço recebe um corante, bem como uma "
     "marcação nuclear."),
    ("zh_CN", DISTANCE_GATE, "基于距离的后端所用的距离门限，以 xy 像素为单位，"
     "z 按 ``anisotropy`` 缩放。"),
    ("hi", DISTANCE_GATE, "दूरी-आधारित बैकएंड के लिए द्वार (gate), xy पिक्सेल में, "
     "z को ``anisotropy`` से स्केल करते हुए।"),
]

REJECTED = [
    ("pt", HOME_SCREEN, "Abra a triagem Home e escolha um módulo.",
     "gui-screen-as-scientific-screen"),
    ("pt", EACH_WELL, "Cada bem recebe um corante.", "scientific-well-as-adverb"),
    ("es", EACH_WELL, "Cada bien recibe un colorante.", "scientific-well-as-adverb"),
    ("pt", LEVENE, "Com três réplicas, o teste de Levene quase não tem poder, "
     "então um valor p alto significa que não foi possível dizer.", None),
    ("zh_CN", DISTANCE_GATE, "基于距离的后端所用的门，以 xy 像素为单位，"
     "z 按 ``anisotropy`` 缩放。", None),
    ("hi", DISTANCE_GATE, "दूरी-आधारित बैकएंड के लिए द्वार, xy पिक्सेल में, "
     "z को ``anisotropy`` से स्केल करते हुए।", None),
    ("hi", DISTANCE_GATE, "दूरी-आधारित बैकएंड के लिए दरवाज़ा, xy पिक्सेल में, "
     "z को ``anisotropy`` से स्केल करते हुए।", None),
    ("pt", DISTANCE_GATE, "portão para os back-ends baseados em distância, em "
     "pixels xy, com z escalado por ``anisotropy``.", None),
]


@pytest.mark.parametrize("language, source, target", ACCEPTED)
def test_correct_domain_words_pass_the_sense_gates(language, source, target):
    assert _semantic_false_friends(source, target, language) == ()


@pytest.mark.parametrize("language, source, target, label", REJECTED)
def test_true_sense_errors_are_still_rejected(language, source, target, label):
    failures = _semantic_false_friends(source, target, language)
    assert failures
    if label is not None:
        assert label in failures
