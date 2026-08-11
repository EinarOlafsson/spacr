|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

spaCR
=====

भाषाएँ: `English <../../../README.rst>`_ · `Svenska <README.sv.rst>`_ ·
`Deutsch <README.de.rst>`_ ·
`Español <README.es.rst>`_ ·
`简体中文 <README.zh_CN.rst>`_ ·
`Português <README.pt.rst>`_ ·
`हिन्दी <README.hi.rst>`_ ·
`한국어 <README.ko.rst>`_ ·
`Íslenska <README.is.rst>`_ ·
`Français <README.fr.rst>`_

`अनुवाद मॉडल की जानकारी <../TRANSLATION_MODELS.md>`_

**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**

spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रत्येक कोशिका को मिले gRNA से जोड़ता है और बताता है कि किन जीनों ने फीनोटाइप बदला। इनपुट के रूप में प्लेट छवियाँ और FASTQ रीड आती हैं; आउटपुट में प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव आकार तथा प्राथमिकता के अनुसार परिणामों की सूची मिलती है।

छवि-आधारित पूल्ड CRISPR स्क्रीनिंग के लिए यह पूरा कार्यप्रवाह है। यदि आपके पास उच्च-सामग्री माइक्रोस्कोपी है लेकिन कोई स्क्रीनिंग नहीं है, तो विभाजन, मापन, एनोटेशन और वर्गीकरण वाले भाग स्वतंत्र रूप से चलाए जा सकते हैं।

छवियाँ, मास्क, इमेज क्रॉप, मापन, एनोटेशन, पूर्वानुमान, बारकोड और वेल पहचानकर्ता एक ही SQLite प्रोजेक्ट में रहते हैं, इसलिए किसी परिणाम के मान को उसके स्रोत ऑब्जेक्ट तक वापस खोजा जा सकता है।

spaCR को डेस्कटॉप एप्लिकेशन के रूप में या वर्कस्टेशन, सर्वर अथवा क्लस्टर पर बिना ग्राफ़िकल इंटरफ़ेस के चलाएँ। दोनों तरीके समान मॉड्यूल चलाते हैं और समर्थित मॉड्यूल में CUDA अपने आप उपयोग होता है।


कार्यप्रवाह का अवलोकन
---------------------

|Tutorials|

.. image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/flow_chart_v3.png
   :alt: spaCR workflow and output organization
   :align: center

माइक्रोस्कोपी छवियाँ (TIFF, OME-TIFF, LIF, CZI, ND2) और सीक्वेंसिंग रीड (FASTQ) पूरक इमेज-विश्लेषण तथा बारकोड-मैपिंग कार्यप्रवाह में जाती हैं। इसके बाद ऑब्जेक्ट तालिकाएँ, इमेज क्रॉप, एनोटेशन, पूर्वानुमान, गाइड पहचान, QC परिणाम और प्रति-वेल सारांश एक साथ विश्लेषित किए जाते हैं।


त्वरित शुरुआत
-------------

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR Python ** 3.9 से 3.14** तक का समर्थन करता है (जिसके अलावा Python 3.14.1, जो टॉर्चविजन को छोड़ देता है) Python 3.12 वैज्ञानिक पैकेजों के सबसे व्यापक विकल्प है. Linux CUDA कार्यप्रवाहों के लिए सिफारिश की जाती है; macOS और Windows भी समर्थित हैं।


स्थापना विवरण
--------------------

|Release| |PyPI| |CondaRecipe| के लिए

**(बेटा) Lightweight डेस्कटॉप इंस्टॉलर:**

.. spacr-installer-links-begin

* `Windows 10/11: डाउनलोड SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe>`_
* `macOS 11+ (इंटेल और एप्पल सिलिकॉन): डाउनलोड SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg>`_
* `64-बिट Linux: डाउनलोड SpaCR 1.5.0.4 <https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run>`_

.. spacr-installer-links-end

हल्के इंस्टॉलर — conda या पहले से स्थापित Python की आवश्यकता नहीं
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

इंस्टॉलर एक निजी Python 3.12 चलने का समय, Qt, PyTorch, spaCR और स्थापना के दौरान वैज्ञानिक निर्भरताओं को डाउनलोड करता है, इसलिए कोई कंड या मौजूदा Python की आवश्यकता नहीं है. पोर्टेबल CPU निर्माण डिफ़ॉल्ट है, जो स्थापना को कई गीगाबाइटों के CUDA पुस्तकालयों को खींचने से रोकता है. Windows एक वैकल्पिक इंस्टॉलर घटक के रूप में NVIDIA गति प्रदान करता है, Linux ``--torch-backend auto`` को स्वीकार करता है, और मानक macOS PyTorch पहियों में Apple MPS गति रखता है.

इंस्टॉलर सहायता, प्रगति और त्रुटियां सभी दस spaCR भाषाओं में ऑपरेटिंग सिस्टम भाषा का पालन करती हैं: अंग्रेजी, स्विड, जर्मन, स्पेनिश, सरल चीनी, पुर्तगाली, भारतीय, कोरियाई, आइसलैंड और फ्रेंच।

Linux पर, इसे खोलने से पहले डाउनलोड किए गए इंस्टाइलर को निष्पादित करें:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

macOS पर, डाउनलोड किया गया ``.pkg`` खोलें. यदि Gatekeeper वर्तमान बीटा इंस्टाइलर को ब्लॉक करता है क्योंकि यह नोटिस नहीं किया गया है, तो ** सिस्टम सेटिंग्स → गोपनीयता और सुरक्षा** खोलें, ** किसी भी तरह से खोलें** spaCR के लिए, फिर पैकेज को फिर से चलाएं.

स्थापनाकर्ता spaCR, Qt, PyTorch और निर्भरता की स्थिरता को पुराने स्थापना को प्रतिस्थापित करने से पहले वैध करता है, इसलिए एक रुकने वाली अपडेट पिछले कार्य वातावरण को स्थान पर छोड़ देता है. एक निदान लॉग ``install.log`` के रूप में निजी स्थापना निर्देशिका spaCR के अंदर रखा जाता है.

PyPI से डेस्कटॉप एप्लिकेशन
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install "spacr[qt]"
   spacr

बिना ग्राफ़िकल इंटरफ़ेस या सर्वर पर स्थापना
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

नवीनतम विकास शाखा
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/EinarOlafsson/spacr.git
   cd spacr
   git switch nightly
   python -m pip install -e ".[qt]"

Conda वातावरण
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n spacr python=3.12 pip -y
   conda activate spacr
   python -m pip install "spacr[qt]"

वैकल्पिक सुविधाएँ
~~~~~~~~~~~~~~~~~~~~~

केवल अतिरिक्त स्थापित करें आपके कार्यप्रवाह की जरूरतों:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

जो एक्सट्रैस रिज़ॉल्यूशन Python संस्करण पर निर्भर करता है. Python 3.13 पर, अल्ट्राक सीमा ``spacr[all]`` और TorchCAM के NumPy सीमा ``attribution`` अतिरिक्त सीमा को सीमित करती है; कोर पैकेज और Qt अनुप्रयोग को प्रभावित नहीं किया जाता है. Python 3.14 पर, btrack इसके अतिरिक्त के माध्यम से उपलब्ध है. pylibCZIrw सीजीआई कनवर्टर वैकल्पिक और परीक्षण नहीं किया गया है; सीजीआई आधारित सीजीआई पढ़ना उपलब्ध है.

विरासत टीके इंटरफ़ेस अभी भी ``spacr-legacy`` के रूप में स्थापित किया गया है लेकिन अब विकसित नहीं किया गया है।


कमांड-लाइन प्रवेश बिंदु
-------------------------

.. code-block:: bash

   spacr                                      # Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

समस्या हल करने के लिए ``SPACR_LOG_LEVEL=DEBUG`` सेट करें. घूर्णन लॉग ``~/.spacr/logs/spacr.log`` में लिखे जाते हैं.


विशेषताएँ
---------

अधिकांश स्क्रीनिंग में उपयोग होने वाले छह मॉड्यूल
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** segments cells, nuclei, pathogens and organelles with Cellpose, in 2D images and in volumetric or time-series data. The model list is read from the installed Cellpose rather than hard-coded, and an object diameter is estimated from the images before the run starts. Masks can be corrected by hand in the layer viewer, or sent to napari and back.

**मेटा ** परियोजना डेटाबेस में परियोजना के मर्फोलॉजी, तीव्रता, संरचना और स्थानिकता सुविधाओं को लिखता है, साथ ही साथ क्रॉपों के साथ. नया 1.5.0.0 में: प्रकाश व्यवस्था प्लेट से फ्लैट फ़ील्ड का अनुमान लगाती है और किसी भी तीव्रता सुविधा को लेने से पहले इसे बाहर निकालती है, जो प्लेट थर्मैप्स को किनारे प्रभाव के रूप में प्रदर्शित करने वाले अच्छी स्थिति के बायस को हटाती है. एक खंड QC बैनर फ्लैट भाषा में बताता है कि मास्क कैसे दिखते हैं, इससे पहले कि माप चलता है; यह सूचित करता है, यह अवरुद्ध नहीं करता है. एक खींचा पॉलीगन रुचि के एक क्षेत्र के लिए माप को सीमित करता है.

**Annotate** एक कीबोर्ड-प्रदर्शन नेटवर्क पर खेती दिखाता है और लेबल सीधे SQLite लिखता है. यह अब सक्रिय-लर्निंग रॉक को बंद करता है: स्क्रीनिंग छोड़ने के बिना लेबल किए गए पर एक मॉडल को रेट्री करें, अनिश्चितता के साथ रेट्री करें, सीखने की कोर को देखें, और एक रुकावट का फैसला प्राप्त करें जब आगे लेबल मॉडल को बदलना बंद कर देते हैं. कवर शायद कक्षा, अच्छी तरह से और प्लेट पर होता है, और प्रत्येक रॉक रिकॉर्ड किया जाता है.

**Classify** trains PyTorch CNNs and transformers on annotated crops, and classical or boosted models on measurement tables. Per-class accuracy is now kept every epoch instead of being discarded, and each checkpoint gets a model card recording its dataset, class balance, split rule and held-out metrics. In the evaluation screen, a confusion-matrix cell is a query: click it to open those crops, with confidently wrong predictions listed apart from uncertain ones.

**मैप बारकोड** FASTQ से पंक्ति, स्तंभ और gRNA बारकोड को खत्म करता है, बर्तनों के लिए मार्गदर्शक पहचान को सौंप देता है, और उन्हें चित्रित कोशिकाओं में जोड़ता है. बारकोड QC रिपोर्टों को बर्तन, टकराव दर और अकादमिक फ्रैक्शन के लिए पढ़ता है, बर्तन के लिए gRNA की संख्या के चारों ओर घूमते हैं, आप कहते हैं कि आप एक निश्चित सीमा के बजाय उम्मीद करते हैं।

**ग्रेसेज** 17 मॉडल परिवारों, मिश्रित मॉडल, रसद और प्रबिट, क्वांटम, बीटा, जीएलएम के साथ क्वाज़ी-बिनोमील वेरिएंट, लासो, रिज, त्वचा तार, हेंग और घोड़े की छड़ी का उपयोग करके हिट्गदर्शन, जीन, स्थिति और नियंत्रण प्रभाव का अनुमान लगाता है। परिणाम एक रैंकिंग, रिकॉर्ड किए गए हिट सूची है, न कि एक कोकोइंट डंप।

1.5.0.0 में नया
~~~~~~~~~~~~~~~

इससे पहले कि एक स्क्रीनिंग मौजूद हो, बिजली / डिजाइन मॉड्यूल जवाब देता है कि कितने सेल और कितने बर्तनों की जरूरत है, अनुक्रम त्रुटि के साथ मूल्यवान और बहुत ही पतली छवि के बर्तनों से आने वाले ड्रॉप के साथ. एक प्रयोग डिजाइनर प्लेट को बाहर रखता है, इसके नियंत्रण और इसके प्रतिलिपि और पाइपलाइन के लिए व्यवस्था को निर्यात करता है. इसके बाद, एक QC डैशबोर्ड एक फैसला में विभाजन, प्लेट, नोटर-समझौता और लीक की जांच करता है, और बैच को ठीक करने के लिए ``center`` और ``zscore`` के बगल में उपलब्ध है।

परिणामों का पता लगाया जाता है और फिर से आयात किया जाता है. एक ग्राफ बिल्डर x, y, रंग, आकार और पहलू पर स्तंभों को खींचकर एक तालिका को खींचता है. एक हिस्टोग्राम या स्केटर पर खींचने वाले दरवाजे फ़िल्टर बन जाते हैं. एक फ़ंक्शन एक्सटोरर वर्गीकरणों को कैसे अच्छी तरह से वे वर्गों को अलग करते हैं. छोटे बहुल, खुराक-प्रश्न फिट, नियंत्रण चार्ट और मजबूत बाहरी पता लगाने का उपयोग एक ही आकार इंजन का उपयोग करता है. एक दृष्टि में वस्तुओं का चयन उन्हें उन सभी में सेट करता है, और एक चयन खोलने से वस्तुओं को उठाया जाता है जो वस्तुओं से आते हैं. एक परत दृश्यर छवियों, लेबल, बिंदुओं और आकारों, एक सिंक्रनात्मक दृष्टि, एक तुलनात्मक नेटवर्क, और एक पेड़ से को को को को को को को को को को को को को को को को को को को को को को को को

Runs are now identifiable. Each carries one run id, one seed and an ``on_error`` policy; Mask, Measure, Classify and the AnnData export register what they wrote in an artifact registry, so an output file leads back to the settings that produced it. A module opens on what the previous step actually wrote, the pipeline graph marks which outputs are stale, run comparison diffs the settings, object counts and hit lists of two runs, and every GUI run emits the equivalent Python script. Measurements export to ``.h5ad`` for scanpy; OME-Zarr and OMERO are available through the Python API. The methods-and-results exporter drafts those two manuscript sections from a structured digest of the run: the model writes the prose, but every number comes from the digest, and a draft containing a number the digest does not contain is rejected. जब स्थापना के साथ कुछ गलत है, तो ``spacr-doctor`` रिपोर्ट करता है कि spaCR वास्तव में चल रहा है, क्या GPU उपयोगी है, क्या Cellpose API spaCR कॉल के अनुरूप है, और क्या परियोजना डेटाबेस और सेटिंग्स ध्वनि हैं, प्रत्येक लाइन पर एक कॉपी-आधारित सुधार के साथ जो एक पास नहीं है।

बहुभाषी डेस्कटॉप इंटरफ़ेस
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → प्राथमिकताएँ → भाषा** अंग्रेजी, स्विडन, जर्मन, स्पेनिश, मंडारिन चीनी, पुर्तगाली, हिंदू, कोरियाई, आइसलैंड या फ्रेंच में चलने वाले अनुप्रयोग को पुनरारंभ किए बिना पुनरारंभ करता है।

Navigation, Preferences, AI and LIVE controls, module descriptions and spaCR-authored console notices follow the selected language. Worker output, logs, tracebacks, paths, database values, annotations, AI responses, measurements and saved results are never translated, so scientific output remains canonical English. Setting tooltips not yet reviewed in a language stay in English rather than becoming a mixed-language explanation. The `स्थान मार्गदर्शिका <https://einarolafsson.github.io/spacr/localization.html>`_ documents the behavior, the environment override, and the `संदर्भ सहायता <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ that is translated with it.

एनिमेटेड सेटिंग गाइड
~~~~~~~~~~~~~~~~~~~~~~~~~

94 संक्षिप्त एनीमेशन बताते हैं कि 143 दृश्य सेटिंग्स एक छवि के लिए क्या करते हैं. एक सेटिंग को हटा दें और ** एनीमेशन** पर क्लिक करें इसके टूल टिप में पाठ के बगल में वर्ग को खेलने के लिए; इसे फिर से फ़्लैट करने के लिए क्लिक करें. एनीमेशन तब तक बंद हो जाते हैं जब तक कि पूछा जाता है, और प्राथमिकताओं में अक्षम हो सकता है. `गैलरी <https://einarolafsson.github.io/spacr/setting_animations.html>`_ उन सभी को दिखाता है, और `एनीमेशन रिकॉर्ड बनाना <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ रिकॉर्ड जिनमें से से से से सेटिंग प्रत्येक के लिए है.

मॉड्यूल संदर्भ
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Module
     - Feature
     - State
     - Description
   * - **Desktop experience**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - Stable
     - Retranslates open and lazily created screens across ten bundled languages.
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - Stable
     - Localizes module summaries and setting-help chrome while preserving exact API URLs.
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - Stable
     - Localizes AI and LIVE controls without changing user or model content.
   * - |api-animations|_
     - |doc-animations|_
     - Stable
     - Plays 94 packaged animations for 143 visual settings from the setting tooltip.
   * - |api-selection|_
     - |api-linked-views|_
     - Alpha
     - Shares one object selection across the table, plate, embedding, scatter and graph views.
   * - |api-doctor|_
     - |api-doctor-checks|_
     - Alpha
     - Diagnoses the install — GPU, Cellpose API, database, settings — with a fix per failing check.
   * - **Image analysis**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - Stable
     - Segments cells, nuclei, pathogens and organelles in 2D images.
   * - |api-mask|_
     - |api-mask-3d|_
     - Beta
     - Segments volumetric images and 4D time series.
   * - |api-illumination|_
     - |api-flatfield|_
     - Alpha
     - Estimates the flat-field from the plate and divides it out before intensity is measured.
   * - |api-measure|_
     - |api-measure-2d|_
     - Stable
     - Measures morphology, intensity, texture and colocalization, and writes the crops.
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - Alpha
     - States what the segmentation looks like before Measure runs, without blocking it.
   * - |api-timelapse|_
     - |api-tracking|_
     - Beta
     - Tracks objects with IoU, Trackpy, btrack, Trackastra or ultrack, and quantifies motility.
   * - |api-layers|_
     - |api-layer-viewer|_
     - Alpha
     - Stacks image, label, point and shape layers, with orthogonal views and a comparison grid.
   * - |api-napari|_
     - |api-napari-curation|_
     - Alpha
     - Hands a mask to napari for correction and takes it back, recording every edit.
   * - **AI and phenotyping**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - Stable
     - Reviews crops on a keyboard-driven grid and saves annotations to SQLite.
   * - |api-active-learning|_
     - |api-al-loop|_
     - Alpha
     - Retrains inside Annotate, re-ranks by uncertainty, and says when labelling can stop.
   * - |api-classify|_
     - |api-classification|_
     - Stable
     - Trains and applies PyTorch CNN and transformer models.
   * - |api-classify|_
     - |api-model-cards|_
     - Alpha
     - Records dataset, class balance, split rule and held-out metrics beside each checkpoint.
   * - |api-confusion|_
     - |api-confusion-drill|_
     - Alpha
     - Opens the crops behind a confusion cell, confident errors listed apart from uncertain ones.
   * - |api-ml|_
     - |api-ml-models|_
     - Stable
     - Trains interpretable classical and boosted models on measurement tables.
   * - |api-classify|_
     - |api-activation|_
     - Beta
     - Explains predictions with Captum, SmoothGrad and TorchCAM.
   * - |api-umap|_
     - |api-embedding|_
     - Beta
     - Explores image embeddings interactively and propagates cluster labels.
   * - **Sequencing and screen analysis**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - Stable
     - Maps row, column and gRNA barcodes from FASTQ reads and assigns guides to imaged cells.
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - Alpha
     - Reports reads per well, collision rate and unmapped fraction against the expected gRNAs per well.
   * - |api-regression|_
     - |api-regression-models|_
     - Stable
     - Estimates guide, gene, condition and control effects with 17 model families.
   * - |api-power|_
     - |api-power-design|_
     - Alpha
     - Answers how many cells and wells a screen needs, with sequencing error and well dropout priced in.
   * - |api-graph|_
     - |api-graph-builder|_
     - Alpha
     - Builds a plot by dragging columns onto x, y, colour, size and facet.
   * - |api-artifacts|_
     - |api-provenance|_
     - Alpha
     - Records the run id, seed and settings behind mask, measure, classify and export outputs.

.. |api-qt-app| replace:: **Qt application**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **Ten-language localization**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **Localized contextual help**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-assisted console**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **Setting animation registry**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **Visual setting animations**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **Selection**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **Linked selection**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **Installation diagnosis**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D mask generation**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D and 4D mask generation**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **Illumination**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **Flat-field correction**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **Object measurements**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **Segmentation QC**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **Pre-run verdict**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **Object tracking**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **Layers**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **Layer viewer**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari bridge**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **Mask curation**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **Manual annotation**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **Active learning**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **Retrain and re-rank**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **Image classification**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **Model cards**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **Activation maps**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **Confusion drill-down**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **Machine learning**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **Measurement classification**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **Interactive embedding**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **Sequencing**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **Map barcodes**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **Barcode QC**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **Well and collision report**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **Screen effect estimation**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **Power and design**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **Artifacts**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **Run provenance**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


डेटा
----

संदर्भ डेटासेट
~~~~~~~~~~~~~~~~~~

- `पूर्ण माइक्रोस्कोप डेटासेट: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `डेटासेट का परीक्षण: चेहरे को हिलाना toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `अनुक्रम डेटा: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `बिजली का विश्लेषण: spaCRPower <https://github.com/maomlab/spaCRPower>`_


योगदान और सहायता
------------------------

बुग रिपोर्ट और केंद्रित फ़ंक्शन अनुरोध `GitHub मुद्दे <https://github.com/EinarOlafsson/spacr/issues>`_ के माध्यम से स्वागत करते हैं. एक विफलता की रिपोर्ट करते समय, इसमें spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग्स और संबंधित लॉग excerpt शामिल हैं. ``spacr-doctor`` आपके लिए इसका अधिकांश संग्रह करता है.

लाइसेंस
~~~~~~~~~

वर्तमान विकास शाखा स्रोत-अनुकूल है `PolyForm गैर-व्यावसायिक लाइसेंस 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. वाणिज्यिक उपयोग के लिए कॉपीराइट धारक से एक अलग लाइसेंस की आवश्यकता होती है. spaCR 1.4.9.9 के माध्यम से जारी संस्करण एमआईटी लाइसेंस के तहत उपलब्ध रहते हैं जो इन रिलीजों के साथ आता है.

ट्यूटोरियल
~~~~~~~~~~

`इंटरैक्टिव spaCR ट्यूटोरियल लाइब्रेरी <https://einarolafsson.github.io/spacr/tutorials/>`_ में आठ भाषाओं में, प्रत्येक अनुप्रयोग के कार्यप्रवाह और स्थापना के वर्णित, वर्णित चलने वाले मार्ग शामिल हैं।

spaCR का संदर्भ
~~~~~~~~~~~~~~~

यदि spaCR आपके शोध में योगदान देता है, तो उद्धरण करें:

Olafsson EB, *et al.* एक संयुक्त छवि-आधारित CRISPR स्क्रीनिंग EAF1 को ESCRT उप-विवाद के *T. gondii* मॉड्यूलर के रूप में पहचानती है।

`BioRxiv प्रीप्रिंट <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `सॉफ्टवेयर संग्रह <https://doi.org/10.5281/zenodo.21343317>`_ के लिए
