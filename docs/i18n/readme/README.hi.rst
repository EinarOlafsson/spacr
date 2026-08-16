|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: दस्तावेज़
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: इंटरैक्टिव ट्यूटोरियल
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI संस्करण
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 से 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: परीक्षण समूह
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt इंटरफ़ेस
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub स्रोत
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub समस्याएँ
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: PolyForm गैर-व्यावसायिक लाइसेंस
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: नवीनतम इंस्टॉलर
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge रेसिपी

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
   :alt: spaCR कार्यप्रवाह और आउटपुट संगठन
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

spaCR Python **3.9 से 3.14** का समर्थन करता है (Python 3.14.1 को छोड़कर, जिसे torchvision समर्थित नहीं करता)। Python 3.12 पर वैकल्पिक वैज्ञानिक पैकेजों का सबसे व्यापक चयन उपलब्ध है। CUDA कार्यप्रवाह के लिए Linux की सिफ़ारिश की जाती है; macOS और Windows भी समर्थित हैं।


स्थापना विवरण
--------------------

|Release| |PyPI| |CondaRecipe|

**(बीटा) हल्के डेस्कटॉप इंस्टॉलर:**

.. spacr-installer-links-begin

|InstallerWindows| |InstallerMacOS| |InstallerLinux|

.. |InstallerWindows| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11: डाउनलोड SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (इंटेल और एप्पल सिलिकॉन): डाउनलोड SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-बिट Linux: डाउनलोड SpaCR 1.5.0.4
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run

.. spacr-installer-links-end

हल्के इंस्टॉलर — conda या पहले से स्थापित Python की आवश्यकता नहीं
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

इंस्टॉलर स्थापना के दौरान एक निजी Python 3.12 रनटाइम, Qt, PyTorch, spaCR और वैज्ञानिक निर्भरताएँ डाउनलोड करता है, इसलिए conda या पहले से स्थापित Python की आवश्यकता नहीं है। पोर्टेबल CPU बिल्ड डिफ़ॉल्ट है, जिससे कई गीगाबाइट की CUDA लाइब्रेरियाँ बिना सूचना डाउनलोड नहीं होतीं। Windows में NVIDIA एक्सेलरेशन वैकल्पिक घटक है, Linux ``--torch-backend auto`` स्वीकार करता है, और macOS का मानक PyTorch wheel Apple MPS एक्सेलरेशन को बनाए रखता है।

इंस्टॉलर की सहायता, प्रगति और त्रुटि संदेश सभी दस spaCR भाषाओं में ऑपरेटिंग सिस्टम की भाषा का अनुसरण करते हैं: अंग्रेज़ी, स्वीडिश, जर्मन, स्पेनिश, सरलीकृत चीनी, पुर्तगाली, हिन्दी, कोरियाई, आइसलैंडी और फ़्रेंच। असमर्थित लोकेल पर अंग्रेज़ी का उपयोग होता है।

Linux पर डाउनलोड किए गए इंस्टॉलर को खोलने से पहले एक्ज़ीक्यूटेबल बनाएँ:

.. code-block:: bash

   chmod +x spaCR-*-Linux-x86_64-Online.run
   ./spaCR-*-Linux-x86_64-Online.run

macOS पर डाउनलोड की गई ``.pkg`` फ़ाइल खोलें। यदि Gatekeeper वर्तमान बीटा इंस्टॉलर को notarize न होने के कारण रोकता है, तो **सिस्टम सेटिंग्स → गोपनीयता और सुरक्षा** खोलें, spaCR के लिए **फिर भी खोलें** चुनें और पैकेज दोबारा चलाएँ।

पुराने इंस्टॉलेशन को बदलने से पहले इंस्टॉलर spaCR, Qt, PyTorch और सभी निर्भरताओं की संगति की जाँच करता है। इसलिए अपडेट बीच में रुकने पर पिछला कार्यशील वातावरण सुरक्षित रहता है। निदान लॉग निजी spaCR इंस्टॉलेशन निर्देशिका में ``install.log`` नाम से रखा जाता है।

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

केवल वही अतिरिक्त पैकेज इंस्टॉल करें जिनकी आपके कार्यप्रवाह को आवश्यकता है:

.. code-block:: bash

   python -m pip install "spacr[trackastra]"    # transformer tracking
   python -m pip install "spacr[ultrack]"       # global-optimization tracking
   python -m pip install "spacr[btrack]"        # btrack timelapse tracking
   python -m pip install "spacr[attribution]"   # TorchCAM methods
   python -m pip install "spacr[boosting]"      # LightGBM and CatBoost
   python -m pip install "spacr[zernike]"       # Zernike measurements
   python -m pip install "spacr[napari]"        # napari mask correction
   python -m pip install "spacr[czi,nd2,lif]"   # vendor file readers

कौन-से extras स्थापित हो सकते हैं, यह Python संस्करण पर निर्भर करता है। Python 3.13 पर ultrack ``spacr[all]`` को सीमित करता है और TorchCAM की NumPy शर्त ``attribution`` extra को सीमित करती है; मुख्य पैकेज और Qt एप्लिकेशन प्रभावित नहीं होते। Python 3.14 पर btrack अपने extra के माध्यम से उपलब्ध है। pylibCZIrw CZI कनवर्टर वैकल्पिक और अपरीक्षित है; czifile पर आधारित CZI रीडिंग उपलब्ध रहती है।

पुराना Tk इंटरफ़ेस अभी भी ``spacr-legacy`` के रूप में इंस्टॉल होता है, लेकिन अब उसका विकास नहीं किया जाता।


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

समस्या-निवारण के लिए ``SPACR_LOG_LEVEL=DEBUG`` सेट करें। रोटेट होने वाली लॉग फ़ाइलें ``~/.spacr/logs/spacr.log`` में लिखी जाती हैं।


विशेषताएँ
---------

अधिकांश स्क्रीनिंग में उपयोग होने वाले छह मॉड्यूल
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mask** Cellpose की सहायता से 2D छवियों, आयतन डेटा और समय-श्रृंखला में कोशिकाओं, नाभिकों, रोगजनकों तथा कोशिकांगों का खंडन करता है। मॉडल सूची को हार्ड-कोड करने के बजाय स्थापित Cellpose से पढ़ा जाता है और रन शुरू होने से पहले छवियों से ऑब्जेक्ट का व्यास अनुमानित किया जाता है। मास्क को परत व्यूअर में हाथ से सुधारा जा सकता है या सुधार के लिए napari को भेजकर वापस लाया जा सकता है।

**Measure** प्रत्येक ऑब्जेक्ट की आकृति, तीव्रता, बनावट और सह-स्थानीयकरण विशेषताओं को इमेज क्रॉप के साथ प्रोजेक्ट डेटाबेस में लिखता है। 1.5.0.0 में नया: प्रकाश-सुधार प्लेट से ही फ्लैट-फ़ील्ड का अनुमान लगाकर तीव्रता मापने से पहले उसका सुधार करता है; इससे प्लेट हीटमैप में किनारे के प्रभाव की तरह दिखने वाला वेल-स्थिति पूर्वाग्रह हटता है। खंडन QC बैनर Measure चलने से पहले सरल भाषा में मास्क की गुणवत्ता बताता है; वह जानकारी देता है, रन को रोकता नहीं। बनाया गया बहुभुज मापन को रुचि के क्षेत्र तक सीमित करता है।

**Annotate** कीबोर्ड से चलने वाले ग्रिड पर इमेज क्रॉप दिखाता है और लेबल सीधे SQLite में लिखता है। अब पूरा सक्रिय-अधिगम चक्र इसी स्क्रीन पर चलता है: मौजूदा लेबल पर मॉडल को दोबारा प्रशिक्षित करें, अनिश्चितता के अनुसार कतार को फिर क्रमबद्ध करें, सीखने का वक्र देखें और जब अतिरिक्त लेबल से मॉडल बदलना बंद हो तो रुकने का संकेत पाएँ। कवरेज कक्षा, वेल और प्लेट के अनुसार दिखाई जाती है और हर चक्र दर्ज किया जाता है।

**Classify** एनोटेट किए गए इमेज क्रॉप पर PyTorch CNN और ट्रांसफ़ॉर्मर तथा मापन तालिकाओं पर पारंपरिक या बूस्टेड मॉडल प्रशिक्षित करता है। अब प्रत्येक epoch में प्रति-वर्ग सटीकता सुरक्षित रहती है और हर चेकपॉइंट के साथ डेटासेट, वर्ग संतुलन, विभाजन नियम तथा होल्ड-आउट मेट्रिक वाला मॉडल कार्ड बनता है। मूल्यांकन स्क्रीन में कन्फ़्यूज़न मैट्रिक्स की सेल एक क्वेरी की तरह काम करती है: उससे संबंधित क्रॉप खोलने के लिए क्लिक करें; उच्च-विश्वास वाली गलत भविष्यवाणियाँ अनिश्चित भविष्यवाणियों से अलग दिखाई जाती हैं।

**Map Barcodes** FASTQ रीड से पंक्ति, स्तंभ और gRNA बारकोड डिकोड करता है, वेल को गाइड पहचान देता है और उन्हें इमेज की गई कोशिकाओं से जोड़ता है। बारकोड QC हर वेल की रीड संख्या, टकराव दर और न मिले रीड का अनुपात बताता है। यह निश्चित सीमा के बजाय आपके बताए प्रत्याशित प्रति-वेल gRNA संख्या के आसपास परीक्षण करता है।

**Regression** 17 मॉडल परिवारों से गाइड, जीन, स्थिति और नियंत्रण प्रभावों का अनुमान लगाता है। इनमें मिश्रित मॉडल, logistic, probit, quantile, beta, quasi-binomial variance वाले GLM, lasso, ridge, elastic net, hinge और horseshoe शामिल हैं। परिणाम केवल गुणांकों का ढेर नहीं, बल्कि क्रमबद्ध और टिप्पणी-युक्त संभावित परिणामों की सूची होती है।

1.5.0.0 में नया
~~~~~~~~~~~~~~~

स्क्रीनिंग शुरू होने से पहले Power / Design मॉड्यूल आवश्यक कोशिकाओं और वेल की संख्या बताता है; इस गणना में अनुक्रमण त्रुटि और बहुत कम कोशिकाएँ चित्रित होने के कारण छूटने वाले वेल, दोनों शामिल होते हैं। प्रयोग डिज़ाइनर प्लेट, नियंत्रण और प्रतिकृतियों को व्यवस्थित करके पाइपलाइन के लिए लेआउट निर्यात करता है। इसके बाद QC डैशबोर्ड खंडन, प्लेट, एनोटेटर सहमति और डेटा रिसाव संबंधी जाँचों को एक निर्णय में जोड़ता है; बैच सुधार के लिए ``center`` और ``zscore`` के साथ ComBat भी उपलब्ध है।

परिणामों को निर्यात करके दोबारा आयात करने के बजाय सीधे spaCR में खोजा जा सकता है। Graph Builder में स्तंभों को x, y, रंग, आकार और पहलू पर खींचकर तालिका का ग्राफ़ बनाया जाता है। आवृत्ति-चित्र या बिखराव-चित्र पर बनाए गए गेट फ़िल्टर बन जाते हैं। Feature Explorer विशेषताओं को इस आधार पर क्रम देता है कि वे कक्षाओं को कितनी अच्छी तरह अलग करती हैं। छोटे बहु-चित्र, मात्रा–प्रतिक्रिया फ़िट, नियंत्रण चार्ट और सुदृढ़ असामान्य-मान पहचान एक ही अक्ष-इंजन का उपयोग करते हैं। एक दृश्य में ऑब्जेक्ट चुनने पर वह सभी दृश्यों में चुना जाता है, और चयन खोलने पर उससे संबंधित इमेज क्रॉप दिखते हैं। Layer Viewer छवियों, लेबल, बिंदुओं और आकारों को परतों में दिखाता है; इसमें लंबवत दृश्य, समकालिक तुलना ग्रिड और कोशिका से नाभिक होते हुए रोगजनक तक का वंश-वृक्ष शामिल है।

अब प्रत्येक रन की स्पष्ट पहचान और ट्रेसिंग की जा सकती है। हर रन में एक रन ID, एक सीड और ``on_error`` नीति होती है; Mask, Measure, Classify और AnnData निर्यात अपने आउटपुट को आर्टिफ़ैक्ट रजिस्ट्री में दर्ज करते हैं, इसलिए किसी आउटपुट फ़ाइल से उसे बनाने वाली सेटिंग तक पहुँचा जा सकता है। मॉड्यूल पिछले चरण द्वारा वास्तव में लिखे गए आउटपुट पर खुलता है, पाइपलाइन ग्राफ़ पुराने हो चुके आउटपुट चिह्नित करता है, रन तुलना दो रन की सेटिंग, ऑब्जेक्ट संख्या और संभावित परिणामों की सूचियों में अंतर दिखाती है, और हर GUI रन समतुल्य Python स्क्रिप्ट भी बनाता है। मापन scanpy के लिए ``.h5ad`` में निर्यात होते हैं; OME-Zarr और OMERO Python API के माध्यम से उपलब्ध हैं। विधि एवं परिणाम निर्यातक रन के संरचित सार से पांडुलिपि के वे दोनों अनुभाग तैयार करता है: मॉडल गद्य लिखता है, पर प्रत्येक संख्या सार से आती है, और सार में अनुपस्थित संख्या वाला मसौदा अस्वीकार कर दिया जाता है। स्थापना में समस्या होने पर ``spacr-doctor`` बताता है कि वास्तव में कौन-सा spaCR चल रहा है, GPU उपयोग योग्य है या नहीं, Cellpose spaCR द्वारा उपयोग किए जाने वाले API से मेल खाता है या नहीं, और प्रोजेक्ट डेटाबेस तथा सेटिंग मान्य हैं या नहीं; हर विफल जाँच के लिए कॉपी किया जा सकने वाला समाधान भी देता है।

बहुभाषी डेस्कटॉप इंटरफ़ेस
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**spaCR → प्राथमिकताएँ → भाषा** से चल रहे एप्लिकेशन की भाषा बिना पुनः आरंभ किए अंग्रेज़ी, स्वीडिश, जर्मन, स्पेनिश, मंदारिन चीनी, पुर्तगाली, हिन्दी, कोरियाई, आइसलैंडिक या फ़्रेंच में बदली जा सकती है। यह चयन सहेजा रहता है और बाद में खुलने वाली स्क्रीन भी उसी भाषा का उपयोग करती हैं।

नेविगेशन, प्राथमिकताएँ, AI और LIVE नियंत्रण, मॉड्यूल विवरण तथा spaCR द्वारा लिखे गए कंसोल संदेश चुनी हुई भाषा में दिखाई देते हैं। वर्कर आउटपुट, लॉग, ट्रेसबैक, पथ, डेटाबेस मान, एनोटेशन, AI उत्तर, मापन और सहेजे गए परिणाम कभी अनुवादित नहीं किए जाते, इसलिए वैज्ञानिक आउटपुट मानक अंग्रेज़ी में ही रहता है। जिन सेटिंग टूलटिप का किसी भाषा में अभी मानवीय पुनरीक्षण नहीं हुआ है, वे मिश्रित-भाषा विवरण बनने के बजाय अंग्रेज़ी में रहते हैं। `स्थानीयकरण मार्गदर्शिका <https://einarolafsson.github.io/spacr/localization.html>`_ में इस व्यवहार, पर्यावरण ओवरराइड और साथ में अनुवादित होने वाली `संदर्भ-सहायता <https://einarolafsson.github.io/spacr/localization.html#contextual-help>`_ का विवरण है।

एनिमेटेड सेटिंग मार्गदर्शन
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

94 छोटे एनिमेशन दिखाते हैं कि 143 दृश्य सेटिंग किसी छवि को कैसे प्रभावित करती हैं। किसी सेटिंग पर पॉइंटर रखें और उसके टूलटिप में **एनिमेशन** पर क्लिक करके पाठ के पास वाला चौकोर पूर्वावलोकन चलाएँ; दोबारा क्लिक करने पर वह बंद हो जाएगा। एनिमेशन माँगे जाने तक बंद रहते हैं और प्राथमिकताओं में पूरी तरह अक्षम भी किए जा सकते हैं। `गैलरी <https://einarolafsson.github.io/spacr/setting_animations.html>`_ में सभी एनिमेशन हैं और `सेटिंग एनिमेशन रजिस्ट्री <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ बताती है कि प्रत्येक एनिमेशन किस सेटिंग से संबंधित है।

मॉड्यूल संदर्भ
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - मॉड्यूल
     - सुविधा
     - स्थिति
     - विवरण
   * - **डेस्कटॉप अनुभव**
     -
     -
     -
   * - |api-qt-app|_
     - |doc-i18n|_
     - स्थिर
     - खुली और आवश्यकता पर बनाई जाने वाली स्क्रीन को दस अंतर्निहित भाषाओं में तुरंत पुनः अनुवादित करता है।
   * - |api-qt-app|_
     - |doc-i18n-help|_
     - स्थिर
     - API URL को ज्यों का त्यों रखते हुए मॉड्यूल सारांश और सेटिंग-सहायता इंटरफ़ेस का स्थानीयकरण करता है।
   * - |api-qt-ai|_
     - |api-qt-ai-console|_
     - स्थिर
     - उपयोगकर्ता या मॉडल की सामग्री बदले बिना AI और LIVE नियंत्रणों का स्थानीयकरण करता है।
   * - |api-animations|_
     - |doc-animations|_
     - स्थिर
     - सेटिंग टूलटिप से 143 दृश्य सेटिंग के लिए 94 अंतर्निहित एनिमेशन चलाता है।
   * - |api-selection|_
     - |api-linked-views|_
     - अल्फ़ा
     - तालिका, प्लेट, एम्बेडिंग, स्कैटर और ग्राफ़ दृश्यों में एक ही ऑब्जेक्ट चयन साझा करता है।
   * - |api-doctor|_
     - |api-doctor-checks|_
     - अल्फ़ा
     - GPU, Cellpose API, डेटाबेस और सेटिंग की जाँच करता है तथा हर विफल जाँच के लिए समाधान देता है।
   * - **छवि विश्लेषण**
     -
     -
     -
   * - |api-mask|_
     - |api-mask-2d|_
     - स्थिर
     - 2D छवियों में कोशिकाओं, नाभिकों, रोगजनकों और कोशिकांगों का खंडन करता है।
   * - |api-mask|_
     - |api-mask-3d|_
     - बीटा
     - आयतन छवियों और 4D समय-श्रृंखला का खंडन करता है।
   * - |api-illumination|_
     - |api-flatfield|_
     - अल्फ़ा
     - प्लेट से फ्लैट-फ़ील्ड का अनुमान लगाकर तीव्रता मापने से पहले उसका सुधार करता है।
   * - |api-measure|_
     - |api-measure-2d|_
     - स्थिर
     - आकृति, तीव्रता, बनावट और सह-स्थानीयकरण मापता है तथा इमेज क्रॉप सहेजता है।
   * - |api-segqc|_
     - |api-segqc-verdict|_
     - अल्फ़ा
     - Measure चलने से पहले खंडन की गुणवत्ता बताता है, पर प्रक्रिया को रोकता नहीं है।
   * - |api-timelapse|_
     - |api-tracking|_
     - बीटा
     - IoU, Trackpy, btrack, Trackastra या ultrack से ऑब्जेक्ट ट्रैक करता है और गतिशीलता मापता है।
   * - |api-layers|_
     - |api-layer-viewer|_
     - अल्फ़ा
     - ऑर्थोगोनल दृश्य और तुलना ग्रिड के साथ छवि, लेबल, बिंदु और आकृति परतें एक साथ दिखाता है।
   * - |api-napari|_
     - |api-napari-curation|_
     - अल्फ़ा
     - सुधार के लिए मास्क napari को भेजकर वापस लेता है और हर संपादन दर्ज करता है।
   * - **AI और फीनोटाइपिंग**
     -
     -
     -
   * - |api-annotate|_
     - |api-annotation|_
     - स्थिर
     - कीबोर्ड-संचालित ग्रिड में इमेज क्रॉप की समीक्षा करता है और एनोटेशन SQLite में सहेजता है।
   * - |api-active-learning|_
     - |api-al-loop|_
     - अल्फ़ा
     - Annotate के भीतर पुनः प्रशिक्षण करता है, अनिश्चितता के अनुसार कतार दोबारा क्रमित करता है और बताता है कि लेबलिंग कब रोकी जा सकती है।
   * - |api-classify|_
     - |api-classification|_
     - स्थिर
     - PyTorch CNN और ट्रांसफ़ॉर्मर मॉडल को प्रशिक्षित और लागू करता है।
   * - |api-classify|_
     - |api-model-cards|_
     - अल्फ़ा
     - हर चेकपॉइंट के साथ डेटासेट, वर्ग संतुलन, विभाजन नियम और होल्ड-आउट मेट्रिक दर्ज करता है।
   * - |api-confusion|_
     - |api-confusion-drill|_
     - अल्फ़ा
     - कन्फ़्यूज़न मैट्रिक्स की किसी सेल से संबंधित क्रॉप खोलता है और उच्च-विश्वास वाली गलतियों को अनिश्चित नमूनों से अलग दिखाता है।
   * - |api-ml|_
     - |api-ml-models|_
     - स्थिर
     - मापन तालिकाओं पर व्याख्यायोग्य पारंपरिक और बूस्टेड मॉडल प्रशिक्षित करता है।
   * - |api-classify|_
     - |api-activation|_
     - बीटा
     - Captum, SmoothGrad और TorchCAM से पूर्वानुमानों की व्याख्या करता है।
   * - |api-umap|_
     - |api-embedding|_
     - बीटा
     - इमेज एम्बेडिंग को इंटरैक्टिव रूप से खोजता है और क्लस्टर लेबल प्रसारित करता है।
   * - **सीक्वेंसिंग और स्क्रीन विश्लेषण**
     -
     -
     -
   * - |api-sequencing|_
     - |api-barcodes|_
     - स्थिर
     - FASTQ रीड से पंक्ति, स्तंभ और gRNA बारकोड मैप करता है तथा इमेज की गई कोशिकाओं को गाइड सौंपता है।
   * - |api-barcode-qc|_
     - |api-barcode-qc-sweep|_
     - अल्फ़ा
     - प्रति वेल अपेक्षित gRNA की संख्या के संदर्भ में प्रति-वेल रीड, टकराव दर और अमैप्ड अंश रिपोर्ट करता है।
   * - |api-regression|_
     - |api-regression-models|_
     - स्थिर
     - 17 मॉडल परिवारों से गाइड, जीन, दशा और नियंत्रण प्रभावों का अनुमान लगाता है।
   * - |api-power|_
     - |api-power-design|_
     - अल्फ़ा
     - सीक्वेंसिंग त्रुटि और वेल ड्रॉपआउट को ध्यान में रखकर बताता है कि स्क्रीन के लिए कितनी कोशिकाएँ और वेल चाहिए।
   * - |api-graph|_
     - |api-graph-builder|_
     - अल्फ़ा
     - स्तंभों को x, y, रंग, आकार और फ़ैसेट पर खींचकर प्लॉट बनाता है।
   * - |api-artifacts|_
     - |api-provenance|_
     - अल्फ़ा
     - Mask, Measure, Classify और निर्यात आउटपुट के पीछे के रन ID, सीड और सेटिंग दर्ज करता है।

.. |api-qt-app| replace:: **Qt एप्लिकेशन**
.. _api-qt-app: https://einarolafsson.github.io/spacr/api/spacr/qt/app/index.html

.. |doc-i18n| replace:: **दस भाषाओं में स्थानीयकरण**
.. _doc-i18n: https://einarolafsson.github.io/spacr/localization.html

.. |doc-i18n-help| replace:: **स्थानीयकृत संदर्भ-सहायता**
.. _doc-i18n-help: https://einarolafsson.github.io/spacr/localization.html#contextual-help

.. |api-qt-ai| replace:: **Qt AI**
.. _api-qt-ai: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-qt-ai-console| replace:: **AI-सहायित कंसोल**
.. _api-qt-ai-console: https://einarolafsson.github.io/spacr/api/spacr/qt/ai/index.html

.. |api-animations| replace:: **सेटिंग एनिमेशन रजिस्ट्री**
.. _api-animations: https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html

.. |doc-animations| replace:: **दृश्य सेटिंग एनिमेशन**
.. _doc-animations: https://einarolafsson.github.io/spacr/setting_animations.html

.. |api-selection| replace:: **चयन**
.. _api-selection: https://einarolafsson.github.io/spacr/api/spacr/selection/index.html

.. |api-linked-views| replace:: **लिंक किया हुआ चयन**
.. _api-linked-views: https://einarolafsson.github.io/spacr/api/spacr/qt/linked_selection/index.html

.. |api-doctor| replace:: **Doctor**
.. _api-doctor: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-doctor-checks| replace:: **स्थापना निदान**
.. _api-doctor-checks: https://einarolafsson.github.io/spacr/api/spacr/doctor/index.html

.. |api-mask| replace:: **Mask**
.. _api-mask: https://einarolafsson.github.io/spacr/api/spacr/core/index.html

.. |api-mask-2d| replace:: **2D मास्क निर्माण**
.. _api-mask-2d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-mask-3d| replace:: **3D और 4D मास्क निर्माण**
.. _api-mask-3d: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks

.. |api-illumination| replace:: **प्रकाश सुधार**
.. _api-illumination: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-flatfield| replace:: **फ्लैट-फ़ील्ड सुधार**
.. _api-flatfield: https://einarolafsson.github.io/spacr/api/spacr/illumination/index.html

.. |api-measure| replace:: **Measure**
.. _api-measure: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html

.. |api-measure-2d| replace:: **ऑब्जेक्ट मापन**
.. _api-measure-2d: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html#spacr.measure.measure_crop

.. |api-segqc| replace:: **खंडन गुणवत्ता नियंत्रण**
.. _api-segqc: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-segqc-verdict| replace:: **रन-पूर्व मूल्यांकन**
.. _api-segqc-verdict: https://einarolafsson.github.io/spacr/api/spacr/seg_qc/index.html

.. |api-timelapse| replace:: **Timelapse**
.. _api-timelapse: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-tracking| replace:: **ऑब्जेक्ट ट्रैकिंग**
.. _api-tracking: https://einarolafsson.github.io/spacr/api/spacr/timelapse/index.html

.. |api-layers| replace:: **परतें**
.. _api-layers: https://einarolafsson.github.io/spacr/api/spacr/layers/index.html

.. |api-layer-viewer| replace:: **परत व्यूअर**
.. _api-layer-viewer: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html

.. |api-napari| replace:: **napari सेतु**
.. _api-napari: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-napari-curation| replace:: **मास्क सुधार**
.. _api-napari-curation: https://einarolafsson.github.io/spacr/api/spacr/napari_bridge/index.html

.. |api-annotate| replace:: **Annotate**
.. _api-annotate: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-annotation| replace:: **मैन्युअल एनोटेशन**
.. _api-annotation: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html

.. |api-active-learning| replace:: **सक्रिय अधिगम**
.. _api-active-learning: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-al-loop| replace:: **पुनः प्रशिक्षण और पुनः क्रम निर्धारण**
.. _api-al-loop: https://einarolafsson.github.io/spacr/api/spacr/active_learning/index.html

.. |api-classify| replace:: **Classify**
.. _api-classify: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-classification| replace:: **छवि वर्गीकरण**
.. _api-classification: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-model-cards| replace:: **मॉडल कार्ड**
.. _api-model-cards: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-activation| replace:: **सक्रियण मानचित्र**
.. _api-activation: https://einarolafsson.github.io/spacr/api/spacr/deep_spacr/index.html

.. |api-confusion| replace:: **Confusion**
.. _api-confusion: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-confusion-drill| replace:: **कन्फ़्यूज़न मैट्रिक्स की विस्तृत जाँच**
.. _api-confusion-drill: https://einarolafsson.github.io/spacr/api/spacr/confusion/index.html

.. |api-ml| replace:: **मशीन लर्निंग**
.. _api-ml: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-ml-models| replace:: **मापन वर्गीकरण**
.. _api-ml-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-umap| replace:: **Image UMAP**
.. _api-umap: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-embedding| replace:: **इंटरैक्टिव एम्बेडिंग**
.. _api-embedding: https://einarolafsson.github.io/spacr/api/spacr/app_umap/index.html

.. |api-sequencing| replace:: **सीक्वेंसिंग**
.. _api-sequencing: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcodes| replace:: **बारकोड मैपिंग**
.. _api-barcodes: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html

.. |api-barcode-qc| replace:: **बारकोड गुणवत्ता नियंत्रण**
.. _api-barcode-qc: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-barcode-qc-sweep| replace:: **वेल और टकराव रिपोर्ट**
.. _api-barcode-qc-sweep: https://einarolafsson.github.io/spacr/api/spacr/sequencing_qc/index.html

.. |api-regression| replace:: **Regression**
.. _api-regression: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-regression-models| replace:: **स्क्रीन प्रभाव अनुमान**
.. _api-regression-models: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html

.. |api-power| replace:: **Power**
.. _api-power: https://einarolafsson.github.io/spacr/api/spacr/power_model/index.html

.. |api-power-design| replace:: **सांख्यिकीय शक्ति और डिज़ाइन**
.. _api-power-design: https://einarolafsson.github.io/spacr/api/spacr/power_simulate/index.html

.. |api-graph| replace:: **Graph**
.. _api-graph: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_spec/index.html

.. |api-graph-builder| replace:: **Graph Builder**
.. _api-graph-builder: https://einarolafsson.github.io/spacr/api/spacr/qt/widgets/graph_builder/index.html

.. |api-artifacts| replace:: **आर्टिफ़ैक्ट**
.. _api-artifacts: https://einarolafsson.github.io/spacr/api/spacr/artifacts/index.html

.. |api-provenance| replace:: **रन की उत्पत्ति**
.. _api-provenance: https://einarolafsson.github.io/spacr/api/spacr/runctx/index.html


डेटा
----

संदर्भ डेटासेट
~~~~~~~~~~~~~~~~~~

- `पूर्ण माइक्रोस्कोप डेटासेट: BioStudies S-BIAD2135 <https://doi.org/10.6019/S-BIAD2135>`_
- `परीक्षण डेटासेट: Hugging Face toxo_mito <https://huggingface.co/datasets/einarolafsson/toxo_mito>`_
- `अनुक्रम डेटा: NCBI BioProject PRJNA1261935 <https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935>`_
- `सांख्यिकीय शक्ति विश्लेषण: spaCRPower <https://github.com/maomlab/spaCRPower>`_


योगदान और सहायता
------------------------

बग रिपोर्ट और विशिष्ट सुविधा अनुरोधों का `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ पर स्वागत है। किसी विफलता की रिपोर्ट में spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग और संबंधित लॉग का अंश शामिल करें। ``spacr-doctor`` इनमें से अधिकांश जानकारी आपके लिए इकट्ठी कर देता है।

लाइसेंस
~~~~~~~~~

वर्तमान विकास शाखा का स्रोत `PolyForm Noncommercial License 1.0.0 <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ के अधीन उपलब्ध है। वाणिज्यिक उपयोग के लिए कॉपीराइट धारक से अलग लाइसेंस लेना आवश्यक है। spaCR 1.4.9.9 तक जारी संस्करण उनके साथ दिए गए MIT License के अधीन उपलब्ध रहेंगे।

ट्यूटोरियल
~~~~~~~~~~

`इंटरैक्टिव spaCR ट्यूटोरियल लाइब्रेरी <https://einarolafsson.github.io/spacr/tutorials/>`_ में आठ भाषाओं में स्थापना और अनुप्रयोग के हर कार्यप्रवाह की वाचित और उपशीर्षक-युक्त मार्गदर्शिकाएँ हैं।

spaCR का संदर्भ
~~~~~~~~~~~~~~~

यदि spaCR आपके शोध में योगदान देता है, तो उद्धरण करें:

Olafsson EB, *et al.* एक पूल्ड, छवि-आधारित CRISPR स्क्रीन EAF1 की पहचान *T. gondii* द्वारा ESCRT तंत्र के अपहरण के नियामक के रूप में करती है।

`bioRxiv प्रीप्रिंट <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `सॉफ़्टवेयर अभिलेख <https://doi.org/10.5281/zenodo.21343317>`_
