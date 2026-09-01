|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
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
   :alt: BSD 3-Clause लाइसेंस
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: नवीनतम इंस्टॉलर
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge संस्करण

.. image:: ../../../spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

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

**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**

spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रति-वस्तु फीनोटाइप को अनुक्रमण से प्राप्त गाइड प्रचुरता के साथ एकीकृत करता है और अनुमान लगाता है कि कौन-से जीन फीनोटाइपिक परिवर्तनों से जुड़े हैं। प्लेट छवियों और FASTQ रीड से शुरू करके, यह प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव अनुमान तथा प्राथमिकता के अनुसार क्रमित हिट सूची बनाता है।

छवि-आधारित पूल्ड CRISPR स्क्रीनिंग के लिए spaCR छवि विभाजन से हिट प्राथमिकता तक का कार्यप्रवाह प्रदान करता है। अनुक्रमण-आधारित स्क्रीनिंग के बिना उच्च-सामग्री माइक्रोस्कोपी अध्ययनों में विभाजन, मापन, एनोटेशन और वर्गीकरण मॉड्यूल स्वतंत्र रूप से उपयोग किए जा सकते हैं।

छवियाँ, मास्क, इमेज क्रॉप, मापन, एनोटेशन, पूर्वानुमान, बारकोड और वेल पहचानकर्ता एक ही SQLite प्रोजेक्ट में रहते हैं, इसलिए किसी परिणाम के मान को उसके स्रोत ऑब्जेक्ट तक वापस खोजा जा सकता है।

spaCR को डेस्कटॉप एप्लिकेशन के रूप में या वर्कस्टेशन, सर्वर अथवा क्लस्टर पर बिना ग्राफ़िकल इंटरफ़ेस के चलाएँ। दोनों तरीके समान मॉड्यूल चलाते हैं और समर्थित मॉड्यूल में CUDA अपने आप उपयोग होता है।


कार्यप्रवाह का अवलोकन
--------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Mask API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Measure API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Annotate API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Classify API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Map Barcodes API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Regression API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: ../../../spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**डेटा**

|App_foreign|\ |App_run_compare|\ |App_experiment_design|\ |App_power|\ |App_dose_response|\ |App_qc_dashboard|

**Tools**

|App_make_masks|\ |App_align|\ |App_umap|\ |App_gate_editor|\ |App_graph_builder|

**परख**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

.. |App_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 15.466%
   :alt: Import API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 15.466%
   :alt: Run Compare API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 15.466%
   :alt: Experiment Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 15.466%
   :alt: Power / Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 15.466%
   :alt: Dose–Response API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |App_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 15.466%
   :alt: QC API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 15.466%
   :alt: Make Masks API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 15.466%
   :alt: Align & Stitch API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 15.466%
   :alt: Image UMAP API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 15.466%
   :alt: Gate Editor API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 15.466%
   :alt: Graph Builder API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 15.466%
   :alt: Plaque Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 15.466%
   :alt: Recruitment API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 15.466%
   :alt: Invasion Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 15.466%
   :alt: Replication Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle

.. spacr-workflow-end

किसी कार्यप्रवाह मॉड्यूल का API पृष्ठ खोलने के लिए उसे चुनें। ग्रिड में अन्य सभी ऐप उसी श्रेणी और क्रम में हैं जैसा spaCR की होम स्क्रीन पर है।


spaCR इंस्टॉल करें
-----------------

डेस्कटॉप एप्लिकेशन
~~~~~~~~~~~~~~~~~~~

डेस्कटॉप इंस्टॉलर में एक निजी Python वातावरण शामिल है, इसलिए कॉन्डा और मौजूदा Python स्थापना की आवश्यकता नहीं है।

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: ../../../spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Windows 10/11 के लिए spaCR 1.5.0.4 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: ../../../spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: macOS 11+ (Intel और Apple Silicon) के लिए spaCR 1.5.0.4 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: ../../../spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: 64-बिट Linux के लिए spaCR 1.5.0.4 डाउनलोड करें
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: ../../../spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: spaCR के पुराने इंस्टॉलर
   :target: ../../source/installers.rst

.. spacr-installer-links-end

पहले तीन आइकन वर्तमान रिलीज डाउनलोड करते हैं. spaCR आईकन पूरे इंस्टॉलर संग्रहालय को खोलता है. इंस्टॉलर लिंक और संस्करण फ़ाइल नाम जारी कार्यप्रवाह द्वारा अद्यतन किए जाते हैं; पिछले इंस्टोलर एक ही रिलीज़ संग्रहीत में रहते हैं.

Linux पर डाउनलोड की गई फ़ाइल को निष्पादन योग्य बनाएँ और चलाएँ:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

macOS पर, ``.pkg`` खोलें. वर्तमान बीटा नोटिस नहीं किया गया है; यदि Gatekeeper इसे ब्लॉक करता है, तो **सिस्टम सेटिंग्स → गोपनीयता और सुरक्षा → किसी भी तरह से खोलें** का चयन करें.

अद्यतन, अनइंस्टॉल, ऑफ़लाइन और समस्या हल करने के लिए निर्देशों के लिए `इंस्टॉलर गाइड <../../source/installer_guide.rst>`_ देखें।

conda-forge से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~~~~~~~~

आधिकारिक conda-forge पैकेज सक्रिय वातावरण में spaCR और उसकी डेस्कटॉप निर्भरताएँ इंस्टॉल करता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

PyPI से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~

PyPI रिलीज़ के लिए, Conda वातावरण के भीतर pip से spaCR इंस्टॉल करें। Python 3.12 में वैकल्पिक वैज्ञानिक पैकेजों की सबसे व्यापक उपलब्धता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR Python **3.9 से 3.14** तक समर्थित है, लेकिन Python 3.14.1 समर्थित नहीं है क्योंकि torchvision उसे बाहर रखता है। CUDA कार्यप्रवाहों के लिए Linux अनुशंसित है; macOS और Windows भी समर्थित हैं।

सर्वर, क्लस्टर या CI रनर पर Qt को छोड़ दें:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

वैकल्पिक एकीकरण अलग से स्थापित किए जाते हैं, उदाहरण के लिए ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` और ``spacr[czi,nd2,lif]``. पूर्ण अतिरिक्त और Python संस्करण संगतता तालिका में `स्थापना गाइड <../../source/installer_guide.rst>`_ देखें.

कमांड-लाइन प्रवेश बिंदु
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

समस्या निवारण के समय ``SPACR_LOG_LEVEL=DEBUG`` सेट करें। रोटेटिंग लॉग ``~/.spacr/logs/spacr.log`` में लिखे जाते हैं।

``spacr-run --list`` उन मॉड्यूलों की सूची दिखाता है जिनके पास ग्राफ़िकल इंटरफ़ेस के बिना चलाने के लिए कमांड-लाइन प्रवेश बिंदु हैं। केवल GUI में उपलब्ध एनोटेशन, क्यूरेशन, तुलना और अन्वेषण मॉड्यूल इस सूची में शामिल नहीं होते।


मुख्य वर्कफ़्लो
---------------

मुख्य कार्यप्रवाह में छह मॉड्यूल हैं:

- **Mask** Cellpose से कोशिकाओं, नाभिकों, रोगजनकों और कोशिकांगों का विभाजन करता है।
- **Measure** आकृति-विज्ञान, तीव्रता, टेक्सचर, स्थानिक और सह-स्थानीकरण विशेषताओं के साथ ऑब्जेक्ट क्रॉप को SQLite में लिखता है।
- **Annotate** कीबोर्ड से संचालित ग्रिड में क्रॉप को लेबल करता है और सक्रिय-अधिगम कतारों का समर्थन करता है।
- **Classify** छवि- या मापन-आधारित मॉडल प्रशिक्षित करता है और प्रत्येक चेकपॉइंट के साथ होल्ड-आउट डेटा पर प्रदर्शन दर्ज करता है।
- **Map Barcodes** FASTQ रीड को वेल और gRNA से मैप करता है तथा प्रचुरता, टकराव और कवरेज का QC प्रदान करता है।
- **Regression** सतत, भिन्नात्मक और गणना प्रतिक्रियाओं के अनुकूल मॉडल परिवारों से गाइड, जीन, स्थिति और नियंत्रण प्रभावों का अनुमान लगाता है।

उसी प्रोजेक्ट में प्रायोगिक प्लेटें डिज़ाइन की जा सकती हैं, सांख्यिकीय पावर का अनुमान लगाया जा सकता है, बैच प्रभाव सुधारे जा सकते हैं, सेगमेंटेशन गुणवत्ता की जाँच की जा सकती है, परस्पर जुड़े प्लॉट और क्रॉप देखे जा सकते हैं, AnnData निर्यात किया जा सकता है, बाधित काम फिर शुरू किया जा सकता है और प्रत्येक परिणाम के लिए प्रयुक्त सेटिंग्स दर्ज की जा सकती हैं।

होस्ट स्क्रीन से उपलब्ध मॉड्यूल
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

बीस मॉड्यूल अलग-अलग Home टाइलों के रूप में दिखने के बजाय उनसे संबंधित होस्ट स्क्रीन में समाहित हैं। प्रत्येक मॉड्यूल अपनी होस्ट स्क्रीन की शीर्ष पट्टी से खुलता है और सक्रिय प्रोजेक्ट का उपयोग करता है। Mask, Measure, Annotate, Classify, Map Barcodes, Regression, Image UMAP और Make Masks में ये समाहित मॉड्यूल उपलब्ध हैं। इनके सहायता और API दस्तावेज़ उपलब्ध रहते हैं, और पाइपलाइन प्रवेश बिंदु वाले मॉड्यूल अब भी ग्राफ़िकल इंटरफ़ेस के बिना चलाए जा सकते हैं। `फ़ीचर गाइड <../../source/features.rst>`_ में प्रत्येक समाहित मॉड्यूल और उसकी होस्ट स्क्रीन सूचीबद्ध है।

Make Masks
~~~~~~~~~~

Make Masks **Data** के अंतर्गत उपलब्ध है और सेगमेंटेशन मास्क को मैन्युअल रूप से सुधारने की सुविधा देता है। इसकी शीर्ष पट्टी से Cellpose कार्यप्रवाह भी खोले जा सकते हैं। कैनवस में नौ टूल हैं: **Brush**, **Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**, **Zoom** और **Recrop**। Draw स्वतंत्र रूप से खींची गई बंद रूपरेखा से एक भरा हुआ लेबल बनाता है। Divide उपयोगकर्ता द्वारा निर्धारित रेखा के सहारे मर्ज हुए ऑब्जेक्ट को अलग करता है और अन्य सभी ऑब्जेक्ट लेबल सुरक्षित रखता है।

Recrop क्यूरेशन के लिए तैयार की गई, अनेक ऑब्जेक्ट वाली छवि से एकल-ऑब्जेक्ट फ़ील्ड निकालता है। किसी ऑब्जेक्ट के चारों ओर बाउंडिंग बॉक्स बनाने पर उससे संबंधित छवि और मास्क क्षेत्र नए फ़ील्ड के रूप में लिखे जाते हैं, वह फ़ील्ड वर्तमान फ़ील्ड के बाद कतार में रखा जाता है और मूल बहु-ऑब्जेक्ट फ़ील्ड क्यूरेशन कतार से हटा दिया जाता है। Recrop लेबल पिक्सेल संपादित करने के बजाय सक्रिय फ़ील्ड बदलता है।

Make Masks से Cellpose-SAM चलाने पर मास्क के पास दो मध्यवर्ती आउटपुट दिखते हैं: **कोशिका-प्रायिकता मानचित्र** और **प्रवाह क्षेत्र**। मास्क प्रायिकता मानचित्र पर लगाए गए थ्रेशहोल्ड से निर्धारित होता है, और प्रवाह-संगति जाँच उन ऑब्जेक्ट को अस्वीकार कर सकती है जिनके व्युत्पन्न प्रवाह अनुमानित क्षेत्र से भिन्न हों। किसी गलत या अधूरे मास्क का मूल्यांकन करते समय कम कोशिका प्रायिकता और असंगत प्रवाह में अंतर करने के लिए इन आउटपुट की जाँच करें।

**अन्य संसाधन**

- `इंटरैक्टिव ट्यूटोरियल <https://einarolafsson.github.io/spacr/tutorials/>`_ — स्थापना से हिट जांच के माध्यम से 73 निर्देशित कार्यप्रवाह।
- `Python API त्वरित प्रारंभ <../../source/python_api.rst>`_ - स्क्रिप्ट, नोटबुक या एक क्लस्टर से पाइपलाइन चलाएं और वैध करें।
- `सुविधा गाइड <../../source/features.rst>`_ — क्षमताओं, परिपक्वता और वैकल्पिक एकीकरण।
- `शुद्ध API संदर्भ <https://einarolafsson.github.io/spacr/api/index.html>`_ - कार्य के आधार पर समर्थित प्रवेश बिंदु, पूर्ण मॉड्यूल संदर्भ एक स्तर गहरा है।
- `भाषा और अनुवाद गाइड <../../source/localization.rst>`_ — इंटरफ़ेस भाषाएं, संदर्भ सहायता और वैज्ञानिक-आउटपुट नीति।

भाषा और अनुवाद
~~~~~~~~~~~~~~~~~~~~~~

इंटरफ़ेस नेविगेशन और प्राथमिकताओं में दस भाषाओं का समर्थन करता है। AI और LIVE नियंत्रण, मॉड्यूल विवरण और समीक्षित संदर्भ सहायता भी अनुवादित हैं। पुनः आरंभ किए बिना **spaCR → प्राथमिकताएँ → भाषा** में भाषा बदलें। लॉग, पथ, डेटाबेस मान और मापन कभी अनुवादित नहीं होते; वैज्ञानिक आउटपुट मानक अंग्रेज़ी में रहता है। `संदर्भ-सहायता नीति <../../source/localization.rst#contextual-help>`_ देखें।

एनिमेटेड सेटिंग मार्गदर्शन
~~~~~~~~~~~~~~~~~~~~~~~~~

दृश्य व्याख्या वाली सेटिंग के टूलटिप में **Animation** नियंत्रण मिलता है। `सेटिंग एनिमेशन गैलरी <https://einarolafsson.github.io/spacr/setting_animations.html>`_ या `सेटिंग एनिमेशन रजिस्ट्री <https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_ देखें।

डेटा
----

संदर्भ डेटासेट
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: ../../../spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: BioStudies माइक्रोस्कोपी डेटासेट खोलें
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: ../../../spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Hugging Face परीक्षण डेटासेट खोलें
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: ../../../spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: NCBI अनुक्रमण डेटासेट खोलें
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: ../../../spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: spaCRPower खोलें
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: ../../../spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: bioRxiv प्रीप्रिंट खोलें
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

प्रदर्शन का निदान
----------------------

हार्डवेयर रिपोर्ट बनाएँ और उसे प्रदर्शन-संबंधी इश्यू के साथ संलग्न करें::

    python tools/spacr_hardware_report.py

यह कमांड रिपोर्ट को प्रदर्शित करता है और उसकी एक प्रति ``~/.spacr/reports`` में सहेजता है; अंतिम पंक्ति सहेजे गए पथ की पहचान करती है। ``--quick`` लंबे बेंचमार्क छोड़ देता है, और ``--out PATH`` किसी अन्य आउटपुट स्थान को चुनता है।

रिपोर्ट कोई प्रोजेक्ट नहीं खोलती और प्रोजेक्ट डेटा नहीं पढ़ती। यह इम्पोर्ट और संख्यात्मक लाइब्रेरी की टाइमिंग, डिस्प्ले स्केलिंग, सक्रिय प्राथमिकताएँ, मुख्य विंडो और मॉड्यूल स्क्रीन का निर्माण तथा ऐनिमेशन प्रदर्शन रिकॉर्ड करती है। यह केवल रिपोर्ट फ़ाइल को आउटपुट के रूप में बनाती है।

यह प्रोसेसर आर्किटेक्चर के अनुकरण—जैसे Apple Silicon पर x86_64 Python बिल्ड—और NumPy द्वारा उपयोग किए जाने वाले BLAS कार्यान्वयन की भी पहचान करती है। इनमें से कोई भी प्रदर्शन को काफी प्रभावित कर सकता है।

योगदान और सहायता
------------------------

बग रिपोर्ट और स्पष्ट रूप से सीमित फ़ीचर अनुरोध `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ के माध्यम से भेजें। किसी विफलता की रिपोर्ट करते समय spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग्स और संबंधित लॉग अंश शामिल करें। ``spacr-doctor`` इनमें से अधिकांश जानकारी एकत्र करता है; प्रदर्शन संबंधी समस्या की रिपोर्ट करते समय हार्डवेयर रिपोर्ट भी शामिल करें।

लाइसेंस
~~~~~~~~~

spaCR `BSD 3-Clause License <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_ के अंतर्गत ओपन सोर्स है — वही लाइसेंस जो CellProfiler, napari और Cellpose उपयोग करते हैं। इसे किसी भी उद्देश्य के लिए, व्यावसायिक रूप से भी, उपयोग किया जा सकता है। संस्करण 1.5.0.0 से 1.5.0.4 तक PolyForm Noncommercial License 1.0.0 के अंतर्गत और 1.4.9.9 तक के संस्करण MIT License के अंतर्गत जारी हुए थे; वे संस्करण उनके साथ जारी लाइसेंस के अंतर्गत उपलब्ध रहेंगे।

ट्यूटोरियल
~~~~~~~~~

`इंटरैक्टिव spaCR ट्यूटोरियल लाइब्रेरी <https://einarolafsson.github.io/spacr/tutorials/>`_ में स्थापना और प्रत्येक ऐप कार्यप्रवाह के वर्णित तथा कैप्शनयुक्त मार्गदर्शन हैं: आठ भाषाओं में 50 आवाज़ों के साथ 73 पाठ।

spaCR का संदर्भ
~~~~~~~~~~~~~~

यदि spaCR आपके शोध में योगदान देता है, तो इसका उद्धरण दें:

Olafsson EB, *et al.* एक संयोजित छवि-आधारित CRISPR स्क्रीनिंग EAF1 को *T. gondii* के रूप में पहचानती है ESCRT उप-विवाद का मॉड्यूलर।

`BioRxiv प्रीप्रिंट <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `सॉफ्टवेयर संग्रह <https://doi.org/10.5281/zenodo.21343316>`_ के लिए

आभार
~~~~~~~~~~~~~~~

spaCR NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch और Qt सहित मुक्त वैज्ञानिक सॉफ़्टवेयर पर आधारित है। बहुभाषी दस्तावेज़ और इंटरफ़ेस कैटलॉग तैयार करने में उपयोग किए गए मॉडल के लिए `अनुवाद मॉडल श्रेय <../TRANSLATION_MODELS.md>`_ देखें।
