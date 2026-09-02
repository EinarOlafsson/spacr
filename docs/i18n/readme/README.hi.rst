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

.. spacr-language-picker-begin

भाषाएँ: `🌐 हिन्दी ▾ <README.md>`_

.. spacr-language-picker-end

**CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।**

spaCR उच्च-सामग्री माइक्रोस्कोपी छवियों में एकल कोशिकाओं का विभाजन और मापन करता है, प्रति-वस्तु फीनोटाइप को अनुक्रमण से प्राप्त गाइड प्रचुरता के साथ एकीकृत करता है और अनुमान लगाता है कि कौन-से जीन फीनोटाइपिक परिवर्तनों से जुड़े हैं। प्लेट छवियों और FASTQ रीड से शुरू करके, यह प्रति-वस्तु मापन, प्रशिक्षित वर्गीकारक, प्रति-गाइड और प्रति-जीन प्रभाव अनुमान तथा प्राथमिकता के अनुसार क्रमित हिट सूची बनाता है।

segmentation, measurement, annotation और classification modules भी एक sequencing arm के बिना चलता है।

छवियों, मास्क, क्रॉपों, मापों, नोटों, भविष्यवाणियों, बारकोड और अच्छी तरह से पहचानकर्ताओं एक SQLite परियोजना में रहते हैं।

यह एक डेस्कटॉप एप्लिकेशन के रूप में चलता है या एक कार्यस्थल, सर्वर या क्लस्टर पर सिर के बिना।

हार्डवेयर समर्थन
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


spaCR इंस्टॉल करें
-----------------

डेस्कटॉप एप्लिकेशन
~~~~~~~~~~~~~~~~~~~

इंस्टॉलर अपने स्वयं के Python को जोड़ते हैं. कॉन्डा की आवश्यकता नहीं है.

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

PyPI से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~

PyPI रिलीज़ के लिए, Conda वातावरण के भीतर pip से spaCR इंस्टॉल करें। Python 3.12 में वैकल्पिक वैज्ञानिक पैकेजों की सबसे व्यापक उपलब्धता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR का समर्थन करता है Python **3.9 से 3.14** तक, Python 3.14.1 को छोड़कर, जो torchvision को छोड़ देता है. Linux सबसे भारी CUDA और ROCm कार्यप्रवाहों के लिए सिफारिश की जाती है; macOS और Windows का भी समर्थन किया जाता है, और दोनों का उपयोग उनके GPUs - macOS के माध्यम से धातु, जो एप्पल सिलिकॉन और एएमडी कार्ड को कवर करता है इंटेल मैक में, और Windows द्वारा CUDA या DirectML।

सर्वर, क्लस्टर या CI रनर पर Qt को छोड़ दें:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

वैकल्पिक एकीकरण अलग से स्थापित किए जाते हैं, उदाहरण के लिए ``spacr[zarr]``, ``spacr[omero]``,``spacr[napari]`` और ``spacr[czi,nd2,lif]``. पूर्ण अतिरिक्त और Python संस्करण संगतता तालिका में `स्थापना गाइड <../../source/installer_guide.rst>`_ देखें.

conda-forge से इंस्टॉलेशन
~~~~~~~~~~~~~~~~~~~~~~~~

आधिकारिक conda-forge पैकेज सक्रिय वातावरण में spaCR और उसकी डेस्कटॉप निर्भरताएँ इंस्टॉल करता है:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

स्रोत से स्थापित करें
~~~~~~~~~~~~~~~~~~~

संग्रहालय को क्लोन करें और इसे संपादित करने योग्य मोड में स्थापित करें, ताकि आपका कामकाजी कॉपी *is* स्थापित पैकेज और संपादन पुन: स्थापित किए बिना प्रभावी हो जाएं::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

डिफ़ॉल्ट शाखा ``nightly`` है. एक विशिष्ट रिलीज के लिए::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

बाद में परिवर्तनों को खींचने के लिए, क्लोन के अंदर से::

    git pull
    pip install -e .

दूसरी पंक्ति केवल तब आवश्यक है जब निर्भरता या प्रवेश बिंदु बदल जाते हैं; Python कोड इसके बिना उठाया जाता है. यदि एक कमांड अभी भी खींचने के बाद पुराने कोड चलाता है, तो ``spacr-doctor`` रिपोर्ट करता है कि ``spacr`` वास्तव में आपके रास्ते पर है, जो सामान्य कारण है.

स्रोत से स्थापित करें (प्रकाश)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

पूर्ण क्लोन: 427 एमबी. कोर क्लन: 76 एमबी।

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

स्काइप ``docs/``, ``tests/`` , Cellpose चेक पॉइंट, संग्रहीत आंकड़े और विस्तारित अनुवाद कैटलॉग. परिणाम एक सामान्य चेकअप है.

विकल्प: ``--dir``, ``--branch`` (डिफ़ॉल्ट ``main``), ``--with-tests``, [``--with-docs``,] ``--with-translations`` और [``--no-install``।

``packaging/source_install_excludes.txt`` प्रत्येक पारित मार्ग को सूचीबद्ध करता है।


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
-------------

मुख्य कार्यप्रवाह में छह मॉड्यूल हैं:

- **Mask** Cellpose से कोशिकाओं, नाभिकों, रोगजनकों और कोशिकांगों का विभाजन करता है।
- **Measure** आकृति-विज्ञान, तीव्रता, टेक्सचर, स्थानिक और सह-स्थानीकरण विशेषताओं के साथ ऑब्जेक्ट क्रॉप को SQLite में लिखता है।
- **Annotate** कीबोर्ड से संचालित ग्रिड में क्रॉप को लेबल करता है और सक्रिय-अधिगम कतारों का समर्थन करता है।
- **Classify** छवि- या मापन-आधारित मॉडल प्रशिक्षित करता है और प्रत्येक चेकपॉइंट के साथ होल्ड-आउट डेटा पर प्रदर्शन दर्ज करता है।
- **Map Barcodes** FASTQ रीड को वेल और gRNA से मैप करता है तथा प्रचुरता, टकराव और कवरेज का QC प्रदान करता है।
- **Regression** सतत, भिन्नात्मक और गणना प्रतिक्रियाओं के अनुकूल मॉडल परिवारों से गाइड, जीन, स्थिति और नियंत्रण प्रभावों का अनुमान लगाता है।

उसी प्रोजेक्ट में प्रायोगिक प्लेटें डिज़ाइन की जा सकती हैं, सांख्यिकीय पावर का अनुमान लगाया जा सकता है, बैच प्रभाव सुधारे जा सकते हैं, सेगमेंटेशन गुणवत्ता की जाँच की जा सकती है, परस्पर जुड़े प्लॉट और क्रॉप देखे जा सकते हैं, AnnData निर्यात किया जा सकता है, बाधित काम फिर शुरू किया जा सकता है और प्रत्येक परिणाम के लिए प्रयुक्त सेटिंग्स दर्ज की जा सकती हैं।

spaCR मॉड्यूल
-------------

.. spacr-workflow-begin

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|
| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|
| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|
| |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: ../../../spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Mask API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: ../../../spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Measure API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: ../../../spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Annotate API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: ../../../spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Classify API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: ../../../spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Map Barcodes API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: ../../../spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Regression API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: ../../../spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Import API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: ../../../spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Run Compare API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: ../../../spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Experiment Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: ../../../spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Power / Design API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: ../../../spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Dose–Response API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: ../../../spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: QC API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: ../../../spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Make Masks API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: ../../../spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Align & Stitch API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: ../../../spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Image UMAP API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: ../../../spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Gate Editor API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: ../../../spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Graph Builder API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: ../../../spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Plaque Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: ../../../spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Recruitment API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: ../../../spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Invasion Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: ../../../spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Replication Assay API खोलें
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

प्रत्येक मॉड्यूल spaCR जहाज है, जिस क्रम में होम स्क्रीन उन्हें सूचीबद्ध करता है: छह पाइपलाइन मॉडल पहले, फिर सब कुछ. उस मॉडल के API पृष्ठ को खोलने के लिए एक शीट चुनें.


Make Masks
~~~~~~~~~~

मेक मास्क के तहत दिखाई देता है **Tools** के लिए मैन्युअल को सही करने के लिए सेगमेंटेशन मास्क; इसके मस्टहेड Cellpose कार्यप्रवाहों को खोलता है. नौ टूल: **Brush**, **Erase**, [**Erases object**, (**Wand +**), **Wang −**, "**Draw**, '**Divide**, #**Zoom** और #**Recrop**. ड्राइंग एक भरा लेबल बनाता है एक बंद आउटलाइन से, विभाजित एक मिश्रित वस्तु को एक ड्रॉप लाइन के साथ, एक बहुतायत में एक वस्तु बदलता है अपने स्वयं के फ़ील्ड में.

Cellpose-SAM यहां चलता है सेल संभावना मानचित्र और मास्क के बगल में प्रवाह क्षेत्र दिखाता है. प्रत्येक टूल के लिए `फ़ीड गाइड <../../source/features.rst>`_ देखें.

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

जानवरों का मॉडल
~~~~~~~~~~~~~~~

spaCR प्रशिक्षित मॉडलों की एक सूची के साथ आता है और आवश्यकता पड़ने पर उन्हें डाउनलोड करता है। उन्हें देखने और स्थापित करने के लिए होम स्क्रीन से **Model Zoo** खोलें, या सेटिंग्स फ़ाइल में एक कुंजी दें -- ``pathogen_model: toxoplasma_pv_v1`` -- और पहली बार आवश्यकता होने पर मॉडल डाउनलोड होकर उसका चेकसम सत्यापित किया जाता है। प्रत्येक प्रकाशित प्रविष्टि में SHA-256 होता है; जिसमें यह नहीं है उसे स्थापित करने के बजाय अस्वीकार कर दिया जाता है, क्योंकि कटे हुए या बदले गए चेकपॉइंट को असली से अलग नहीं पहचाना जा सकता।

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Key
     - Trained on
     - Measured performance and limits
   * - ``toxoplasma_pv_v1``
       (cpsam_v2_toxo_r2)
     - Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. 115 pairs (104 train / 11 test), 100 epochs, base cpsam_v2
     - F1 0.867 at IoU 0.5 against 0.713 for stock cpsam; AJI 0.808 against 0.426; accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
   * - ``toxoplasma_plaque_v1``
       (cpsam_plaque_r3)
     - Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set
     - F1 0.856 in-domain and 0.834 on the literature set, against 0.718 / 0.755 for round 1; round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
   * - ``toxoplasma_well_detector_v1``
       (yolo_welldetect_v3.pt)
     - Whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987; locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

.. spacr-model-zoo-end

उपरोक्त आंकड़े प्रकाशन में मापा गया है, और सीमाओं को उनके साथ व्यक्त किया गया है: एक मॉडल उस पर मापा काम के लिए उपयोगी है, न कि प्रत्येक कार्य के लिए. ``toxoplasma_well_detector_v1`` और ``toxoplasma_plaque_v1`` एक पाइपलाइन के दो आधे हैं - डिटेक्टर बर्तनों को ढूंढता है, सेगमेंटर उनके अंदर प्लेटों को पाता है और अच्छी व्यास यह है कि माइक्रोस्कोप के बीच क्षेत्रों की तुलना की जा सकती है।

मॉडलों को उनके लेखक के स्वयं के Hugging Face खाते पर होस्ट किया जाता है, इसलिए एक में योगदान करने का मतलब किसी और के लेखन एक्सेस को सौंपना नहीं है. ``spacr.model_zoo`` का ``publish_model`` अपलोड करता है और जोड़ने के लिए कैटलॉग पंक्ति प्रिंट करता है.


प्रदर्शन का निदान
----------------------

हार्डवेयर रिपोर्ट बनाएँ और उसे प्रदर्शन-संबंधी इश्यू के साथ संलग्न करें::

    python tools/spacr_hardware_report.py

``~/.spacr/reports`` पर बचत करें और मार्ग प्रिंट करें. ``--quick`` लंबे संदर्भ संकेतों को स्काइप करता है; ``--out PATH`` स्थान निर्धारित करता है.

कोई परियोजना डेटा नहीं पढ़ता. टाइम आयात, संख्यात्मक पुस्तकालय, खिड़की निर्माण और एनीमेशन. रिपोर्ट प्रोसेसर-आर्किटेक्चर इमुलेशन (एक x86_64 Python Apple सिलिकॉन पर निर्माण) और NumPy के BLAS कार्यान्वयन।

कमांड लाइन संदर्भ
----------------------

नीचे दिए गए प्रत्येक कमांड को ``pip install spacr`` द्वारा स्थापित किया जाता है. उनमें से सभी ``--help`` को स्वीकार करते हैं.

आवेदन शुरू करने के लिए
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` मॉडल सेटअप स्क्रीनिंग को स्काइप करता है, जो अन्यथा एक अप्रत्याशित नौकरी को अवरुद्ध करेगा।

``spacr-qt`` और ``spacr-nightly`` ``spacr`` के सहयोगी हैं।

जब spaCR शुरू नहीं होता है
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` प्रत्येक चेक पर एक पंक्ति प्रिंट करता है, जिसमें प्रत्येक विफलता के लिए एक कमांड चलाया जाता है. यह यह भी रिपोर्ट करता है कि ``spacr`` मार्ग पर है, जो कि एक पुराने संपादित करने योग्य स्थापना की छाया है.

``safespacr`` प्रत्येक प्राथमिकता को अपने डिफ़ॉल्ट के रूप में पढ़ता है और पृष्ठभूमि, एनीमेशन, वर्बस लॉगिंग और प्री-लोड को मजबूर करता है. जब एक सहेजा गया प्राथमिकता लॉन्च को तोड़ देती है तो इसका उपयोग करें. यह स्थायी रूप से कुछ भी नहीं बदलता है.

मॉड्यूल बेहोश रूप से चलाएं
~~~~~~~~~~~~~~~~~~~~~~~~~~

कोई Qt, कोई डिस्प्ले नहीं - क्लस्टर, सर्वर और सीआई के लिए।

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` उसी सेटिंग्स को पढ़ता है जो चलता है और रिपोर्ट करता है कि क्या खो रहा है, विरोधाभासी है या कुछ भी नहीं बता रहा है।

``spacr-run --list`` केवल एक शीर्ष के बिना प्रवेश बिंदु के साथ मॉड्यूल दिखाता है; नोटिस, चिकित्सा और अन्वेषण इंटरैक्टिव हैं और अनदेखा किया गया है।

बाद में एक दौड़ की जांच करें
~~~~~~~~~~~~~~~~~~~~~~~~~~~

प्रत्येक रन ``~/.spacr/runs`` में रिकॉर्ड किया जाता है, जिसमें इसकी सेटिंग्स, हैश किए गए इनपुट, आउटपुट्स, चेतावनी, संस्करण और बीज होते हैं।

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

डेटा की जांच और स्थापना
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

परिवेश
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

रूटिंग लॉग ``~/.spacr/logs/spacr.log`` में लिखे जाते हैं. उस फ़ाइल को एक बग रिपोर्ट में जोड़ें.


योगदान और सहायता
------------------------

बग रिपोर्ट और स्पष्ट रूप से सीमित फ़ीचर अनुरोध `GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_ के माध्यम से भेजें। किसी विफलता की रिपोर्ट करते समय spaCR संस्करण, ऑपरेटिंग सिस्टम, Python संस्करण, मॉड्यूल सेटिंग्स और संबंधित लॉग अंश शामिल करें। ``spacr-doctor`` इनमें से अधिकांश जानकारी एकत्र करता है; प्रदर्शन संबंधी समस्या की रिपोर्ट करते समय हार्डवेयर रिपोर्ट भी शामिल करें।

लाइसेंस
~~~~~~~~~

spaCR is released under the `BSD 3 क्लास लाइसेंस <https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

यदि spaCR प्रकाशित कार्य में योगदान देता है, तो एक उद्धरण का मूल्यांकन किया जाता है और लाइसेंस की शर्त नहीं है - नीचे `Citing spaCR`_ देखें।

ट्यूटोरियल
~~~~~~~~~

`इंटरैक्टिव spaCR ट्यूटोरियल लाइब्रेरी <https://einarolafsson.github.io/spacr/tutorials/>`_ में स्थापना और प्रत्येक ऐप कार्यप्रवाह के वर्णित तथा कैप्शनयुक्त मार्गदर्शन हैं: आठ भाषाओं में 50 आवाज़ों के साथ 73 पाठ।

spaCR का संदर्भ
~~~~~~~~~~~~~~

यदि spaCR आपके शोध में योगदान देता है, तो इसका उद्धरण दें:

Olafsson EB, *et al.* एक संयोजित छवि-आधारित CRISPR स्क्रीनिंग EAF1 को *T. gondii* के रूप में पहचानती है ESCRT उप-विवाद का मॉड्यूलर।

`BioRxiv प्रीप्रिंट <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ · `सॉफ्टवेयर संग्रह <https://doi.org/10.5281/zenodo.21343316>`_

आभार
~~~~~~~~~~~~~~~

spaCR NumPy, pandas, scikit-image, scikit-learn, Cellpose, PyTorch और Qt सहित मुक्त वैज्ञानिक सॉफ़्टवेयर पर आधारित है। बहुभाषी दस्तावेज़ और इंटरफ़ेस कैटलॉग तैयार करने में उपयोग किए गए मॉडल के लिए `अनुवाद मॉडल श्रेय <../TRANSLATION_MODELS.md>`_ देखें।
