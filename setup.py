import subprocess
from setuptools import setup, find_packages

# Ensure you have read the README.rst content into a variable, e.g., `long_description`
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dependencies = [
    'numpy>=1.26.4,<2.0',
    'pandas>=2.2.1,<3.0',
    'scipy>=1.12.0,<2.0',
    'cellpose>=3.0.6,<4.0',
    'scikit-image>=0.22.0,<1.0',
    'scikit-learn>=1.4.1,<2.0',
    'scikit-posthocs>=0.10.0, <0.20',
    'mahotas>=1.4.13,<2.0',
    'btrack>=0.6.5,<1.0',
    'trackpy>=0.6.2,<1.0',
    'statsmodels>=0.14.1,<1.0',
    'shap>=0.45.0,<1.0',
    'torch>=2.0,<3.0',
    'torchvision>=0.1,<1.0',
    'torch-geometric>=2.5,<3.0',
    'torchcam>=0.4.0,<1.0',
    'transformers>=4.45.2, <5.0',
    'segmentation_models_pytorch>=0.3.3',
    'monai>=1.3.0',
    'captum>=0.7.0, <1.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'matplotlib_venn>=1.1,<2.0',
    'adjustText>=1.2.0,<2.0',
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    'opencv-python-headless>=4.9.0.80,<5.0',
    'pillow>=10.2.0,<11.0',
    'tifffile>=2023.4.12',
    'nd2reader>=3.3.0, <4.0',
    'czifile',
    'readlif',
    'imageio>=2.34.0,<3.0',
    'pingouin>=0.5.5,<1.0',
    'umap-learn>=0.5.6,<1.0',
    'ttkthemes>=3.2.2,<4.0',
    'xgboost>=2.0.3,<3.0',
    'PyWavelets>=1.6.0,<2.0',
    'ttf_opensans>=2020.10.30',
    'customtkinter>=5.2.2,<6.0', 
    'biopython>=1.80,<2.0',
    'lxml>=5.1.0,<6.0',
    'psutil>=5.9.8, <6.0',
    'gputil>=1.4.0, <2.0', 
    'gpustat>=1.1.1,<2.0',
    'pyautogui>=0.9.54,<1.0',
    'tables>=3.8.0,<4.0',
    'rapidfuzz>=3.9, <4.0',
    'keyring>=15.1, <16.0',
    'screeninfo>=0.8.1,<1.0',
    'fastremap>=1.14.1',
    'pytz>=2023.3.post1',
    'tqdm>=4.65.0',
    'wandb>=0.16.2',
    'openai>=1.50.2, <2.0',
    'gdown',
    'IPython>=8.18.1,<9.0',
    'ipykernel',
    'ipywidgets>=8.1.2,<9.0',
    'brokenaxes>=0.6.2,<1.0',
    'huggingface-hub>=0.24.0,<0.25'
]

setup(
    name="spacr",
    version="0.4.60",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="Spatial phenotype analysis of crisp screens (SpaCr)",
    long_description=long_description,
    long_description_content_type='text/x-rst', 
    url="https://github.com/EinarOlafsson/spacr",
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={'spacr': ['resources/data/*', 'resources/models/cp', 'resources/icons/*', 'resources/font/**/*', 'resources/MEDIAR/**/*', 'resources/MEDIAR_weights/*', 'resources/images/*'],},
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'mask=spacr.app_mask:start_mask_app',
            'measure=spacr.app_measure:start_measure_app',
            'make_masks=spacr.app_make_masks:gui_make_masks',
            'annotate=spacr.app_annotate:gui_annotate',
            'classify=spacr.app_classify:start_classify_app',
            'sim=spacr.app_sim:gui_sim',
            'spacr=spacr.gui:gui_app',
        ],
    },
    extras_require={
        'dev': ['pytest>=3.9,<3.11'],
        'headless': ['opencv-python-headless'],
        'full': ['opencv-python'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)

deps = ['pyqtgraph>=0.13.7,<0.14',
        'pyqt6>=6.7.1,<6.8',
        'pyqt6.sip',
        'qtpy>=2.4.1,<2.5',
        'superqt>=0.6.7,<0.7',
        'pyqtgraph',
        'pyqt6',
        'pyqt6.sip',
        'qtpy',
        'superqt']

for dep in deps:
    try:
        subprocess.run(['pip', 'install', dep], check=True)
    except subprocess.CalledProcessError:
        pass
