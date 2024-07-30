from setuptools import setup, find_packages
import subprocess

# Function to determine the CUDA version
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode('utf-8')
        if 'release' in output:
            return output.split('release ')[1].split(',')[0].replace('.', '')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

# Ensure you have read the README.rst content into a variable, e.g., `long_description`
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dependencies = [
    'torch>=2.2.1,<3.0',
    'torchvision>=0.17.1,<1.0',
    'torch-geometric>=2.5.1,<3.0',
    'numpy>=1.26.4,<2.0',
    'pandas>=2.2.1,<3.0',
    'statsmodels>=0.14.1,<1.0',
    'scikit-image>=0.22.0,<1.0',
    'scikit-learn>=1.4.1,<2.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'shap>=0.45.0,<1.0',
    'pillow>=10.2.0,<11.0',
    'imageio>=2.34.0,<3.0',
    'scipy>=1.12.0,<2.0',
    'ipywidgets>=8.1.2,<9.0',
    'mahotas>=1.4.13,<2.0',
    'btrack>=0.6.5,<1.0',
    'trackpy>=0.6.2,<1.0',
    'cellpose>=3.0.6,<4.0',
    'IPython>=8.18.1,<9.0',
    'opencv-python-headless>=4.9.0.80,<5.0',
    'umap-learn>=0.5.6,<1.0',
    'ttkthemes>=3.2.2,<4.0',
    'xgboost>=2.0.3,<3.0',
    'PyWavelets>=1.6.0,<2.0',
    'torchcam>=0.4.0,<1.0',
    'ttf_opensans>=2020.10.30',
    'customtkinter>=5.2.2,<6.0', 
    'biopython>=1.80,<2.0',
    'lxml>=5.1.0,<6.0',
    'huggingface-hub>=0.24.0,<0.25'
]

setup(
    name="spacr",
    version="0.2.21",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="Spatial phenotype analysis of crisp screens (SpaCr)",
    long_description=long_description,
    url="https://github.com/EinarOlafsson/spacr",
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={'spacr': ['resources/models/cp/*', 'resources/icons/*'],},
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

cuda_version = get_cuda_version()

if cuda_version:
    dgl = f'dgl-cu{cuda_version}==0.9.1'
else:
    dgl = 'dgl==0.9.1'  # Fallback to CPU version if no CUDA is detected
try:
    subprocess.run(['pip', 'install', dgl], check=True)
except subprocess.CalledProcessError:
    subprocess.run(['pip', 'install', 'dgl'], check=True)

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