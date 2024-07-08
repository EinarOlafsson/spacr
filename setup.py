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

cuda_version = get_cuda_version()

if cuda_version:
    dgl_dependency = f'dgl-cu{cuda_version}==0.9.1'  # Specify the version of DGL compatible with your setup
else:
    dgl_dependency = 'dgl==0.9.1'  # Fallback to CPU version if no CUDA is detected

# Ensure you have read the README.rst content into a variable, e.g., `long_description`
with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dependencies = [
    dgl_dependency,
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
    'lxml>=5.1.0,<6.0'
]

setup(
    name="spacr",
    version="0.0.82",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="Spatial phenotype analysis of crisp screens (SpaCr)",
    long_description=long_description,
    url="https://github.com/EinarOlafsson/spacr",
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={'spacr': ['models/cp/*'],},
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'mask=spacr.gui_mask_app:gui_mask',
            'measure=spacr.gui_measure_app:gui_measure',
            'make_masks=spacr.mask_app:gui_make_masks',
            'annotate=spacr.annotate_app:gui_annotation',
            'classify=spacr.gui_classify_app:gui_classify',
            'sim=spacr.gui_sim_app:gui_sim',
            'gui=spacr.gui:gui_app',
            'gui2=spacr.gui_2:gui_app',
        ],
    },
    extras_require={
        'dev': ['pytest>=3.9'],
        'headless': ['opencv-python-headless'],
        'full': ['opencv-python'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)