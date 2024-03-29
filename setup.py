from setuptools import setup, find_packages

# Ensure you have read the README.md content into a variable, e.g., `long_description`
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dependencies = [
    'torch>=2.2.1',
    'torchvision>=0.17.1',
    'torch-geometric>=2.5.1',
    'numpy>=1.26.4',
    'pandas>=2.2.1',
    'statsmodels>=0.14.1',
    'scikit-image>=0.22.0',
    'scikit-learn>=1.4.1',
    'seaborn>=0.13.2',
    'matplotlib>=3.8.3',
    'pillow>=10.2.0',
    'imageio>=2.34.0',
    'scipy>=1.12.0',
    'ipywidgets>=8.1.2',
    'mahotas>=1.4.13',
    'btrack>=0.6.5',
    'trackpy>=0.6.2',
    'cellpose>=3.0.6',
    'IPython>=8.18.1',
    'opencv-python-headless>=4.9.0.80',
    'umap>=0.1.1',
    'ttkthemes>=3.2.2',
    'lxml>=5.1.0'
]

setup(
    name="spacr",
    version="0.0.2",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="Spatial phenotype analysis of crisp screens (SpaCr)",
    long_description=long_description,
    url="https://github.com/EinarOlafsson/spacr",
    packages=find_packages(exclude=["tests.*", "tests"]),
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'gui_mask=spacr.gui_mask_app:gui_mask',
            'gui_measure=spacr.gui_measure_app:gui_measure',
            'gui_make_masks=spacr.mask_app:gui_make_masks',
            'gui_annotation=spacr.annotate_app:gui_annotation',
            'gui_classify=spacr.gui_classify_app:gui_classify',
            'gui_sim=spacr.gui_sim_app:gui_sim',
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