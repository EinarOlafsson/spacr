from setuptools import setup, find_packages

# Ensure you have read the README.md content into a variable, e.g., `long_description`
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

dependencies = [
    'torch',
    'torchvision',
    'torch-geometric',
    'numpy',
    'pandas',
    'statsmodels',
    'scikit-image',
    'scikit-learn',
    'seaborn',
    'matplotlib',
    'pillow',
    'imageio',
    'scipy',
    'ipywidgets',
    'mahotas',
    'btrack',
    'trackpy',
    'cellpose',
    'IPython',
    'opencv-python-headless',
    'umap',
    'ttkthemes',
    'lxml'
]

setup(
    name="spacr",
    version="0.0.17",
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