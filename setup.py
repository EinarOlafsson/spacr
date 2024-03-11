from setuptools import setup, find_packages

dependencies = [
    'torch',
    'torchvision',
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
    version="0.0.11",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="Spatial phenotype analysis of crisp screens (SpaCr). A collection of functions for generating cellpose masks -> single object images and measurements -> annotation and classification of single object images. Spacr uses batch normalization to facilitate accurate segmentation of objects with low foreground representation.",
    long_description=open('README.md').read(),
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
    ],