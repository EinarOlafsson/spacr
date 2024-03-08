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
    'ttkthemes'
]

setup(
    name="spacr",
    version="0.0.1",
    author="Einar Birnir Olafsson",
    author_email="olafsson@med.umich.com",
    description="A brief description of your package",
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
)

