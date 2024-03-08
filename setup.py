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
            'gui_mask=spacr.gen_masks_gui:gui_mask',
            'gui_measure=spacr.measure_crop_gui:gui_measure',
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

