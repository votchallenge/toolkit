#!/usr/bin/env python

from distutils.core import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='vot-toolkit',
    version='8.0',
    description='Perform visual object tracking experiments and analyze results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luka Cehovin Zajc',
    author_email='luka.cehovin@gmail.com',
    url='https://github.com/votchallenge/toolkit',
    packages=['vot'],
    install_requires=[
        "vot-trax>=3.0",
        "tqdm==4.32",
        "numpy>=1.16",
        "opencv-python>=4.0",
        "six",
        "plotly>=4.5",
        "dash>=1.8",
        "pylatex>=1.3",
        "jsonschema>=3.2",
        "pyYAML>=5.3",
        "matplotlib>=3.1",
        "Pillow>=7.0",
        "numba>=0.48",
        "requests>=2.22"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.4',
    entry_points={
        'console_scripts': ['vot=vot.cli:main'],
    },
)

