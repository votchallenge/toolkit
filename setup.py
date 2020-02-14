#!/usr/bin/env python

from distutils.core import setup

setup(name='Visual Object Tracking toolkit',
      version='8.0',
      description='Perform visual object tracking experiments and analyze results',
      author='Luka Cehovin Zajc',
      author_email='luka.cehovin@gmail.com',
      url='https://github.com/votchallenge/toolkit',
      packages=['vot'],
      install_requires=[
        "tqdm==4.32",
        "numpy>=1.16",
        "opencv-python>=4.0",
        "trax>=3.0",
        "six",
        "plotly>=4.5",
        "dash>=1.8",
        "pylatex>=1.3",
        "jsonschema>=3.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    entry_points={
        'console_scripts': ['vot=vot.cli:main'],
    },
)

