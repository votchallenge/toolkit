#!/usr/bin/env python

from distutils.core import setup

setup(name='Visual Object Tracking toolkit',
      version='1.0',
      description='Perform visual tracking experiments and analyze results',
      author='Luka Cehovin Zajc',
      author_email='luka.cehovin@gmail.com',
      url='https://github.com/votchallenge/toolkit',
      packages=['vot', 'vot.training'],
      install_requires=[
        "tqdm==4.32.2",
        "numpy>=1.16",
        "opencv-python>=4.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)

