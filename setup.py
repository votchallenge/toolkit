#!/usr/bin/env python

from distutils.core import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []
if path.isfile(path.join(this_directory, "requirements.txt")):
    with open(path.join(this_directory, "requirements.txt"), encoding='utf-8') as f:
        install_requires = f.readlines()

setup(name='vot-toolkit',
    version='0.2.0',
    description='Perform visual object tracking experiments and analyze results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luka Cehovin Zajc',
    author_email='luka.cehovin@gmail.com',
    url='https://github.com/votchallenge/toolkit',
    packages=['vot', 'vot.analysis', 'vot.dataset', 'vot.experiment', 'vot.region', 'vot.stack', 'vot.tracker', 'vot.utilities'],
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['vot=vot.cli:main'],
    },
)

