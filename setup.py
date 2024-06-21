#!/usr/bin/env python

from os.path import join, dirname, abspath, isfile
from setuptools import find_packages, setup
import json

this_directory = abspath(dirname(__file__))
with open(join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = []
if isfile(join(this_directory, "requirements.txt")):
    with open(join(this_directory, "requirements.txt"), encoding='utf-8') as f:
        install_requires = f.readlines()

__version__ = "0.0.0"

exec(open(join(dirname(__file__), 'vot', 'version.py')).read())

entrypoints = {
    'console_scripts': ['vot=vot.utilities.cli:main'],
}

for r in ["analysis", "downloader", "experiment", "indexer", "loader", "transformer"]:
    registry = join(this_directory, "data", "aliases", r + ".json")    
    if isfile(registry):
        try:
            with open(registry, encoding='utf-8') as f:
                data = json.load(f)
                
                entrypoints['vot_' + r] = ["%s=%s" % (k, ":".join(v.rsplit(".", 1))) for k, v in data.items()]
        except Exception:
            pass

setup(name='vot-toolkit',
    version=__version__,
    description='Perform visual object tracking experiments and analyze results',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Luka Cehovin Zajc',
    author_email='luka.cehovin@gmail.com',
    url='https://github.com/votchallenge/toolkit',
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.7',
    entry_points=entrypoints,
    extras_require = {
        'jupyter':  ["ipywidgets", "jupyter", "itables"],
    },
)

