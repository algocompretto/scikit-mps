"""
Multiple Point Simulation (MPS) using MPSlib
See:
https://github.com/algocompretto/scikit-mps
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


from distutils.core import setup

setup(
    name='scikit-mps',
    version='0.1.0',
    description='Multiple point statistical (MPS) simulation',
    author='Thomas Mejer Hansen and Gustavo Pretto Scholze',
    author_email='thomas.mejer.hansen@gmail.com & scholzegustavo@gmail.com',
    url='https://github.com/algocompretto/scikit-mps',  # use the URL to the github repo
    download_url='https://github.com/algocompretto/scikit-mps/master.zip',
    keywords=['geostatistics', 'simulation', 'MPS'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy >= 1.0.2', 'matplotlib >= 1.0.0'],
)
