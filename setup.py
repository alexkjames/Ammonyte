import os
import sys
import io

from setuptools import setup, find_packages

version = '0.0.1'

# Read the readme file contents into variable
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='Ammonite',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    version=version,
    license='GPL-3.0 License',
    description='A Python package for nonlinear paleoclimate data analysis',
    long_description=read("README.md"),
    long_description_content_type = 'text/markdown',
    author='Alexander James',
    author_email='akjames@usc.edu',
    url='https://github.com/alexkjames/Ammonite',
    keywords=['Paleoclimate, Data Analysis, Nonlinear'],
    classifiers=[],
    install_requires=[
        "pyleoclim>=0.7.0",
        "scipy>=1.7.1",
        "numpy>=1.21.5",
        "PyRQA>=8.0.0",
    ],
    python_requires=">=3.8.0"
)