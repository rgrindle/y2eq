"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Dec 15, 2020

PURPOSE: Install code in PredictingUnseenFunctions, so that
         code inside this directory can import necessary files.

NOTES:

TODO:
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SRGD",
    version="0.0.1",
    author="Ryan Grindle",
    author_email="ryan.grindle@uvm.edu",
    description="The goal of this project is to create a neural network capable of generalization in the task of symbolic regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rgrindle/EquationLearner",
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
