'''Setup.py'''

from distutils.core import setup
from setuptools import find_packages

setup(
    name='ellipsinator',
    version='0.0.2',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/mckib2/ellipsinator',
    license='GPLv3',
    description='Ellipse tools for Python',
    long_description=open('README.rst', encoding='utf-8').read(),
    install_requires=[
        "numpy>=1.19.1",
    ],
    python_requires='>=3.5',
)
