#!/usr/bin/env python

from setuptools import setup
import code

setup(
    name='code',
    version=str(code.__version__),
    description='code for gan demo.',
    author='Tao LIN',
    author_email='itamtao@gmail.com',
    packages=['code', 'code.model', 'code.utils'],
    scripts=[],
    keywords=['gan', 'demo'],
    package_data={},
    install_requires=open('requirements.txt').read().split()
)
