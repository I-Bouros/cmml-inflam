#
# cmmlinflam setuptools script
#
# This file is part of CMMLINFLAM
# (https://github.com/I-Bouros/cmml-inflam.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the cmmlinflam module.
    The easiest way would be to just ``import cmmlinflam ``, but note that this may  # noqa
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('cmmlinflam'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()


setup(
    # Module name (lowercase)
    name='cmmlinflam',

    # Version
    version=get_version(),

    description='A collection of stochastic processes models used in the modelling of the effect of inflammation in the evolution of CMML.',  # noqa

    long_description_content_type='text/markdown',

    long_description=get_readme(),

    license='MIT "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='',

    maintainer_email='',

    url='https://github.com/I-Bouros/cmml-inflam.git',

    # Packages to include
    packages=find_packages(include=('cmmlinflam', 'cmmlinflam.*')),
    include_package_data=True,

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'numpy>=1.8',
        'pandas',
        'scipy',
        'pints @ git+git://github.com/pints-team/pints.git#egg=pints-latest',
        'plotly'
    ],
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    },
)
