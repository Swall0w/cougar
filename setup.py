from os import path
import re

from setuptools import find_packages, setup

package_name = "cougar"
root_dir = path.abspath(path.dirname(__file__))


with open(path.join(root_dir, package_name, '__init__.py'), encoding='utf-8') as f:
    init_text = f.read()
    version = re.search(r'__version__ = [\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__ =\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__ =\s*[\'\"](.+?)[\'\"]', init_text).group(1)

assert version
assert author
assert url

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    version=version,
    description="PyTorch deep learning Vision library for fast prototyping",
    long_description=long_description,
    author=author,
    url=url,
    include_package_data=True,
    license=license,
    packages=find_packages(exclude=('tests')),
    test_suite='tests',
    entry_points="""
    [console_scripts]
    pig = pig.pig:main
    """,
    )