from distutils.core import setup
from setuptools import find_packages

NAME = 'dialogue_state.baseline.v1'
VERSION = '0.1.1'
LICENSE = 'MIT'
DESCRIPTION = 'State Prediction for Task-oriented task'

with open('README.md', encoding="utf-8") as file:
    description = file.read()

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages('src'),
    license=LICENSE,
    zip_safe=True,
    description=DESCRIPTION,
    long_description=description,
    long_description_content_type='text/markdown',
    author='Tien V. Nguyen',
    author_email='tiennv@gradients.host',
    url='https://github.com/gradients-tech/dialogstate',
    python_require='>=3.8.15',
    keywords=[],
    install_requires=[
        'accelerate~=0.17',
        'datasets~=2.10.1',
        'torch~=1.13.1',
        'transformers~=4.26.1',
        'nltk~=3.8.1',
        'numpy~=1.24.2',
        'filelock~=3.9.0',
        'tqdm~=4.65.0',
        'pandas~=1.5.3',
        'setuptools~=65.6.3',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)