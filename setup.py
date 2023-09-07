from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='sider',
    description='Dual representation learning model for drug-side effect frequency prediction',
    version='1.0',
    #packages=find_packages(''),
    #package_dir={'': '.'},
    py_modules=[splitext(basename(path))[0] for path in glob('./*.py')],
)