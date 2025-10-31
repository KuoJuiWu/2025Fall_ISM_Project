from setuptools import setup, find_packages

setup(
    name='jwst-utils',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=['astropy', 'matplotlib', 'numpy', 'scipy'],
)
