import os
import sys
from setuptools import setup, find_packages
import pathlib
from itertools import chain

here = pathlib.Path(__file__).parent.resolve()
package_basedir = os.path.abspath(os.path.dirname(__file__))
package_basename = 'victor'
long_description = (here / "README.md").read_text(encoding="utf-8")

sys.path.insert(0, os.path.abspath(package_basename))
import _version
version = _version.__version__

extras_require = {
    'mcmc': ['cobaya','getdist'],
    'camb': ['camb']
}
extras_require['all'] = list(set(chain(*extras_require.values())))

setup(name=package_basename,
      version=version,
      description="Python code for void-galaxy or density-split-galaxy cross-correlation models and fitters",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/seshnadathur/victor/tree/develop",
      author='Seshadri Nadathur',
      author_email='seshadri.nadathur@port.ac.uk',
      license='GPLv3',
      install_requires=['numpy>=1.17.2','scipy>=1.6.3','matplotlib>=3.1.1','PyYAML>=5.1','h5py','astropy'],
      python_requires=">=3.7.4",
      extras_require=extras_require,
      packages=[package_basename]
      )
