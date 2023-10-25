import numpy
import os
from setuptools import setup
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize

# See https://stackoverflow.com/a/21621689/1862861
# for why this is here.
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())

ext_modules=[
    Extension(name="waveform",
              sources=["waveform.pyx"],
              libraries=['m'],  # Unix-like specific
              extra_compile_args=['-O3', '-ffast-math'],
              include_dirs=[numpy.get_include(), "."])
    ]

setup(name="gift",
      ext_modules=cythonize(ext_modules, language_level='3'),
      include_dirs=[numpy.get_include(), "."],
      cmdclass={'build_ext': build_ext},
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',],
      keywords="gravitational waves cosmology bayesian inference",
      packages=['hns'],
      install_requires=['numpy', 'scipy', 'corner', 'cython'],
      package_data={"": ['*.pyx', '*.pxd']},
      )
