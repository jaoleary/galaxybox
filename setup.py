from setuptools import setup, find_packages

setup(name='galaxybox',
      version='0.0.1',
      description='A set of tools for analysing simulation output',
      author='Joseph A. O\'Leary',
      author_email='joleary@usm.lmu.de',
      license='GNU',
      url = 'https://github.com/jaoleary/galaxybox',
      packages=find_packages(include = ['galaxybox', 'galaxybox.*']),
      include_package_data=True,
      install_requires=[
            'astropy',
            'h5py',
            'halotools',
            'numpy',
            'pandas',
            'scipy',
            'tqdm'],
      zip_safe=False)
