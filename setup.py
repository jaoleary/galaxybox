from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
      long_description = fh.read()

setup(name='galaxybox',
      version='0.0.1',
      description='A set of tools for analysing EMERGE simulation output',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      author='Joseph A. O\'Leary',
      author_email='joleary@usm.lmu.de',
      license='GNU',
      url = 'https://github.com/jaoleary/galaxybox',
      packages=find_packages(include = ['galaxybox', 'galaxybox.*']),
      include_package_data=True,
      install_requires=[
            'astropy',
            'halotools',
            'h5py',
            'numpy',
            'pandas',
            'scipy>=1.4.0',
            'tqdm'],
      zip_safe=False)
