from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='MDA diffusion',
      version='0.0.0',
      description='Minimum dissipation approximation for hydrodynamic size.',
      url='https://github.com/RadostW/MininumDissipationApproximation/',
      author='Radost Waszkiewicz',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      project_urls = {
          'Documentation': 'https://mdadiffusion.readthedocs.io',
          'Source': 'https://github.com/RadostW/minimumdissipationapproximation/'
      },
      license='GNU GPLv3',
      packages=['mdadiffusion'],
      zip_safe=False)
