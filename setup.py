from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='mdadiffusion',
      version='0.0.1',
      description='Minimum dissipation approximation for hydrodynamic size.',
      url='https://github.com/RadostW/mdadiffusion/',
      author='Radost Waszkiewicz',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      project_urls = {
          # 'Documentation': 'https://mdadiffusion.readthedocs.io', # TODO
          'Source': 'https://github.com/RadostW/mdadiffusion/'
      },
      license='MIT',
      packages=['mdadiffusion'],
      install_requires=['numpy>1.16','pygrpy','sarw_spheres'],
      zip_safe=False)
