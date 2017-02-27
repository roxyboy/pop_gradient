from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pop_gradient',
      version='0.1',
      description='Statistical analysis for POP model in CESM',
      author='Takaya Uchida',
      author_email='takaya@ldeo.columbia.edu',
      license='LDEO',
      packages=['pop_gradient'],
      install_requires=[
          'numpy','scipy','xarray','netCDF4'
      ],
      zip_safe=False)
