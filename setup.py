from setuptools import setup

setup(
    name='pysliceplorer',
    version='0.1',
    description='Python Library for Slicing Visualization',
    url='https://github.com/OnlyPaul/pysliceplorer',
    author='Pavares Charoenchaipiyakul',
    author_email='pavares.cha@gmail.com',
    license='MIT',
    packages=['pysliceplorer'],
    install_requires=[
        'numpy',
        'sobol_seq'
      ],
    zip_safe=False
)