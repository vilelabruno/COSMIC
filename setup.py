from setuptools import setup, find_packages

setup(
    name='pyExtremeHelper',
    version='0.3',
    packages=find_packages(),
    license='MIT',
    description='Um pacote Python para analise de eventos extremos',
    long_description=open('README.md').read(),
    install_requires=['numpy'],  
)
