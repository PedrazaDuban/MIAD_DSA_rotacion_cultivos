from setuptools import setup, find_packages

setup(
    name='modelo_rendimiento_cultivos',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'geopandas',
        # ... otras dependencias ...
    ],
)
