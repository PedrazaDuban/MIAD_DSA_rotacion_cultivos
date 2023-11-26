from setuptools import setup, find_packages

setup(
    name='recomendacion-cultivos',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'mlflow',
        # Agrega otras dependencias según sea necesario
    ],
)
