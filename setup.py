from setuptools import setup, find_packages

setup(
    name="STitch3D",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scanpy',
        'anndata',
        'scipy',
        'matplotlib'
    ]
)