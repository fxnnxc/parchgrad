from setuptools import find_packages, setup

setup(
    name="parchgrad",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        'tqdm',
        'omegaconf',
        'timm',
        'einops'
    ]
)

