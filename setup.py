from setuptools import setup, find_packages


setup(
    name='waterbirds',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'pandas',
        'pytorch-lightning',
        'seaborn',
        'torch',
        'torchvision'
    ]
)