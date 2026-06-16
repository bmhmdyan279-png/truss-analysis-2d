from setuptools import setup, find_packages

setup(
    name="truss_analysis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pytest>=7.0.0",
    ],
)