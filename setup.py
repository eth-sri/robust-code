from setuptools import find_packages
from setuptools import setup

setup(
    name="robustcode",
    version="0.1",
    description="Adversarial Robustness for Code (presented at ICML20)",
    author="SRI Lab, ETH Zurich",
    license="Apache License 2.0",
    install_requires=[
        "pygments==2.4.2",
        "nltk>=3.4.5",
        "scikit-learn==0.21.2",
        "datasketch==1.4.5",
        "tqdm==4.32.2",
        "torch==1.4.0",
        "torchtext==0.4.0",
        "sty==1.0.0b11",
        "dgl-cu102==0.4.3.post2",
        "networkx==2.3",  # https://networkx.github.io/
        "matplotlib==3.1.1",
        "pycparser==2.19",
        "grequests==0.3.0",
        "pytest==5.4.1",
        "pandas==1.0.5",
    ],
    extras_require={"test": ["pytest"], "dev": ["pre-commit", "flake8", "black"]},
    setup_requires=["wheel"],
    packages=find_packages(),
)
