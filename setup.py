from setuptools import find_packages, setup

setup(
    name="neuroaugment",
    version="0.1.0",
    description="Causality-aware augmentation and simulation for multimodal physiological time series.",
    author="NeuroAugment contributors",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "torch==2.2.2",
        "click==8.1.7",
        "PyYAML==6.0.1",
        "matplotlib==3.8.2",
        "tqdm==4.66.1",
        "networkx==3.2.1",
        "Jinja2==3.1.6",
        "MarkupSafe==3.0.3",
        "filelock==3.19.1",
        "fsspec==2025.10.0",
        "sympy==1.14.0",
    ],
    extras_require={
        "dev": ["pytest==7.4.3", "ruff==0.1.8", "mypy==1.7.1"],
        "privacy": ["opacus==1.4.0"],
        "federated": ["flwr==1.6.0"],
        "viz": ["umap-learn==0.5.5"],
    },
    entry_points={"console_scripts": ["neuroaugment=neuroaugment.cli.main:cli"]},
)
