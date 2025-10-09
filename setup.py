from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="cosmic",
    version="0.3.0",
    packages=find_packages(),
    license="MIT",
    description="Ferramentas de análise para dados magnéticos da missão Cluster (ESA).",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=["numpy>=1.24", "scipy>=1.9", "matplotlib>=3.6", "pandas>=1.5"],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
