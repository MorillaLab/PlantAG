from setuptools import setup, find_packages

setup(
    name="planttag",
    version="1.0.0",
    description="Plant Tissue Annotation via Graph Topology — automated cell-type annotation for plant spatial transcriptomics (3-layer GCN, hidden=256)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aberbache M., Luna A.M., Morilla I.",
    author_email="ian.morilla@ihsm.uma-csic.es",
    url="https://github.com/MorillaLab/PlantAG",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13",
        "torch-geometric>=2.2",
        "giotto-tda>=0.5",
        "scanpy>=1.9",
        "anndata>=0.9",
        "numpy>=1.23",
        "scipy>=1.9",
        "pandas>=1.5",
        "scikit-learn>=1.1",
        "leidenalg>=0.9",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "velocity": ["scvelo>=0.3"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    keywords="spatial transcriptomics plant cell annotation GNN topological data analysis tomato geminivirus",
)
