"""
Setup configuration for Fashion Retrieval System.
"""
from setuptools import setup, find_packages

setup(
    name="fashion-retrieval",
    version="0.1.0",
    description="Multi-vector fashion image retrieval with CLIP, FashionCLIP, and attribute decomposition",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "pinecone>=3.0.0",
        "datasets>=2.14.0",
        "Pillow>=9.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.2.0",
        "streamlit>=1.28.0",
        "spacy>=3.6.0",
    ],
    entry_points={
        "console_scripts": [
            "fashion-index=part_a_indexer.index:main",
            "fashion-evaluate=scripts.evaluate:main",
        ]
    },
)
