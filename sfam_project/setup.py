from setuptools import setup, find_packages
from pathlib import Path

# 1. Read the README file for the PyPI description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="sfam-ADR",
    version="1.2.5",  # <--- BUMPED TO 1.x.x TO FIX UPLOAD ERROR
    description="A neuro-symbolic, privacy-preserving biometric authentication engine.",
    
    # 2. This ensures your README.md shows up on PyPI
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    author="lumine8",
    url="https://github.com/Lumine8/SFAM",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "timm",
        "pillow",
        "torchvision",
        "opencv-python",
        "fastapi",
        "uvicorn",
        "pydantic"
    ],
    python_requires='>=3.8',
)