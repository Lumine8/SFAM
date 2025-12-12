from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="sfam-ADR",
    version="1.0.0",
    description="Secure Feature Abstraction Model (SFAM) & SecuADR Engine",
    
    # ⚠️ THIS MAKES YOUR PYPI PAGE LOOK GOOD
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    author="lumine8",
    url="https://github.com/Lumine8/SFAM",  # Adds a link to your GitHub
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