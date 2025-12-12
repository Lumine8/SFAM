from setuptools import setup, find_packages

setup(
    name="sfam-ADR",
    version="1.0.0",
    description="Secure Feature Abstraction Model (SFAM) & SecuADR Engine",
    author="lumine8",
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
