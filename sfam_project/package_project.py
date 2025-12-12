import os
import shutil

# 1. DEFINE FILE CONTENTS

setup_py_code = """from setuptools import setup, find_packages

setup(
    name="sfam",
    version="1.0.0",
    description="Secure Feature Abstraction Model (SFAM) & SecuADR Engine",
    author="Your Name",
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
"""

init_py_code = """# This makes the folder a Python Package
# We expose the main classes so you can do: 'from sfam import SFAM'

from .models.sfam_net import SFAM
from .data.gesture_loader import GestureCapture
"""

# 2. PERFORM RESTRUCTURING

def package_it():
    print("📦 Packaging SFAM as a library...")

    # A. Create the inner 'sfam' package folder
    if not os.path.exists("sfam"):
        os.makedirs("sfam")
        print("   -> Created 'sfam' package directory")

    # B. Move existing modules into 'sfam/'
    # We move 'models', 'data', 'training', 'eval' if they exist
    folders_to_move = ["models", "data", "training", "eval"]
    
    for folder in folders_to_move:
        if os.path.exists(folder):
            shutil.move(folder, f"sfam/{folder}")
            print(f"   -> Moved '{folder}' into 'sfam/'")

    # C. Create __init__.py to expose the API
    with open("sfam/__init__.py", "w", encoding="utf-8") as f:
        f.write(init_py_code)
    print("   -> Created sfam/__init__.py")

    # D. Create setup.py at the root
    with open("setup.py", "w", encoding="utf-8") as f:
        f.write(setup_py_code)
    print("   -> Created setup.py")

    # E. Fix imports in main.py and server.py (if they exist)
    # They need to change from 'from models...' to 'from sfam.models...' 
    # OR we just rely on the installed package structure.
    # Let's update main.py to use the new package style for local testing.
    
    print("\n✅ Packaging Structure Complete!")
    print("   To install this library, run:")
    print("   pip install -e .")

if __name__ == "__main__":
    package_it()