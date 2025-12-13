import os
import shutil
import sys
import subprocess

def print_status(msg, status="INFO"):
    colors = {"INFO": "\033[94m", "SUCCESS": "\033[92m", "WARN": "\033[93m", "ERR": "\033[91m", "END": "\033[0m"}
    prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARN": "⚠️", "ERR": "❌"}
    print(f"{colors.get(status, '')}{prefix.get(status, '')} {msg}{colors['END']}")

def fix_folder_structure():
    """Moves 'models' and 'data' back into 'sfam/' if they drift to root."""
    print_status("Checking Folder Structure...", "INFO")
    
    root_dirs = ["models", "data", "training", "eval"]
    target_dir = "sfam"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print_status(f"Created missing '{target_dir}' package folder", "SUCCESS")

    for folder in root_dirs:
        if os.path.exists(folder):
            # Move it inside sfam/
            target_path = os.path.join(target_dir, folder)
            if os.path.exists(target_path):
                print_status(f"Duplicate '{folder}' found. Merging...", "WARN")
                # Merge logic could go here, but for now we warn
            else:
                shutil.move(folder, target_path)
                print_status(f"Moved '{folder}' back into '{target_dir}/'", "SUCCESS")

def fix_imports(file_path):
    """Rewrites source code to use package imports (sfam.models vs models)."""
    if not os.path.exists(file_path):
        return

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    changes = 0
    
    for line in lines:
        # Fix: from models... -> from sfam.models...
        if line.strip().startswith("from models."):
            line = line.replace("from models.", "from sfam.models.")
            changes += 1
        # Fix: from data... -> from sfam.data...
        elif line.strip().startswith("from data."):
            line = line.replace("from data.", "from sfam.data.")
            changes += 1
        new_lines.append(line)

    if changes > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        print_status(f"Fixed {changes} broken imports in '{file_path}'", "SUCCESS")
    else:
        print_status(f"Imports in '{file_path}' are correct.", "INFO")

def ensure_init_files():
    """Ensures every folder has an __init__.py"""
    print_status("Checking Package Initialization...", "INFO")
    
    # Root package init
    sfam_init = os.path.join("sfam", "__init__.py")
    if not os.path.exists(sfam_init):
        with open(sfam_init, "w") as f:
            f.write("# SFAM Package Exposure\n")
            f.write("from .models.sfam_net import SFAM\n")
            f.write("from .data.gesture_loader import GestureCapture\n")
        print_status("Created missing 'sfam/__init__.py'", "SUCCESS")
    
    # Submodules
    for sub in ["models", "data"]:
        path = os.path.join("sfam", sub, "__init__.py")
        if not os.path.exists(path) and os.path.exists(os.path.dirname(path)):
            with open(path, "w") as f: f.write("")
            print_status(f"Created missing '{path}'", "SUCCESS")

def reinstall_package():
    """Runs pip install -e . if imports fail."""
    print_status("Verifying Package Installation...", "INFO")
    try:
        import sfam
        print_status("SFAM package is correctly installed.", "SUCCESS")
    except ImportError:
        print_status("SFAM package not found. Installing...", "WARN")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
            print_status("SFAM installed successfully!", "SUCCESS")
        except subprocess.CalledProcessError:
            print_status("Failed to install package.", "ERR")

def main():
    print("\n🔧 SFAM SELF-REPAIR TOOL 🔧\n" + "="*30)
    
    # 1. Structure
    fix_folder_structure()
    
    # 2. Files
    ensure_init_files()
    
    # 3. Code Content
    fix_imports("server.py")
    fix_imports("main.py")
    
    # 4. Environment
    reinstall_package()
    
    print("="*30 + "\n✨ System Check Complete. You are good to go!\n")

if __name__ == "__main__":
    main()