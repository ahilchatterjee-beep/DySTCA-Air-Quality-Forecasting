import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def clean(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in dirs:
            if name == "__pycache__":
                shutil.rmtree(os.path.join(root, name))
        for name in files:
            if name.endswith(".pyc") or name.endswith(".log") or name.endswith(".tmp"):
                os.remove(os.path.join(root, name))

clean(PROJECT_ROOT)
print("Cleaned package junk.")
