import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

def create_zip():
    print("Creating DySTCA_FINAL_KAGGLE.zip...")
    shutil.make_archive(
        base_name=os.path.join(PARENT_DIR, "DySTCA_FINAL_KAGGLE_Verified"),
        format="zip",
        root_dir=PARENT_DIR,
        base_dir="DySTCA_FINAL_KAGGLE"
    )
    print("Package created at:", os.path.join(PARENT_DIR, "DySTCA_FINAL_KAGGLE_Verified.zip"))

if __name__ == "__main__":
    create_zip()
