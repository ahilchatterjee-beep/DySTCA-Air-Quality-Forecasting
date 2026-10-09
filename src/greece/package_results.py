import os
import zipfile
import glob

files_to_zip = [
    "best_spatiotemporal_model_greece.pth",
    "greece_test_metrics.json",
    "greece_training_history.json",
    "greece_predictions.npz",
    "config.py"
]
files_to_zip.extend(glob.glob("results_graphs/*"))
files_to_zip.extend(glob.glob("all_generated_graphs/*"))
files_to_zip.extend(glob.glob("paper_artifacts/*"))

with zipfile.ZipFile("DySTCA_Greece_Results.zip", 'w') as z:
    for f in files_to_zip:
        if os.path.exists(f):
            z.write(f)
            print(f"Added {f} to zip.")
        else:
            print(f"Warning: {f} not found.")

print("\nCreated DySTCA_Greece_Results.zip successfully!")
