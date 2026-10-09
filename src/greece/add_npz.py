with open('Training/evaluate_greece_only.py', encoding='utf-8') as f:
    content = f.read()

np_save_code = """
print("Saving greece_predictions.npz...")
import numpy as np
np.savez_compressed("greece_predictions.npz", 
    targets_original=targets_original.numpy() if hasattr(targets_original, "numpy") else np.array(targets_original), 
    predictions_original=predictions_original.numpy() if hasattr(predictions_original, "numpy") else np.array(predictions_original))
print("Saved greece_predictions.npz")
with open("test_metrics_summary.json"
"""

content = content.replace('with open("test_metrics_summary.json"', np_save_code)

with open('Training/evaluate_greece_only.py', 'w', encoding='utf-8') as f:
    f.write(content)
