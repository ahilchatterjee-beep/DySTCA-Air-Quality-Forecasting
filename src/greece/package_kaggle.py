import os
import zipfile
import shutil

# Make sure config.py is restored to 100 epochs for kaggle
with open('config.py', 'r') as f:
    config_content = f.read()

config_content = config_content.replace('MAX_EPOCHS = 2', 'MAX_EPOCHS = 100')

with open('config_for_kaggle.py', 'w') as f:
    f.write(config_content)

files_to_zip = [
    "requirements.txt",
    "kaggle_train_greece.ipynb",
    "data/greek_urban_air_quality_2020_2024.csv",
    "preprocessing/__init__.py",
    "preprocessing/greek_preprocessing.py",
    "Model/__init__.py",
    "Model/new_model.py",
    "Model/revin.py",
    "Temporal/__init__.py",
    "Temporal/new_TE.py",
    "Temporal/tcn.py",
    "Temporal/residual_block.py",
    "Spatial/__init__.py",
    "Spatial/new_spatial_encoder.py",
    "Spatial/new_graph_builder.py",
    "Spatial/new_node_features.py",
    "Spatial/multihead_gat.py",
    "Spatial/gat_layer.py",
    "Spatial/hyper_heuristic.py",
    "Fusion/__init__.py",
    "Fusion/fusion_layer.py",
    "Prediction/__init__.py",
    "Prediction/prediction_layer.py",
    "Training/__init__.py",
    "Training/train_greece.py"
]

with zipfile.ZipFile('DySTCA_Greece_Kaggle.zip', 'w') as zipf:
    zipf.write('config_for_kaggle.py', 'config.py')
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file, file)
        else:
            print(f"Warning: {file} not found.")

print("DySTCA_Greece_Kaggle.zip created successfully.")
