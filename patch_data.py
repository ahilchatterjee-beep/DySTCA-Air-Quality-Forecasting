import os

def patch_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        
    out = []
    for line in lines:
        if line.strip() == "return {":
            out.append("    return {\n")
            out.append("        'X_train': X_train_tensor[:32],\n")
            out.append("        'Y_train': Y_train_tensor[:32],\n")
            out.append("        'station_records_train': records_train[:32],\n")
            out.append("        'X_val': X_val_tensor[:32],\n")
            out.append("        'Y_val': Y_val_tensor[:32],\n")
            out.append("        'station_records_val': records_val[:32],\n")
            out.append("        'X_test': X_test_tensor[:32],\n")
            out.append("        'Y_test': Y_test_tensor[:32],\n")
            out.append("        'station_records_test': records_test[:32],\n")
            out.append("        'y_scaler': target_scaler,\n")
            out.append("        'feature_scaler': feature_scaler,\n")
            out.append("    }\n")
            break
        else:
            out.append(line)
            
    with open(path, 'w') as f:
        f.writelines(out)

patch_file('c:/Users/Ahil Chatterjee/Desktop/Parent/DySTCA_FINAL_KAGGLE/src/delhi/preprocessing/pre.py')
patch_file('c:/Users/Ahil Chatterjee/Desktop/Parent/DySTCA_FINAL_KAGGLE/src/greece/preprocessing/pre.py')
