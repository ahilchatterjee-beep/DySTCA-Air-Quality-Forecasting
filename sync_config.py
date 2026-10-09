import os

with open('c:/Users/Ahil Chatterjee/Desktop/Parent/DySTCA_FINAL_KAGGLE/src/delhi/config.py', 'r') as f:
    delhi = f.read()

greece_vars = {
    'MAX_EPOCHS': '1',
    'NUM_STATIONS': '31',
    'SEQUENCE_LENGTH': '24',
    'PREDICTION_HORIZON': '8',
    'DATA_INTERVAL_DESCRIPTION': '"native 3-hour intervals from Greek dataset (NOT interpolated to hourly)"',
    'FORECAST_DESCRIPTION': '"8 forecast steps = 24 hours (8 * 3h = 24h)"'
}

out = []
for line in delhi.split('\n'):
    mod = False
    for k, v in greece_vars.items():
        if line.startswith(f'{k} ='):
            out.append(f'{k} = {v}')
            mod = True
            break
    if not mod:
        out.append(line)

with open('c:/Users/Ahil Chatterjee/Desktop/Parent/DySTCA_FINAL_KAGGLE/src/greece/config.py', 'w') as f:
    f.write('\n'.join(out))
