import os
import sys
import subprocess

commands = [
    [sys.executable, 'src/delhi/Training/reproduce_baseline_run.py'],
    [sys.executable, 'src/delhi/Training/multiseed_train.py', '--seed', '42'],
    [sys.executable, 'src/delhi/Training/multiseed_train.py', '--seed', '123'],
    [sys.executable, 'src/delhi/Training/multiseed_train.py', '--seed', '2024'],
    [sys.executable, 'src/delhi/Training/multiseed_train.py', '--seed', '3407'],
    [sys.executable, 'src/delhi/Training/multiseed_train.py', '--seed', '9999'],
    [sys.executable, 'run_ablation.py', '--variant', 'full'],
    [sys.executable, 'run_ablation.py', '--variant', 'wo_tcn'],
    [sys.executable, 'run_ablation.py', '--variant', 'wo_gat'],
    [sys.executable, 'run_ablation.py', '--variant', 'wo_cross_attention'],
    [sys.executable, 'run_ablation.py', '--variant', 'wo_dynamic_graph'],
    [sys.executable, 'run_ablation.py', '--variant', 'autoregressive'],
    [sys.executable, 'run_greece.py']
]

for cmd in commands:
    print('Starting:', ' '.join(cmd))
    subprocess.run(cmd)
