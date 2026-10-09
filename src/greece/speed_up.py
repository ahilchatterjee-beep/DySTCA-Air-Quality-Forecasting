with open('Training/evaluate_greece_only.py', encoding='utf-8') as f:
    content = f.read()

content = "import torch\ntorch.set_num_threads(16)\n" + content

with open('Training/evaluate_greece_only.py', 'w', encoding='utf-8') as f:
    f.write(content)
