import os

path = "c:/Users/Ahil Chatterjee/Desktop/Parent/DySTCA_FINAL_KAGGLE/validate_final_package.py"
with open(path, "r") as f:
    src = f.read()

src = src.replace('import os', 'import os\nimport datetime')
src = src.replace('"baseline_delhi_reproduced_2026-10-05"', f'f"baseline_delhi_reproduced_{{datetime.date.today()}}"')

with open(path, "w") as f:
    f.write(src)
