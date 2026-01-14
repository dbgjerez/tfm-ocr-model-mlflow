import os
import shutil

source_dir = "English/Fnt"
target_dir = "English/Fnt_ANPR"

os.makedirs(target_dir, exist_ok=True)

# Classes we keep: Samples 1-10 (digits) and 11-36 (uppercase)
valid_samples = list(range(1, 37))  # 1..36

for s in valid_samples:
    sample_name = f"Sample{s:03d}"
    src = os.path.join(source_dir, sample_name)
    dst = os.path.join(target_dir, sample_name)
    shutil.copytree(src, dst)

print("✔ ANPR dataset extracted successfully!")

