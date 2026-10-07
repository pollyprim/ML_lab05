"""Create a small SYNTHETIC dataset (20 classes) to smoke-test the pipeline.
NOT real competition data - only used to verify code correctness here."""
import numpy as np, pandas as pd, random
from pathlib import Path
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED)
root = Path("synthetic"); 
for split in ["train", "val", "test"]:
    (root/split).mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
rows = {"train": [], "val": [], "test": []}
counts = {"train": 30, "val": 6, "test": 12}
img_id = 0
for cls in range(20):
    # class-specific pattern: base color + geometric motif
    base = rng.integers(0, 200, size=3).astype(np.uint8)
    for split, n in counts.items():
        for _ in range(n):
            img_id += 1
            x = rng.integers(0, 64, size=(64,64,3)).astype(np.uint8) // 4 + base
            # add class-correlated structure
            y0 = cls * 3 % 50
            x[y0:y0+8, :, :] = (base + 40).clip(0,255)
            if split == "val" and rng.random() < 0.3:   # simulate artifacts
                im = Image.fromarray(x).filter.__self__  # noop
                from PIL import ImageFilter
                x = np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(1.5)))
            p = root/split/f"{img_id}.jpg"
            Image.fromarray(x.astype(np.uint8)).save(p, quality=70)
            rows[split].append((f"{img_id}.jpg", cls))

for split in rows:
    df = pd.DataFrame(rows[split], columns=["id","label"]).sample(frac=1, random_state=SEED)
    df.to_csv(root/f"{split}.csv", index=False)
print({s: len(r) for s,r in rows.items()})
