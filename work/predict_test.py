"""Generate final_submission.csv from a trained model on test set."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from common import *

import os
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("DATA_DIR", "./data"))
ckpt, out = sys.argv[2], sys.argv[3]
IMG_SIZE = int(sys.argv[4]) if len(sys.argv) > 4 else 128

test_table = pd.read_csv(DATA / "test.csv", dtype={"id": str})
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD)])
seed_everything()
t0 = time.time()
test_set = ContestDataset(test_table, DATA, "test", eval_transform, IMG_SIZE)
model = torch.load(ckpt, map_location=DEVICE, weights_only=False)
preds = predict(model, test_set)
sub = pd.DataFrame({"id": test_set.ids, "label": preds})
sub.to_csv(out, index=False)
print(f"saved {out}: {len(sub)} rows in {time.time()-t0:.1f}s")
print(sub.head())
