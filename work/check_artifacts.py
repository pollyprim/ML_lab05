"""Diagnostics: compare train vs val image statistics (artifacts)."""
import numpy as np, pandas as pd
from PIL import Image
from pathlib import Path
import torch, torch.nn.functional as F

DATA = Path("/workspace/work/data")
tt = pd.read_csv(DATA/"train.csv", dtype={"id":str}); vt = pd.read_csv(DATA/"val.csv", dtype={"id":str})

def stats(folder, ids):
    arrs = []
    for i in ids[:200]:
        name = i if i.endswith(".jpg") else i+".jpg"
        a = np.asarray(Image.open(DATA/folder/name).convert("RGB"), dtype=np.float32)
        g = a.mean(axis=2)
        f = np.fft.fftshift(np.abs(np.fft.fft2(g)))
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy-h/2)**2 + (xx-w/2)**2)
        hf = f[r > np.percentile(r, 70)].mean() / (f.mean()+1e-9)   # high-freq energy ratio
        lap = np.abs(np.gradient(g,axis=0)[1:-1,1:-1]).mean() + np.abs(np.gradient(g,axis=1)[1:-1,1:-1]).mean()
        arrs.append([a.std(), hf, lap, a.max(), (a>250).mean(), (a<5).mean()])
    return np.array(arrs)

s_tr, s_va = stats("train", tt.id), stats("val", vt.id)
names = ["std", "hf_ratio", "grad_mag", "max_px", "sat_white", "sat_dark"]
for j,n in enumerate(names):
    print(f"{n:10s} train mean {s_tr[:,j].mean():8.3f} | val mean {s_va[:,j].mean():8.3f}")
