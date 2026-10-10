"""Robust final pipeline v2: ResNet-18 pretrained, artifact-aware training.

Key fixes vs final_submission.py (which scored 0.22 on Kaggle):
  1. TRAIN/VAL images are "perfect" samples; TEST images carry real-world artifacts
     (blur / color inverse / low resolution). Training only on clean data causes a
     domain shift. Here every train image is randomly corrupted with the SAME
     artifact families at strong probabilities so the model learns to classify
     degraded inputs.
  2. Test-time augmentation (TTA): predictions are averaged over original +
     4 rotations + horizontal flip (aerial scenes have no canonical "up"),
     which typically adds +1-2% accuracy for free.
  3. Higher input resolution (default 160) - satellite objects are small and
     get destroyed by aggressive downscaling.

Usage:
    python robust.py <DATA_DIR> [out_csv] [img_size] [epochs] [--no-tta] [--clean-train]

Run:    DATA_DIR=/path/to/competition/data python robust.py $DATA_DIR final_submission.csv 160 25
Eval:  add --holdout to keep val out of fitting and report honest val accuracy.
"""
import sys, os, json, time, copy, random, io
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

# ---------------- config ----------------
SEED = 42
NUM_CLASSES = 20
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 160
EPOCHS = 25
LR = 2e-4
WD = 1e-4
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- artifact simulation ----------------
class RandomBlur:
    """Strong Gaussian blur - simulates optical malfunction / defocus."""
    def __init__(self, p=0.5, sig=(0.3, 2.0)): self.p, self.sig = p, sig
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(*self.sig)
            return transforms.functional.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])
        return img

class RandomLowRes:
    """Downscale then upscale - simulates low-resolution satellite capture."""
    def __init__(self, p=0.4, scales=(0.25, 0.5)): self.p, self.scales = p, scales
    def __call__(self, img):
        if random.random() < self.p:
            s = random.uniform(*self.scales)
            w, h = img.size
            small = img.resize((max(2, int(w * s)), max(2, int(h * s))), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return img

class RandomInvert:
    """Color inversion (negative film) - one of the stated artifact types."""
    def __init__(self, p=0.15): self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.invert(img)
        return img

class AddGaussianNoise:
    """Additive sensor noise on tensors in [0,1]."""
    def __init__(self, p=0.5, std=(0.01, 0.08)): self.p, self.std = p, std
    def __call__(self, img):
        if random.random() < self.p:
            s = float(np.random.uniform(*self.std))
            return torch.clamp(img + torch.randn_like(img) * s, 0, 1)
        return img

class RandomJPEG:
    """JPEG compression artifacts at various qualities."""
    def __init__(self, p=0.5, qrange=(10, 60)): self.p, self.qrange = p, qrange
    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(*self.qrange)
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img

def make_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        RandomJPEG(p=0.5),
        RandomBlur(p=0.5),
        RandomLowRes(p=0.4),
        RandomInvert(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
        AddGaussianNoise(p=0.5),
    ])

def make_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMNET_MEAN, IMNET_STD),
    ])

# ---------------- dataset ----------------
def image_file(data_dir, folder, image_id):
    name = image_id if str(image_id).endswith(".jpg") else f"{image_id}.jpg"
    return data_dir / folder / name

class ContestDataset(Dataset):
    def __init__(self, table, data_dir, folder, transform, img_size):
        self.ids = table.id.astype(str).tolist()
        self.labels = table.label.tolist() if "label" in table.columns else [-1] * len(self.ids)
        self.transform = transform
        self.images = []
        for image_id in self.ids:
            with Image.open(image_file(data_dir, folder, image_id)) as img:
                img = img.convert("RGB").resize((img_size, img_size), Image.BOX)
                self.images.append(np.asarray(img, dtype=np.uint8))
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        # .copy() is required: PIL must own a writable buffer; without it the
        # dataset crashes on Windows (OSError: image is in an unsupported mode)
        # and may share memory between samples.
        img = Image.fromarray(np.ascontiguousarray(self.images[i].copy()))
        return self.transform(img), self.labels[i]

class ConcatDatasetX(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.sizes = [len(d) for d in datasets]
        self.ids = sum([d.ids for d in datasets], [])
        self.labels = sum([d.labels for d in datasets], [])
    def __len__(self): return sum(self.sizes)
    def __getitem__(self, i):
        for d, s in zip(self.datasets, self.sizes):
            if i < s: return d[i]
            i -= s
        raise IndexError

# ---------------- model ----------------
def make_resnet(unfreeze="layer3+4"):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in backbone.parameters():
        p.requires_grad = False
    target = {"none": [], "layer4": [backbone.layer4], "layer3+4": [backbone.layer3, backbone.layer4],
              "all": [backbone]}[unfreeze]
    for blk in target:
        for p in blk.parameters():
            p.requires_grad = True
    nf = backbone.fc.in_features
    backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf, NUM_CLASSES))
    return backbone

# ---------------- train / eval ----------------
criterion = nn.CrossEntropyLoss()

def make_loader(ds, shuffle):
    g = torch.Generator().manual_seed(SEED)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=g, num_workers=0)

@torch.no_grad()
def evaluate(model, ds):
    model.eval(); correct = 0; total = 0
    for x, y in make_loader(ds, False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
    return correct / total

@torch.no_grad()
def predict_proba(model, ds, tta=True):
    """Returns class probabilities per sample. With TTA averages logits over
    identity + 90/180/270 rotations + horizontal flip (5 views)."""
    model.eval()
    n = len(ds)
    acc = torch.zeros(n, NUM_CLASSES, device=DEVICE)
    views = [lambda x: x]
    if tta:
        views += [
            lambda x: torch.rot90(x, 1, dims=(2, 3)),
            lambda x: torch.rot90(x, 2, dims=(2, 3)),
            lambda x: torch.rot90(x, 3, dims=(2, 3)),
            lambda x: torch.flip(x, dims=[3]),
        ]
    loader = make_loader(ds, shuffle=False)
    idx = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        outs = [model(v(x)) for v in views]
        probs = torch.stack([torch.softmax(o, 1) for o in outs]).mean(0)
        acc[idx:idx + len(x)] = probs.cpu()
        idx += len(x)
    return acc.numpy()

def train_model(model, name, train_set, val_set, epochs=EPOCHS, lr=LR):
    seed_everything()
    model = model.to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(make_loader(train_set, True)))
    best_state, best_acc = None, -1
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot_loss = 0; corr = 0
        for x, y in make_loader(train_set, True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step(); sched.step()
            tot_loss += loss.item() * len(y); corr += (model(x).argmax(1) == y).sum().item()
        tr_acc = corr / len(train_set)
        va_acc = evaluate(model, val_set) if val_set is not None else float("nan")
        history.append({"epoch": ep, "train_loss": tot_loss / len(train_set),
                        "train_acc": tr_acc, "val_acc": va_acc})
        # keep last weights (val may be part of training -> monitor only)
        if ep == epochs or (val_set is not None and va_acc > best_acc):
            best_acc = max(best_acc, va_acc if val_set is not None else 0)
            best_state = copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == 1:
            print(f"{name} | ep {ep:2d} | loss {tot_loss/len(train_set):.3f} | train acc {tr_acc:.3f} | val acc {va_acc:.3f}", flush=True)
    model.load_state_dict(best_state)
    dt = time.time() - t0
    print(f"{name}: done in {dt:.1f}s", flush=True)
    return pd.DataFrame(history), dt

# ---------------- main ----------------
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    DATA = Path(args[0]) if args else Path(os.environ.get("DATA_DIR", "./data"))
    OUT = Path(args[1]) if len(args) > 1 else Path("final_submission.csv")
    IMG_SIZE = int(args[2]) if len(args) > 2 else 160
    EPOCHS = int(args[3]) if len(args) > 3 else 25
    USE_TTA = "--no-tta" not in flags
    HOLDOUT = "--holdout" in flags
    CLEAN_TRAIN = "--clean-train" in flags   # ablation: old behaviour (no artifact aug)

    seed_everything()
    train_table = pd.read_csv(DATA / "train.csv", dtype={"id": str})
    val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})
    test_table = pd.read_csv(DATA / "test.csv", dtype={"id": str})

    tr_tf = make_eval_transform() if CLEAN_TRAIN else make_train_transform()
    ev_tf = make_eval_transform()

    t0 = time.time()
    train_set = ContestDataset(train_table, DATA, "train", tr_tf, IMG_SIZE)
    val_set = ContestDataset(val_table, DATA, "val", ev_tf, IMG_SIZE)
    test_set = ContestDataset(test_table, DATA, "test", ev_tf, IMG_SIZE)
    print(f"data loaded in {time.time()-t0:.1f}s | train {len(train_set)} val {len(val_set)} test {len(test_set)} | device {DEVICE}")

    if HOLDOUT:
        fit_set, mon_set = train_set, val_set
    else:
        val_as_train = ContestDataset(val_table, DATA, "val", tr_tf, IMG_SIZE)
        fit_set, mon_set = ConcatDatasetX([train_set, val_as_train]), None

    model = make_resnet(unfreeze="layer3+4")
    history, dt = train_model(model, "resnet18-robust", fit_set, mon_set, epochs=EPOCHS)
    history.to_csv("history_robust.csv", index=False)

    if HOLDOUT:
        acc = evaluate(model, val_set)
        print(f"HELD-OUT VAL ACCURACY: {acc:.4f}")
        json.dump({"mode": "holdout", "val_acc": acc, "img_size": IMG_SIZE,
                   "epochs": EPOCHS, "tta": USE_TTA, "clean_train": CLEAN_TRAIN,
                   "train_time_s": round(dt, 1)}, open("result_robust.json", "w"), indent=1)
    else:
        torch.save(model, "model_robust.pt")
        t1 = time.time()
        probs = predict_proba(model, test_set, tta=USE_TTA)
        preds = probs.argmax(1)
        sub = pd.DataFrame({"id": test_set.ids, "label": preds})
        assert sub.id.tolist() == test_table.id.tolist()
        assert sub.label.between(0, NUM_CLASSES - 1).all()
        sub.to_csv(OUT, index=False)
        print(f"submission -> {OUT} ({len(sub)} rows) in {time.time()-t1:.1f}s (predict incl. TTA)")
        json.dump({"mode": "fit-train-val", "img_size": IMG_SIZE, "epochs": EPOCHS,
                   "tta": USE_TTA, "clean_train": CLEAN_TRAIN, "rows": len(sub),
                   "train_time_s": round(dt, 1),
                   "predict_time_s": round(time.time() - t1, 1)},
                  open("result_robust.json", "w"), indent=1)
    print("DONE")
