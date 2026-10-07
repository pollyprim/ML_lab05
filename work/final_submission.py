"""Final pipeline: train ResNet-18 (pretrained) on train+val, predict test, save submission."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import transforms, models
from common import *

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/week5_data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("final_submission.csv")
IMG_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 128

train_table = pd.read_csv(DATA / "train.csv", dtype={"id": str})
val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})
test_table = pd.read_csv(DATA / "test.csv", dtype={"id": str})

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
geo_aug = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
           transforms.RandomChoice([transforms.RandomRotation((a,a)) for a in (0,90,180,270)]),
           transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
artifact_aug = [transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.5))], p=0.3),
                transforms.RandomInvert(p=0.1), transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)]
class AddGaussianNoise:
    def __init__(self, std=(0.01,0.06)): self.std = std
    def __call__(self, img):
        s = float(np.random.uniform(*self.std)); return torch.clamp(img + torch.randn_like(img)*s, 0, 1)
train_transform = transforms.Compose(geo_aug + artifact_aug + [transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD), AddGaussianNoise()])
eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD)])

def make_resnet(pretrained=True, unfreeze="all"):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if pretrained:
        for p in backbone.parameters(): p.requires_grad = False
        blocks = {"none":[], "layer4":[backbone.layer4], "layer3+4":[backbone.layer3, backbone.layer4], "all":[backbone]}[unfreeze]
        for blk in blocks:
            for p in blk.parameters(): p.requires_grad = True
    nf = backbone.fc.in_features; backbone.fc = nn.Identity()
    classifier = nn.Sequential(nn.Linear(nf, 256), nn.LeakyReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
    return nn.Sequential(backbone, classifier)

class ConcatDataset2(Dataset):
    """Concatenation of several ContestDataset objects (different folders)."""
    def __init__(self, datasets):
        self.datasets = datasets
        self.sizes = [len(d) for d in datasets]
        self.ids = sum([d.ids for d in datasets], [])
        self.labels = sum([d.labels for d in datasets], [])
    def __len__(self):
        return sum(self.sizes)
    def __getitem__(self, i):
        for d, s in zip(self.datasets, self.sizes):
            if i < s:
                return d[i]
            i -= s
        raise IndexError

seed_everything()
# combine train+val for final fit (allowed in the team phase per rules)
train_only_set = ContestDataset(train_table, DATA, "train", train_transform, IMG_SIZE)
val_as_train_set = ContestDataset(val_table, DATA, "val", train_transform, IMG_SIZE)
combined_set = ConcatDataset2([train_only_set, val_as_train_set])
model = make_resnet(pretrained=True, unfreeze="all")
history, best_acc, dt = train_model(model, "final", combined_set, val_as_train_set,
                                    epochs=25, lr=1e-4, weight_decay=1e-4, mixup_alpha=0.2, log_every=5)
# NOTE: val here is part of training; used only as a sanity monitor, not a held-out metric.
test_set = ContestDataset(test_table, DATA, "test", eval_transform, IMG_SIZE)
t0 = time.time()
preds = predict(model, test_set)
sub = pd.DataFrame({"id": test_set.ids, "label": preds})
assert list(sub.columns) == ["id", "label"]
assert sub.id.tolist() == test_table.id.tolist()
assert sub.label.between(0, NUM_CLASSES - 1).all()
sub.to_csv(OUT, index=False)
torch.save(model, "model_final.pt")
print(f"submission saved to {OUT} ({len(sub)} rows) in {time.time()-t0:.1f}s; train time {dt:.1f}s")
