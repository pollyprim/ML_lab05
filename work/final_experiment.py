"""Final experiment: ResNet-18 pretrained, robust augmentation (blur/noise/inverse), 128px.
Run: python final_experiment.py <mode: train|finetune|scratch> [epochs]
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import transforms, models
from common import *

DATA = Path("/workspace/work/data")   # will be repointed to real data
train_table = pd.read_csv(DATA / "train.csv", dtype={"id": str})
val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})

IMG_SIZE = 128
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# base geometric/color aug (aerial: no "up" direction)
geo_aug = [
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
]
# artifact simulation matching val/test corruption: blur, noise, color inverse, JPEG
artifact_aug = [
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
    transforms.RandomInvert(p=0.1),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
]

class AddGaussianNoise:
    """Custom augmentation: additive Gaussian noise (simulates sensor artifacts)."""
    def __init__(self, std=(0.01, 0.06)):
        self.std = std
    def __call__(self, img):  # img: FloatTensor in [0,1]
        s = float(np.random.uniform(*self.std))
        return torch.clamp(img + torch.randn_like(img) * s, 0, 1)
    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"

train_transform = transforms.Compose(geo_aug + artifact_aug + [
    transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD), AddGaussianNoise()])
eval_transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(IMNET_MEAN, IMNET_STD)])

def make_resnet(pretrained=True, unfreeze="layer4"):
    if pretrained:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = models.resnet18(weights=None)
    if pretrained:
        for p in backbone.parameters():
            p.requires_grad = False
        target = {"none": [], "layer4": [backbone.layer4], "layer3+4": [backbone.layer3, backbone.layer4],
                  "all": [backbone]}[unfreeze]
        for blk in target:
            for p in blk.parameters():
                p.requires_grad = True
    nf = backbone.fc.in_features
    backbone.fc = nn.Identity()
    classifier = nn.Sequential(nn.Linear(nf, 256), nn.LeakyReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
    return nn.Sequential(backbone, classifier)

mode = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
seed_everything()
train_set = ContestDataset(train_table, DATA, "train", train_transform, IMG_SIZE)
val_set = ContestDataset(val_table, DATA, "val", eval_transform, IMG_SIZE)

if mode == "scratch":
    model = make_resnet(pretrained=False)
    lr, wd, mixup = 1e-3, 1e-4, 0.2
elif mode == "finetune":
    model = make_resnet(pretrained=True, unfreeze="all")
    lr, wd, mixup = 1e-4, 1e-4, 0.2
else:  # train = head+layer4 only
    model = make_resnet(pretrained=True, unfreeze="layer4")
    lr, wd, mixup = 3e-4, 1e-4, 0.2

print(f"mode={mode} trainable={count_parameters(model)} epochs={epochs} lr={lr}")
history, best_acc, dt = train_model(model, mode, train_set, val_set, epochs=epochs, lr=lr,
                                    weight_decay=wd, mixup_alpha=mixup, log_every=2)
history.to_csv(f"history_{mode}.csv", index=False)
torch.save(model.state_dict(), f"model_{mode}.pt")
_, final_acc = evaluate(model, val_set)
json.dump({"mode": mode, "best_val_acc": best_acc, "final_val_acc": final_acc,
           "train_time_s": round(dt,1), "params": count_parameters(model),
           "epochs": epochs, "lr": lr, "img_size": IMG_SIZE}, open(f"result_{mode}.json","w"), indent=1)
print("DONE", mode, best_acc, final_acc, dt)
