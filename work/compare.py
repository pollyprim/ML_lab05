"""Experiment 1: MLP vs simple CNN vs advanced CNN vs pretrained ResNet-18 (64px).
Run: python compare.py <which>   where which in {mlp, cnn, advcnn, resnet}"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torchvision import transforms, models
from common import *

DATA = Path("/workspace/work/data")
train_table = pd.read_csv(DATA / "train.csv", dtype={"id": str})
val_table = pd.read_csv(DATA / "val.csv", dtype={"id": str})

MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
IMG_SIZE = 64

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([transforms.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def make_models():
    mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * IMG_SIZE * IMG_SIZE, 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * (IMG_SIZE // 2) ** 2, 128), nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),
    )
    advcnn = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),   # 32
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),   # 16
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),            # 1
        nn.Flatten(),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, NUM_CLASSES),
    )
    return {"mlp": mlp, "cnn": cnn, "adv_cnn": advcnn}

def make_resnet():
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.layer4.parameters():
        p.requires_grad = True
    nf = backbone.fc.in_features
    backbone.fc = nn.Identity()
    classifier = nn.Sequential(nn.Linear(nf, 256), nn.LeakyReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
    return nn.Sequential(backbone, classifier)

which = sys.argv[1]
seed_everything()
train_set = ContestDataset(train_table, DATA, "train", train_transform, IMG_SIZE)
val_set = ContestDataset(val_table, DATA, "val", eval_transform, IMG_SIZE)

if which == "resnet":
    model = make_resnet()
    epochs, lr = 15, 3e-4
else:
    model = make_models()[which]
    epochs, lr = 30, 1e-3

print(f"model={which} trainable_params={count_parameters(model)} device={DEVICE}")
history, best_acc, dt = train_model(model, which, train_set, val_set, epochs=epochs, lr=lr)
history.to_csv(f"history_{which}.csv", index=False)
torch.save(model.state_dict(), f"model_{which}.pt")
# final accuracy with restored best weights on held-out val
_, final_acc = evaluate(model, val_set)
json.dump({"model": which, "best_val_acc": best_acc, "final_val_acc": final_acc,
           "train_time_s": round(dt, 1), "params": count_parameters(model),
           "epochs": epochs, "lr": lr, "img_size": IMG_SIZE},
          open(f"result_{which}.json", "w"), indent=1)
print("DONE", which, best_acc, final_acc, dt)
