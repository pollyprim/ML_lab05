"""Shared utilities for the Week-5 satellite classification lab."""
import os, copy, random, time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
NUM_CLASSES = 20
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def image_file(data_dir, folder, image_id):
    name = image_id if str(image_id).endswith(".jpg") else f"{image_id}.jpg"
    return data_dir / folder / name

class ContestDataset(Dataset):
    """In-memory dataset. Images are kept as uint8 numpy arrays (cheap RAM-wise);
    PIL tensors are produced on the fly inside the transform pipeline."""
    def __init__(self, table, data_dir, folder, transform, img_size, resize=True):
        self.ids = table.id.astype(str).tolist()
        self.labels = table.label.tolist() if "label" in table.columns else [-1] * len(self.ids)
        self.transform = transform
        self.resize = resize
        self.img_size = img_size
        self.images = []
        for image_id in tqdm(self.ids, desc=f"loading {folder}", leave=False):
            with Image.open(image_file(data_dir, folder, image_id)) as img:
                img = img.convert("RGB")
                if resize:
                    img = img.resize((img_size, img_size), Image.BOX)
                self.images.append(np.asarray(img, dtype=np.uint8))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        return self.transform(img), self.labels[i]

criterion = nn.CrossEntropyLoss()

def make_loader(dataset, shuffle):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator,
                      num_workers=0, drop_last=False)

@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in make_loader(dataset, shuffle=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * len(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    return total_loss / len(dataset), correct / len(dataset)

@torch.no_grad()
def predict(model, dataset):
    model.eval()
    predictions = []
    for images, _ in make_loader(dataset, shuffle=False):
        outputs = model(images.to(DEVICE))
        predictions.append(outputs.argmax(dim=1).cpu())
    return torch.cat(predictions).numpy()

@torch.no_grad()
def predict_proba(model, dataset):
    model.eval()
    probs = []
    for images, _ in make_loader(dataset, shuffle=False):
        outputs = model(images.to(DEVICE))
        probs.append(torch.softmax(outputs, dim=1).cpu())
    return torch.cat(probs).numpy()

def train_model(model, name, train_set, val_set, epochs=30, lr=1e-3, log_every=5, weight_decay=0.0,
                mixup_alpha=0.0):
    seed_everything()
    model = model.to(DEVICE)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = make_loader(train_set, shuffle=True)

    history, best_acc, best_weights = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(images.size(0), device=images.device)
                mixed = lam * images + (1 - lam) * images[perm]
                outputs = model(mixed)
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[perm])
                correct += (outputs.argmax(dim=1) == labels).sum().item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        scheduler.step()
        train_loss = total_loss / len(train_set)
        train_acc = correct / len(train_set)
        val_loss, val_acc = evaluate(model, val_set)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc": train_acc, "val_acc": val_acc})
        if val_acc > best_acc:
            best_acc, best_weights = val_acc, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"{name} | epoch {epoch:2d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | "
                  f"train acc {train_acc:.3f} | val acc {val_acc:.3f}", flush=True)
    model.load_state_dict(best_weights)
    dt = time.time() - t0
    print(f"{name}: best validation accuracy {best_acc:.3f} | training time {dt:.1f}s", flush=True)
    return pd.DataFrame(history), best_acc, dt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
