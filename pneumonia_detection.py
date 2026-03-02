"""
pneumonia detection from chest x-rays
  1.1 - train resnet-18 from scratch
  1.2 - fine-tune pretrained resnet-18
"""

import os
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR   = "./chest_xray"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES    = ["NORMAL", "PNEUMONIA"]
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

print(f"device: {DEVICE}")


train_tf = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.ColorJitter(brightness=0.2, contrast=0.2),
  transforms.ToTensor(),
  transforms.Normalize(MEAN, STD),
])

eval_tf = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(MEAN, STD),
])


def get_loaders():
  splits = {
    "train": datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf),
    "val":   datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf),
    "test":  datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tf),
  }
  loaders = {
    k: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(k == "train"), num_workers=4, pin_memory=True)
    for k, ds in splits.items()
  }
  for k, ds in splits.items():
    print(f"{k}: {len(ds)}")
  return loaders["train"], loaders["val"], loaders["test"]


def build_resnet18(pretrained=False):
  weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
  model = models.resnet18(weights=weights)
  model.fc = nn.Linear(model.fc.in_features, 2)
  return model.to(DEVICE)


def train(model, train_loader, val_loader, tag="model"):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

  history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
  best_acc, best_weights = 0.0, copy.deepcopy(model.state_dict())

  for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    for phase, loader in (("train", train_loader), ("val", val_loader)):
      model.train() if phase == "train" else model.eval()
      loss_sum = correct = total = 0

      for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
          out   = model(imgs)
          loss  = criterion(out, labels)
          preds = out.argmax(1)
          if phase == "train":
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * len(imgs)
        correct  += (preds == labels).sum().item()
        total    += len(imgs)

      history[f"{phase}_loss"].append(loss_sum / total)
      history[f"{phase}_acc"].append(correct / total)

      if phase == "val" and (correct / total) > best_acc:
        best_acc = correct / total
        best_weights = copy.deepcopy(model.state_dict())

    scheduler.step()
    tl, ta = history["train_loss"][-1], history["train_acc"][-1]
    vl, va = history["val_loss"][-1],   history["val_acc"][-1]
    print(f"[{tag}] {epoch:02d}/{NUM_EPOCHS}  train {ta:.3f}/{tl:.4f}  val {va:.3f}/{vl:.4f}  ({time.time()-t0:.0f}s)")

  print(f"[{tag}] best val acc: {best_acc:.4f}")
  model.load_state_dict(best_weights)
  torch.save(model.state_dict(), f"{tag}_best.pth")
  return model, history


def evaluate(model, test_loader):
  model.eval()
  all_preds, all_labels = [], []
  misclassified = []

  with torch.no_grad():
    for imgs, labels in test_loader:
      imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
      preds = model(imgs).argmax(1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

      for i in (preds != labels).nonzero(as_tuple=True)[0]:
        misclassified.append((imgs[i].cpu(), labels[i].item(), preds[i].item()))

  acc = np.mean(np.array(all_preds) == np.array(all_labels))
  print(f"\ntest acc: {acc:.4f}")
  print(classification_report(all_labels, all_preds, target_names=CLASSES))
  return acc, all_preds, all_labels, misclassified


def denorm(t):
  img = t.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN)
  return np.clip(img, 0, 1)


def plot_loss_curves(h1, h2, out="loss_curves.png"):
  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  for ax, h, title in zip(axes, [h1, h2], ["1.1 – from scratch", "1.2 – fine-tuned"]):
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["train_loss"], label="train",      color="steelblue")
    ax.plot(ep, h["val_loss"],   label="validation", color="orange")
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=0.3)
  plt.tight_layout()
  plt.savefig(out, dpi=150); plt.close()


def plot_confusion_matrix(labels, preds, title, out):
  cm = confusion_matrix(labels, preds)
  fig, ax = plt.subplots(figsize=(6, 5))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
              xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
  ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
  plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_failures(misclassified, tag, n=6, out=None):
  samples = misclassified[:n]
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  fig.suptitle(f"failure cases – {tag}", fontsize=13)
  for ax, (img, true, pred) in zip(axes.flat, samples):
    ax.imshow(denorm(img))
    ax.set_title(f"true: {CLASSES[true]}  pred: {CLASSES[pred]}", fontsize=9, color="red")
    ax.axis("off")
  for ax in axes.flat[len(samples):]:
    ax.axis("off")
  plt.tight_layout()
  plt.savefig(out or f"failures_{tag}.png", dpi=150); plt.close()


if __name__ == "__main__":
  train_loader, val_loader, test_loader = get_loaders()

  print("\n--- 1.1  from scratch ---")
  m_scratch, h_scratch = train(build_resnet18(pretrained=False), train_loader, val_loader, tag="scratch")
  acc_s, preds_s, labels_s, fails_s = evaluate(m_scratch, test_loader)

  print("\n--- 1.2  fine-tuned ---")
  m_ft, h_ft = train(build_resnet18(pretrained=True), train_loader, val_loader, tag="finetune")
  acc_ft, preds_ft, labels_ft, fails_ft = evaluate(m_ft, test_loader)

  plot_loss_curves(h_scratch, h_ft)
  plot_confusion_matrix(labels_s,  preds_s,  "confusion – from scratch", "cm_scratch.png")
  plot_confusion_matrix(labels_ft, preds_ft, "confusion – fine-tuned",   "cm_finetune.png")
  plot_failures(fails_s,  "from scratch")
  plot_failures(fails_ft, "fine-tuned")

  print(f"\nscratch:    {acc_s:.4f}")
  print(f"fine-tuned: {acc_ft:.4f}")
