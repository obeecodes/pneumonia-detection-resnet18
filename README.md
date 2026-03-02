# Pneumonia Detection Using Chest X Ray Images
CAP 5516 Medical Image Computing | Assignment 1

Binary classification of chest X ray images (NORMAL vs. PNEUMONIA) using ResNet 18.

- **Task 1.1** Train ResNet 18 from scratch
- **Task 1.2** Fine tune ImageNet pre trained ResNet 18

---

## Dataset

Download from Kaggle: [Chest X Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The easiest way to download is via kagglehub:

```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)
```

The dataset should have the following structure:

```
chest_xray/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

Update `DATA_DIR` in `pneumonia_detection.py` to point to your local dataset path.

---

## Setup

```bash
pip install torch torchvision matplotlib seaborn scikit-learn kagglehub
```

---

## Running the Code

```bash
python3 pneumonia_detection.py
```

Both tasks run sequentially. Training will automatically use a GPU if one is available, otherwise it falls back to CPU. On a GPU each epoch takes around 40 seconds; on CPU expect significantly longer.

---

## Output Files

| File | Description |
|------|-------------|
| `scratch_best.pth` | Best model weights for Task 1.1 |
| `finetune_best.pth` | Best model weights for Task 1.2 |
| `loss_curves.png` | Training and validation loss curves |
| `cm_scratch.png` | Confusion matrix for Task 1.1 |
| `cm_finetune.png` | Confusion matrix for Task 1.2 |
| `failures_from scratch.png` | Misclassified examples for Task 1.1 |
| `failures_fine-tuned.png` | Misclassified examples for Task 1.2 |

---

## Results

| Task | Description | Test Accuracy |
|------|-------------|---------------|
| 1.1 | ResNet 18 from scratch | 80.61% |
| 1.2 | ResNet 18 fine tuned (ImageNet) | 82.37% |

---

## Model Architecture

ResNet 18 with the final fully connected layer replaced:

```python
model = models.resnet18(weights=...)
model.fc = nn.Linear(model.fc.in_features, 2)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input size | 224 x 224 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam (lr=1e-3, weight decay=1e-4) |
| LR Scheduler | StepLR (step size=7, gamma=0.1) |
| Loss | Cross Entropy |
