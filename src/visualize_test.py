import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_from_disk

from dataset import ATRClothesDataset
from transforms import get_val_transforms
from model import UNetResNet34

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, "data", "Images", "test")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

BATCH_SIZE = 8
THRESHOLD = 0.45  
N_SAMPLES = 6
SEED = 42


def load_model():
    model = UNetResNet34(pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_dice={checkpoint['val_dice']:.4f})")
    return model


def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def run_inference_on_test(model, test_loader):

    records = []
    with torch.no_grad():
        for images, masks in test_loader:
            images_gpu = images.to(DEVICE)
            logits = model(images_gpu)
            probs = torch.sigmoid(logits)
            preds = (probs >= THRESHOLD).float().cpu()

            masks_f = masks.float().unsqueeze(1)
            eps = 1e-6
            for i in range(images.shape[0]):
                p, t = preds[i:i + 1], masks_f[i:i + 1]
                intersection = (p * t).sum()
                union = ((p + t) > 0).float().sum()
                iou = ((intersection + eps) / (union + eps)).item()
                records.append((images[i], masks[i], preds[i, 0], iou))
    return records


def save_grid(records, save_path, title):
    fig, axes = plt.subplots(len(records), 3, figsize=(9, 3 * len(records)))
    if len(records) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (image, gt_mask, pred_mask, iou) in enumerate(records):
        image_np = unnormalize(image)
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title("Original Image")
        axes[row, 1].imshow(gt_mask.numpy(), cmap="Greens")
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(pred_mask.numpy(), cmap="Reds")
        axes[row, 2].set_title(f"Prediction (IoU={iou:.2f})")
        for col in range(3):
            axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


def main():
    print(f"Device: {DEVICE}")
    model = load_model()

    test_split = load_from_disk(TEST_PATH)
    test_dataset = ATRClothesDataset(test_split, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Running inference on {len(test_dataset)} test images...")
    records = run_inference_on_test(model, test_loader)

    random.seed(SEED)
    random_sample = random.sample(records, N_SAMPLES)
    save_grid(
        random_sample,
        os.path.join(OUTPUT_DIR, "test_predictions_random.png"),
        "Random Test Samples (typical performance)",
    )

    best_sample = sorted(records, key=lambda r: r[3], reverse=True)[:N_SAMPLES]
    save_grid(
        best_sample,
        os.path.join(OUTPUT_DIR, "test_predictions_best.png"),
        "Best Test Predictions (strengths)",
    )

    mean_iou = np.mean([r[3] for r in records])
    print(f"\nMean test IoU across all {len(records)} samples: {mean_iou:.4f}")


if __name__ == "__main__":
    main()