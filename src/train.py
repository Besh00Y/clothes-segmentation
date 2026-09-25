import os

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from dataset import ATRClothesDataset
from transforms import get_train_transforms, get_val_transforms
from model import UNetResNet34
from losses import BCEDiceLoss
from metrics import get_metrics, update_metrics, compute_metrics, reset_metrics


BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
TRAIN_PATH = r"data\Images\train"
VAL_PATH = r"data\Images\val"
CHECKPOINT_DIR = r"outputs\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_one_epoch(model,loader,criterion,optimizer,metrics):

    model.train()
    total_loss = 0.0
    reset_metrics(metrics)
    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.float().unsqueeze(1).to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        update_metrics(metrics,logits.detach(),masks)

    avg_loss = total_loss / len(loader)
    results = compute_metrics(metrics)

    return avg_loss, results

def validate( model, loader, criterion, metrics):

    model.eval()
    total_loss = 0.0
    reset_metrics(metrics)
    with torch.no_grad():

        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.float().unsqueeze(1).to(DEVICE)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            update_metrics(metrics,logits, masks)

    avg_loss = total_loss / len(loader)
    results = compute_metrics(metrics)

    return avg_loss, results

def main():

    print("=" * 60)
    print("Clothes Segmentation Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print("\nLoading dataset...")

    train_split = load_from_disk(TRAIN_PATH)
    val_split = load_from_disk(VAL_PATH)

    train_dataset = ATRClothesDataset(
        train_split,
        transform=get_train_transforms()
    )

    val_dataset = ATRClothesDataset(
        val_split,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    print("\nCreating model...")

    model = UNetResNet34(
        pretrained=True
    )

    model = model.to(DEVICE)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    train_metrics = get_metrics(DEVICE)
    val_metrics = get_metrics(DEVICE)

    best_val_dice = 0.0
    print("\nStarting training...\n")

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_results = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            train_metrics
        )

        val_loss, val_results = validate(
            model,
            val_loader,
            criterion,
            val_metrics
        )

        scheduler.step( val_results["dice"] )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_results['dice']:.4f} | "
            f"Train IoU: {train_results['iou']:.4f}"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Dice:   {val_results['dice']:.4f} | "
            f"Val IoU:    {val_results['iou']:.4f}"
        )

        print(
            f"Val Precision: {val_results['precision']:.4f} | "
            f"Val Recall:    {val_results['recall']:.4f}"
        )

        print(
            f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_results["dice"] > best_val_dice:

            best_val_dice = val_results["dice"]
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                },
                checkpoint_path
            )

            print(
                f"✓ Best model saved: "
                f"{checkpoint_path}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()