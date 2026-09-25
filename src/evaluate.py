import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from datasets import load_from_disk

from dataset import ATRClothesDataset
from transforms import get_val_transforms
from model import UNetResNet34

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Images",
    "train"
)

VAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Images",
    "val"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Images",
    "test"
)

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "checkpoints",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

BATCH_SIZE = 8

THRESHOLDS_TO_TRY = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

N_WORST_TO_SHOW = 6
def load_model():

    print("\nLoading best model...")
    model = UNetResNet34(pretrained=False)

    checkpoint = torch.load( CHECKPOINT_PATH, map_location=DEVICE )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best validation Dice: "
        f"{checkpoint['val_dice']:.4f}"
    )

    return model

def calculate_iou( pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6 ):

    intersection = (
        pred * target
    ).sum(dim=(1, 2, 3))

    union = (
        (pred + target) > 0
    ).float().sum(dim=(1, 2, 3))

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return iou


def calculate_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
):
    intersection = (
        pred * target
    ).sum(dim=(1, 2, 3))

    dice = (
        2 * intersection + eps
    ) / (
        pred.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
        + eps
    )

    return dice


def calculate_precision(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
):

    intersection = (
        pred * target
    ).sum(dim=(1, 2, 3))

    false_positive = (
        pred.sum(dim=(1, 2, 3))
        - intersection
    )

    precision = (
        intersection + eps
    ) / (
        intersection
        + false_positive
        + eps
    )

    return precision


def calculate_recall(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6
):
    intersection = (
        pred * target
    ).sum(dim=(1, 2, 3))

    false_negative = (
        target.sum(dim=(1, 2, 3))
        - intersection
    )

    recall = (
        intersection + eps
    ) / (
        intersection
        + false_negative
        + eps
    )

    return recall


def find_best_threshold(
    model,
    val_loader
):

    print("\n" + "=" * 60)
    print("Threshold Optimization")
    print("Using Validation Set")
    print("=" * 60)

    all_probs = []
    all_targets = []

    model.eval()

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)

            masks = (
                masks
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            logits = model(images)

            probs = torch.sigmoid(logits)

            all_probs.append(
                probs.cpu()
            )

            all_targets.append(
                masks.cpu()
            )

    # Combine all validation batches
    probs = torch.cat(
        all_probs,
        dim=0
    )

    targets = torch.cat(
        all_targets,
        dim=0
    )

    best_threshold = 0.5
    best_iou = -1.0

    threshold_results = []

    for threshold in THRESHOLDS_TO_TRY:

        predictions = (
            probs >= threshold
        ).float()

        ious = calculate_iou(
            predictions,
            targets
        )

        mean_iou = ious.mean().item()

        threshold_results.append(
            (threshold, mean_iou)
        )

        print(
            f"Threshold = {threshold:.2f} "
            f"| Val IoU = {mean_iou:.4f}"
        )

        if mean_iou > best_iou:

            best_iou = mean_iou
            best_threshold = threshold

    print("\nBest threshold:")
    print(
        f"Threshold = {best_threshold:.2f}"
    )
    print(
        f"Validation IoU = {best_iou:.4f}"
    )

    return best_threshold


def evaluate_on_test(
    model,
    test_loader,
    threshold
):

    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print(
        f"Threshold = {threshold:.2f}"
    )
    print("=" * 60)

    all_ious = []
    all_dices = []
    all_precisions = []
    all_recalls = []

    per_image_records = []

    model.eval()

    with torch.no_grad():

        for batch_idx, (
            images,
            masks
        ) in enumerate(test_loader):

            images_gpu = images.to(DEVICE)

            masks_gpu = (
                masks
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            # Model prediction
            logits = model(
                images_gpu
            )

            probs = torch.sigmoid(
                logits
            )

            predictions = (
                probs >= threshold
            ).float()

            # Calculate metrics for each image
            batch_ious = calculate_iou(
                predictions,
                masks_gpu
            )

            batch_dices = calculate_dice(
                predictions,
                masks_gpu
            )

            batch_precisions = calculate_precision(
                predictions,
                masks_gpu
            )

            batch_recalls = calculate_recall(
                predictions,
                masks_gpu
            )

            all_ious.extend(
                batch_ious.cpu().numpy()
            )

            all_dices.extend(
                batch_dices.cpu().numpy()
            )

            all_precisions.extend(
                batch_precisions.cpu().numpy()
            )

            all_recalls.extend(
                batch_recalls.cpu().numpy()
            )

            # Save images for error analysis
            for i in range(
                images.shape[0]
            ):

                per_image_records.append(
                    (
                        images[i].cpu(),
                        masks[i].cpu(),
                        predictions[i, 0].cpu(),
                        batch_ious[i].item()
                    )
                )

            if (
                batch_idx + 1
            ) % 50 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(test_loader)} batches"
                )

    # Final test metrics
    mean_iou = np.mean(
        all_ious
    )

    mean_dice = np.mean(
        all_dices
    )

    mean_precision = np.mean(
        all_precisions
    )

    mean_recall = np.mean(
        all_recalls
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(
        f"Test IoU:       {mean_iou:.4f}"
    )

    print(
        f"Test Dice:      {mean_dice:.4f}"
    )

    print(
        f"Test Precision: {mean_precision:.4f}"
    )

    print(
        f"Test Recall:    {mean_recall:.4f}"
    )

    print("=" * 60)

    return per_image_records


def unnormalize(image_tensor):

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    image = (
        image_tensor * std
        + mean
    )

    image = image.clamp(
        0,
        1
    )

    image = image.permute(
        1,
        2,
        0
    )

    return image.numpy()

def save_error_analysis(
    per_image_records,
    save_path,
    n_worst=N_WORST_TO_SHOW
):

    print("\n" + "=" * 60)
    print("Error Analysis")
    print("=" * 60)

    # Sort images by IoU
    worst_cases = sorted(
        per_image_records,
        key=lambda record: record[3]
    )[:n_worst]

    fig, axes = plt.subplots(
        n_worst,
        3,
        figsize=(9, 3 * n_worst)
    )

    # Handle case where n_worst = 1
    if n_worst == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, (
        image,
        ground_truth,
        prediction,
        iou
    ) in enumerate(worst_cases):

        image_np = unnormalize(
            image
        )

        axes[row, 0].imshow(
            image_np
        )

        axes[row, 0].set_title(
            "Original Image"
        )

        axes[row, 1].imshow(
            ground_truth.numpy(),
            cmap="Greens"
        )

        axes[row, 1].set_title(
            "Ground Truth"
        )

        axes[row, 2].imshow(
            prediction.numpy(),
            cmap="Reds"
        )

        axes[row, 2].set_title(
            f"Prediction\nIoU = {iou:.2f}"
        )

        for col in range(3):

            axes[row, col].axis(
                "off"
            )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved error analysis grid:"
    )

    print(
        save_path
    )


def main():

    print("=" * 60)
    print("Clothes Segmentation Evaluation")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    model = load_model()

    print("\nLoading datasets...")

    val_split = load_from_disk(
        VAL_PATH
    )

    test_split = load_from_disk(
        TEST_PATH
    )

    print(
        f"Validation samples: "
        f"{len(val_split)}"
    )

    print(
        f"Test samples: "
        f"{len(test_split)}"
    )


    val_dataset = ATRClothesDataset(
        val_split,
        transform=get_val_transforms()
    )

    test_dataset = ATRClothesDataset(
        test_split,
        transform=get_val_transforms()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    best_threshold = find_best_threshold(
        model,
        val_loader
    )


    per_image_records = evaluate_on_test(
        model,
        test_loader,
        best_threshold
    )


    error_analysis_path = os.path.join(
        OUTPUT_DIR,
        "error_analysis_worst_cases.png"
    )

    save_error_analysis(
        per_image_records,
        error_analysis_path,
        N_WORST_TO_SHOW
    )

    print("\n" + "=" * 60)
    print("Evaluation Completed Successfully")
    print("=" * 60)

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Error analysis saved to:"
    )

    print(
        error_analysis_path
    )

if __name__ == "__main__":
    main()