import os
import argparse

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from model import UNetResNet34
from transforms import get_val_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pth")
DEFAULT_THRESHOLD = 0.45  # from evaluate.py's threshold optimization step


def load_model():
    model = UNetResNet34(pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def segment_image(model, image_path: str, threshold: float = DEFAULT_THRESHOLD):

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = original_rgb.shape[:2]


    transform = get_val_transforms()
    transformed = transform(image=original_rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)  # (1, 3, 256, 256)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (256, 256)


    mask_256 = (probs >= threshold).astype(np.uint8)
    binary_mask = cv2.resize(
        mask_256, (original_w, original_h), interpolation=cv2.INTER_NEAREST
    )

    return original_rgb, binary_mask


def save_outputs(original_rgb, binary_mask, output_dir: str, base_name: str):
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, binary_mask * 255)

    overlay = original_rgb.copy()
    overlay[binary_mask == 1] = (
        0.5 * overlay[binary_mask == 1] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Clothes mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved mask       -> {mask_path}")
    print(f"Saved overlay    -> {overlay_path}")
    print(f"Saved comparison -> {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Run clothes segmentation on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                         help=f"Segmentation threshold (default: {DEFAULT_THRESHOLD}, "
                              f"from evaluate.py's threshold optimization).")
    parser.add_argument("--output_dir", default=os.path.join(BASE_DIR, "outputs", "inference"),
                         help="Where to save the mask/overlay/comparison images.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = load_model()

    print(f"Running inference on: {args.image}")
    original_rgb, binary_mask = segment_image(model, args.image, args.threshold)

    clothes_ratio = binary_mask.mean()
    print(f"Clothes pixel coverage: {clothes_ratio:.1%} of the image")

    base_name = os.path.splitext(os.path.basename(args.image))[0]
    save_outputs(original_rgb, binary_mask, args.output_dir, base_name)


if __name__ == "__main__":
    main()