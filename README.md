# Clothes Segmentation

A binary semantic segmentation model based on **U-Net with a ResNet34 encoder** that separates clothes from non-clothes regions in images.

This project was developed for a clothes segmentation assessment and includes dataset preparation, preprocessing, model training, evaluation, error analysis, and inference on new images.

See [`report.md`](report.md) for the full report, including the dataset choice, architecture, loss function, evaluation results, and limitations.

### Test Set Results

| Metric    |  Score |
| --------- | -----: |
| IoU       | 0.8974 |
| Dice      | 0.9443 |
| Precision | 0.9419 |
| Recall    | 0.9497 |

[View sample predictions](outputs/test_predictions_best.png)

---

## Project Structure

```text
Clothing Segmentation/
├── req.txt
├── report.md
├── README.md
├── .gitignore
├── src/
│   ├── dataset.py              # Downloads/prepares ATR dataset and creates binary masks
│   ├── load_data.py            # Dataset loading helper
│   ├── test_dataloader.py      # DataLoader sanity check
│   ├── test_Cuda.py            # Checks GPU / CUDA compatibility
│   ├── analyze_dataset.py      # Dataset statistics and sample visualization
│   ├── transforms.py           # Preprocessing and augmentation pipelines
│   ├── model.py                # U-Net with ResNet34 encoder
│   ├── losses.py               # BCE + Dice combined loss
│   ├── metrics.py              # IoU, Dice, Precision, Recall
│   ├── train.py                # Training, validation, and checkpointing
│   ├── evaluate.py             # Threshold optimization and test evaluation
│   ├── visualize_test.py       # Random and best-case prediction grids
│   └── inference.py            # Inference on new images
├── data/
│   └── Images/
│       ├── train/
│       ├── val/
│       └── test/
├── outputs/
│   ├── test_predictions_random.png
│   ├── test_predictions_best.png
│   ├── error_analysis_worst_cases.png
│   ├── inference/
│   │   ├── image3_comparison.png
│   │   └── image4_comparison.png
│   └── checkpoints/
│       └── best_model.pth
```

> `data/`, `outputs/checkpoints/`, and most of `outputs/inference/` are not committed to Git because of their size (see `.gitignore`). The two comparison images referenced in `report.md` are kept as an explicit exception.

---

## 1. Setup

### 1.1 Create the environment

```bash
conda create -n clothes-seg python=3.11 -y
conda activate clothes-seg
```

### 1.2 Install PyTorch

Install a PyTorch build compatible with your GPU.

For NVIDIA GPUs supporting CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For other GPUs or CUDA versions, use the appropriate installation command from the official PyTorch installation guide.

PyTorch and torchvision are intentionally not included in `req.txt` to avoid accidentally replacing a GPU-enabled installation with an incompatible CPU-only build.

### 1.3 Install the remaining dependencies

```bash
pip install -r req.txt
```

### 1.4 Verify the environment

```bash
python src/test_Cuda.py
```

This checks whether CUDA is available and performs a real GPU operation to verify that the installed PyTorch build can execute CUDA operations successfully.

---

## 2. Prepare the Dataset

The project uses the **ATR Human Parsing Dataset** from Hugging Face:

`mattmdjaga/human_parsing_dataset`

The original dataset contains human parsing annotations with multiple semantic classes. For this project, the labels are converted into a binary segmentation task:

* `0` → Not Clothes
* `1` → Clothes

The dataset is split into:

* 70% training
* 15% validation
* 15% testing

using a fixed random seed of `42`.

Run:

```bash
python src/dataset.py
```

The processed splits are saved under:

```text
data/Images/
```

### Optional Dataset Analysis

To inspect the dataset statistics and sample masks:

```bash
python src/analyze_dataset.py
```

---

## 3. Model

The segmentation model is a **U-Net with a pretrained ResNet34 encoder**.

### Encoder

The ResNet34 encoder is initialized with ImageNet pretrained weights.

### Decoder

The decoder progressively upsamples the feature maps and combines them with encoder features through skip connections.

This architecture was selected because it provides a good balance between segmentation quality, model capacity, and computational cost.

Input resolution:

```text
256 × 256
```

Output:

```text
1 × 256 × 256
```

where each pixel represents the probability of belonging to the clothes class.

---

## 4. Loss Function

The model uses a combined **BCE + Dice loss**:

```text
Loss = 0.5 × BCE + 0.5 × Dice Loss
```

### Binary Cross Entropy

BCE provides pixel-level classification supervision.

### Dice Loss

Dice loss directly focuses on the overlap between the predicted mask and the ground-truth mask, which is useful for segmentation tasks.

The combination allows the model to learn both pixel-wise classification and region overlap.

---

## 5. Training

Run:

```bash
python src/train.py
```

Training configuration:

* Epochs: `20`
* Batch size: `8`
* Optimizer: `AdamW`
* Initial learning rate: `1e-4`
* Weight decay: `1e-4`
* Scheduler: `ReduceLROnPlateau`
* Loss: `0.5 × BCE + 0.5 × Dice`

The best checkpoint is selected according to validation Dice and saved to:

```text
outputs/checkpoints/best_model.pth
```

The final training run achieved approximately:

```text
Validation Dice: 0.9478
Validation IoU:  0.9008
```

---

## 6. Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluation pipeline:

1. Searches for the best prediction threshold using the validation set.
2. Evaluates the selected threshold on the held-out test set.
3. Calculates IoU, Dice, Precision, and Recall.
4. Generates error-analysis visualizations for difficult test examples.

The best validation threshold was:

```text
0.45
```

### Final Test Results

| Metric    |  Score |
| --------- | -----: |
| IoU       | 89.74% |
| Dice      | 94.43% |
| Precision | 94.19% |
| Recall    | 94.97% |

Test set size:

```text
2,656 images
```

---

## 7. Test Set Visualization

To generate random and best-performing prediction examples:

```bash
python src/visualize_test.py
```

Generated visualizations include:

```text
outputs/test_predictions_random.png
outputs/test_predictions_best.png
```

The error-analysis visualization is generated by:

```bash
python src/evaluate.py
```

and saved as:

```text
outputs/error_analysis_worst_cases.png
```

---

## 8. Inference on a New Image

The trained model can be used to segment clothes in a new image.

```bash
python src/inference.py --image path/to/your/photo.jpg
```

The default threshold is `0.45`. It can also be changed manually:

```bash
python src/inference.py --image path/to/your/photo.jpg --threshold 0.45
```

The output directory can be changed using:

```bash
python src/inference.py \
    --image path/to/your/photo.jpg \
    --output_dir outputs/inference
```

The inference pipeline produces:

```text
<name>_mask.png
<name>_overlay.png
<name>_comparison.png
```

---

## 9. Error Analysis

The model performs well on common clothing regions and typical human poses.

However, some challenging cases were observed during test-set analysis.

Common failure cases include:

* Small or distant people
* Occluded people
* Unusual poses
* Clothing partially covered by hair
* Small disconnected clothing regions
* Confusion between clothes and nearby objects or accessories

These cases are discussed in more detail in [`report.md`](report.md).

[View error analysis](outputs/error_analysis_worst_cases.png)

---

## 10. Strengths

* Good segmentation performance on the held-out ATR test set.
* Uses a pretrained ResNet34 encoder.
* Combines BCE and Dice loss.
* Uses data augmentation during training.
* Includes validation-based threshold selection.
* Includes quantitative evaluation and qualitative error analysis.
* Supports inference on new images.

---

## 11. Limitations

The model is trained on the ATR Human Parsing Dataset, so its performance may decrease on images that differ significantly from the training data.

Potential limitations include:

* Unusual poses
* Heavy occlusion
* Very small people
* Complex backgrounds
* Accessories that are visually similar to clothing
* Small or disconnected clothing regions

The model also performs **binary semantic segmentation**, meaning it identifies clothes as a single class. It does not distinguish individual clothing categories such as shirts, pants, shoes, or dresses in the final output.

---

## 12. Reproducibility

The dataset split uses a fixed random seed:

```text
seed = 42
```

To reproduce the experiment:

```bash
python src/dataset.py
python src/train.py
python src/evaluate.py
```

Minor differences between runs may occur because of GPU-related nondeterminism.

The reported results should therefore be considered reproducible within a small numerical variation rather than guaranteed to be bit-for-bit identical.

---

## 13. Hardware

The project was developed and tested using:

```text
GPU: NVIDIA RTX 5060
VRAM: 8 GB
Input resolution: 256 × 256
Batch size: 8
```

The selected configuration fits within the available GPU memory and was sufficient for training the model on the approximately 12.4k-image training split.

---

## 14. Future Improvements

Possible improvements include:

* Stronger augmentation strategies
* Higher input resolution
* Experimenting with other segmentation architectures
* Better handling of small clothing regions
* Class-specific clothing segmentation
* More diverse external datasets
* Post-processing techniques for cleaner boundaries
* Instance-level clothing segmentation
