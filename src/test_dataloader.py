from datasets import load_from_disk
from torch.utils.data import DataLoader

from dataset import ATRClothesDataset
from transforms import get_train_transforms, get_val_transforms


# Load splits
train_split = load_from_disk(r"data\Images\train")
val_split = load_from_disk(r"data\Images\val")
test_split = load_from_disk(r"data\Images\test")


# Create PyTorch datasets
train_dataset = ATRClothesDataset(
    train_split,
    transform=get_train_transforms()
)

val_dataset = ATRClothesDataset(
    val_split,
    transform=get_val_transforms()
)

test_dataset = ATRClothesDataset(
    test_split,
    transform=get_val_transforms()
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

images, masks = next(iter(train_loader))


print("=" * 50)
print("DataLoader Test")
print("=" * 50)

print(f"Images shape: {images.shape}")
print(f"Masks shape:  {masks.shape}")

print(f"Images dtype: {images.dtype}")
print(f"Masks dtype:  {masks.dtype}")

print(f"Mask values: {masks.unique()}")