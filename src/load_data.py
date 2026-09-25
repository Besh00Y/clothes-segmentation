import numpy as np
from src.dataset import load_dataset
from torch.utils.data import Dataset
 
def load_atr_splits(dataset_name: str = "mattmdjaga/human_parsing_dataset"):

    raw = load_dataset(dataset_name)
    full = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
 
    full = full.train_test_split(test_size=0.30, seed=42)
    train_split = full["train"]
 
    rest = full["test"].train_test_split(test_size=0.50, seed=42)
    val_split, test_split = rest["train"], rest["test"]
 
    return train_split, val_split, test_split


if __name__ == "__main__":
    train_split, val_split, test_split = load_atr_splits()
    print(f"Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

    train_split.save_to_disk(r"data\Images\train")
    val_split.save_to_disk(r"data\Images\val")
    test_split.save_to_disk(r"data\Images\test")

    sample_image = np.array(train_split[0]["image"].convert("RGB"))
    print(f"Image shape: {sample_image.shape}")
