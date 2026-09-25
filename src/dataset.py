import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset

ATR_LABELS = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upperclothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "leftshoe",
    10: "rightshoe",
    11: "face",
    12: "leftleg",
    13: "rightleg",
    14: "leftarm",
    15: "rightarm",
    16: "bag",
    17: "scarf",
}

CLOTHES_CLASS_IDS = {
    1,   # hat
    4,   # upperclothes
    5,   # skirt
    6,   # pants
    7,   # dress
    8,   # belt
    9,   # leftshoe
    10,  # rightshoe
    16,  # bag
    17,  # scarf
}


def mask_to_binary(mask: np.ndarray) -> np.ndarray:

    binary_mask = np.isin( mask,list(CLOTHES_CLASS_IDS)).astype(np.uint8)

    return binary_mask


class ATRClothesDataset(Dataset):

    def __init__(self, hf_split, transform=None):

        self.data = hf_split
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        image = np.array(item["image"].convert("RGB"))
        raw_mask = np.array(item["mask"].convert("L"))
        binary_mask = mask_to_binary(raw_mask)
        if self.transform is not None:

            transformed = self.transform(image=image, mask=binary_mask)
            image = transformed["image"]
            binary_mask = transformed["mask"]

        return image, binary_mask

def load_atr_dataset( train_path="data/Images/train", val_path="data/Images/val", test_path="data/Images/test",):
    
    train_split = load_from_disk(train_path)
    val_split = load_from_disk(val_path)
    test_split = load_from_disk(test_path)

    return train_split, val_split, test_split


if __name__ == "__main__":

    train_split, val_split, test_split = load_atr_dataset()

    print("=" * 50)
    print("ATR Dataset")
    print("=" * 50)

    print(f"Train: {len(train_split)}")
    print(f"Val:   {len(val_split)}")
    print(f"Test:  {len(test_split)}")

    image = np.array( train_split[0]["image"].convert("RGB"))

    raw_mask = np.array(train_split[0]["mask"].convert("L"))

    binary_mask = mask_to_binary(raw_mask)

    print("\nSample:")
    print(f"Image shape:        {image.shape}")
    print(f"Original mask:      {np.unique(raw_mask)}")
    print(f"Binary mask values: {np.unique(binary_mask)}")
    print(f"Clothes ratio:      {binary_mask.mean():.3f}")