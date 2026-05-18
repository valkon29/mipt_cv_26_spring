import os
import numpy as np
import cv2
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import torch


def get_image_pairs(raw_dir: str, target_dir: str) -> List[Tuple[str, str]]:
    raw_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    common_files = []
    for raw_file in raw_files:
        raw_name = os.path.splitext(raw_file)[0].lower()
        for target_file in target_files:
            target_name = os.path.splitext(target_file)[0].lower()
            if raw_name == target_name:
                common_files.append((raw_file, target_file))
                break

    return [(os.path.join(raw_dir, rf), os.path.join(target_dir, tf)) for rf, tf in common_files]


def split_data(pairs: List[Tuple[str, str]],
               test_size: float = 0.2,
               val_size: float = 0.1,
               seed: int = 42) -> Dict[str, List[Tuple[str, str]]]:
    np.random.seed(seed)
    n_total = len(pairs)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    indices = np.random.permutation(n_total)

    return {
        "train": [pairs[i] for i in indices[:n_train]],
        "val": [pairs[i] for i in indices[n_train:n_train + n_val]],
        "test": [pairs[i] for i in indices[n_train + n_val:]]
    }


def load_image(path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0


def calculate_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
    from skimage.color import rgb2lab, deltaE_ciede2000

    lab1, lab2 = rgb2lab(img1), rgb2lab(img2)
    return {
        'psnr': psnr(img1, img2, data_range=1.0),
        'ssim': ssim(img1, img2, multichannel=True, data_range=1.0, channel_axis=-1),
        'delta_e': np.mean(deltaE_ciede2000(lab1, lab2))
    }


class ImageEnhancementDataset(Dataset):

    def __init__(self, image_pairs: List[Tuple[str, str]], img_size: int = 256):
        self.image_pairs = image_pairs
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_path, target_path = self.image_pairs[idx]

        raw_img = load_image(raw_path, target_size=(self.img_size, self.img_size))
        target_img = load_image(target_path, target_size=(self.img_size, self.img_size))

        raw_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float()
        target_tensor = torch.from_numpy(target_img).permute(2, 0, 1).float()

        return raw_tensor, target_tensor


def create_datasets(size: str = 'small',
                   img_size: int = 256,
                   seed: int = 42) -> Dict:

    DATA_ROOT = "mit_adobe_5k_dataset"
    RAW_DIR = os.path.join(DATA_ROOT, "raw")
    TARGET_DIR = os.path.join(DATA_ROOT, "c")

    image_pairs = get_image_pairs(RAW_DIR, TARGET_DIR)

    if size == 'small':
        splits = split_data(image_pairs[:1520], test_size=0.2, val_size=0.1, seed=seed)
        train_subset = splits['train'][:1000]
        val_subset = splits['val'][:500]
        test_subset = splits['test'][:20]
        print("✓ Using SMALL dataset (fast prototyping)")
        print(f"  Train: 1000, Val: 500, Test: 20")
    elif size == 'full':
        splits = split_data(image_pairs, test_size=0.2, val_size=0.1, seed=seed)
        train_subset = splits['train']
        val_subset = splits['val']
        test_subset = splits['test'][:1000]
        print("✓ Using FULL dataset (production)")
        print(f"  Train: 3500, Val: 500, Test: 1000")
    elif size == 'complete':
        splits = split_data(image_pairs, test_size=0.1, val_size=0.1, seed=seed)
        train_subset = splits['train'][:4000]
        val_subset = splits['val']
        test_subset = splits['test']
        print("✓ Using COMPLETE dataset (all 5000 images)")
        print(f"  Train: 4000, Val: 500, Test: 500")
    else:
        raise ValueError(f"Unknown size: {size}. Use 'small', 'full', or 'complete'.")

    train_dataset = ImageEnhancementDataset(train_subset, img_size=img_size)
    val_dataset = ImageEnhancementDataset(val_subset, img_size=img_size)
    test_dataset = ImageEnhancementDataset(test_subset, img_size=img_size)

    print(f"  Created datasets:")
    print(f"    Train: {len(train_dataset)} images")
    print(f"    Val: {len(val_dataset)} images")
    print(f"    Test: {len(test_dataset)} images")

    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'test_pairs': test_subset,
        'metadata': {
            'size': size,
            'img_size': img_size,
            'total_pairs': len(image_pairs),
            'seed': seed
        }
    }


def get_dataloaders(size: str = 'small',
                   img_size: int = 256,
                   batch_size: int = 8,
                   num_workers: int = 0) -> Dict:
    from torch.utils.data import DataLoader

    datasets = create_datasets(size=size, img_size=img_size)

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets['test'],
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"✓ Dataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'test_pairs': datasets['test_pairs'],
        'metadata': datasets['metadata']
    }


if __name__ == "__main__":
    print("Testing dataset module...")
    print("\n=== Small Dataset ===")
    small_data = create_datasets(size='small', img_size=256)

    print("\n=== Full Dataset ===")
    full_data = create_datasets(size='full', img_size=256)

    print("\n✓ Dataset module working correctly!")
