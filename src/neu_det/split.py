"""Stratified, reproducible train/val/test split of NEU-DET images."""
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(images: list[Path], seed: int = 42,
                     val_frac: float = 0.1, test_frac: float = 0.1) -> dict[str, list[str]]:
    by_class = defaultdict(list)
    for img in images:
        by_class[img.parent.name].append(img.stem)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls in sorted(by_class):
        stems = sorted(by_class[cls])
        rng.shuffle(stems)
        n_val = round(len(stems) * val_frac)
        n_test = round(len(stems) * test_frac)
        splits["val"] += stems[:n_val]
        splits["test"] += stems[n_val:n_val + n_test]
        splits["train"] += stems[n_val + n_test:]

    return {name: sorted(stems) for name, stems in splits.items()}