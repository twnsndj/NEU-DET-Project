from pathlib import Path
from neu_det.split import stratified_split

CLASSES = ["crazing", "inclusion", "patches"]


def fake_images(n_per_class=300):
    return [Path(f"images/{c}/{c}_{i}.jpg") for c in CLASSES for i in range(n_per_class)]


def test_sizes_per_class():
    s = stratified_split(fake_images())
    for split, expected in [("train", 240), ("val", 30), ("test", 30)]:
        for c in CLASSES:
            assert sum(stem.startswith(c + "_") for stem in s[split]) == expected


def test_partition():
    images = fake_images()
    s = stratified_split(images)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)  # disjoint
    assert train | val | test == {p.stem for p in images}                 # complete


def test_deterministic():
    images = fake_images()
    assert stratified_split(images, seed=1) == stratified_split(images, seed=1)
    assert stratified_split(images, seed=1) != stratified_split(images, seed=2)
    assert stratified_split(images[::-1], seed=1) == stratified_split(images, seed=1)