"""Check that every NEU-DET image pairs with exactly one VOC annotation."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

RAW = Path("data/raw/NEU-DET")


@dataclass
class PairReport:
    pairs: dict[str, tuple[Path, Path]]  # stem -> (image, xml)
    images_only: set[str]
    xmls_only: set[str]
    duplicates: set[str]
    split_mismatch: set[str]


def _index(root: Path, pattern: str) -> tuple[dict[str, Path], set[str]]:
    index: dict[str, Path] = {}
    dupes: set[str] = set()
    for p in sorted(root.rglob(pattern)):
        if p.stem in index:
            dupes.add(p.stem)
        index[p.stem] = p
    return index, dupes


def _split(path: Path, root: Path) -> str:
    return path.relative_to(root).parts[0]  # "train" or "validation"


def check(root: Path = RAW) -> PairReport:
    images, dup_img = _index(root, "*.jpg")
    xmls, dup_xml = _index(root, "*.xml")
    both = images.keys() & xmls.keys()
    return PairReport(
        pairs={s: (images[s], xmls[s]) for s in sorted(both)},
        images_only=set(images.keys() - xmls.keys()),
        xmls_only=set(xmls.keys() - images.keys()),
        duplicates=dup_img | dup_xml,
        split_mismatch={s for s in both if _split(images[s], root) != _split(xmls[s], root)},
    )


def main(root: Path = RAW) -> int:
    r = check(root)
    print(f"paired: {len(r.pairs)}")
    for s in sorted(r.images_only):
        print(f"ERROR image without XML: {s}")
    for s in sorted(r.xmls_only):
        print(f"ERROR XML without image: {s}")
    for s in sorted(r.duplicates):
        print(f"ERROR duplicate stem: {s}")
    for s in sorted(r.split_mismatch):
        img, xml = r.pairs[s]
        print(f"WARN split mismatch: {s}  image={_split(img, root)}  xml={_split(xml, root)}")
    return 1 if (r.images_only or r.xmls_only or r.duplicates) else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else RAW))