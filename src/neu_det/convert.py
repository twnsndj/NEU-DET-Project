"""Convert Pascal VOC annotations to YOLO label format.

Box width is xmax - xmin (no +1). On 200 px images the difference is 0.5%
and doesn't affect results; it's fixed here so every label is consistent.
"""
from __future__ import annotations

from defusedxml.ElementTree import parse
from PIL import Image

from neu_det.check_data import check

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def voc_box_to_yolo(xmin, ymin, xmax, ymax, w, h) -> tuple[float, float, float, float]:
    """Pure math: corners in pixels -> (cx, cy, bw, bh) normalized to 0-1."""
    if not (0 <= xmin < xmax <= w and 0 <= ymin < ymax <= h):
        raise ValueError(f"bad box {(xmin, ymin, xmax, ymax)} in {w}x{h} image")
    return ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h,
            (xmax - xmin) / w, (ymax - ymin) / h)


def parse_voc(xml_path) -> tuple[int, int, list[tuple[str, int, int, int, int]]]:
    """Read width, height, and (name, xmin, ymin, xmax, ymax) for each object."""
    root = parse(xml_path).getroot()
    w, h = int(root.findtext("size/width")), int(root.findtext("size/height"))
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        corners = (int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.findtext("name").strip(), *corners))
    return w, h, boxes


def to_yolo_lines(xml_path, img_path) -> list[str]:
    """Validate one annotation against its image and return YOLO label lines."""
    w, h, boxes = parse_voc(xml_path)
    with Image.open(img_path) as im:
        if im.size != (w, h):
            raise ValueError(f"{xml_path}: XML says {(w, h)}, image is {im.size}")
    if not boxes:
        raise ValueError(f"{xml_path}: no objects")
    lines = []
    for name, *corners in boxes:
        if name not in CLASS_TO_ID:
            raise ValueError(f"{xml_path}: unknown class {name!r}")
        try:
            cx, cy, bw, bh = voc_box_to_yolo(*corners, w, h)
        except ValueError as e:
            raise ValueError(f"{xml_path}: {e}") from e
        lines.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> int:
    """Dry run: convert every pair, report all problems, write nothing."""
    n_boxes, errors = 0, 0
    for img, xml in check().pairs.values():
        try:
            n_boxes += len(to_yolo_lines(xml, img))
        except ValueError as e:
            errors += 1
            print(f"ERROR {e}")
    print(f"boxes: {n_boxes}  files with errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())