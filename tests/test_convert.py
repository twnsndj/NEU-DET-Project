import pytest
from neu_det.convert import voc_box_to_yolo

def test_worked_example():
    assert voc_box_to_yolo(10, 40, 110, 60, 200, 200) == pytest.approx((0.30, 0.25, 0.50, 0.10))

def test_inverted_box_raises():
    with pytest.raises(ValueError):
        voc_box_to_yolo(110, 40, 10, 60, 200, 200)