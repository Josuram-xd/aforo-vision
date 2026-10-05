import cv2
import numpy as np

from src.identity.periocular import crop_periocular


def _frame_with_eyes(left, right, shape=(300, 400, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    for point in (left, right):
        cv2.circle(frame, point, 3, (255, 255, 255), -1)
    return frame


def _blob_center(crop, half):
    gray = crop.mean(axis=2)
    columns = slice(0, crop.shape[1] // 2) if half == "left" else slice(crop.shape[1] // 2, None)
    ys, xs = np.nonzero(gray[:, columns] > 128)
    return xs.mean() + (0 if half == "left" else crop.shape[1] // 2), ys.mean()


def test_crop_has_natural_size_from_eye_distance():
    frame = _frame_with_eyes((100, 100), (140, 100))  # 40 px apart
    crop = crop_periocular(frame, np.array([[100, 100], [140, 100]]))
    assert crop.shape == (40, 80, 3)  # height 1x, width 2x the inter-ocular distance


def test_eye_order_does_not_matter():
    frame = _frame_with_eyes((100, 100), (140, 100))
    a = crop_periocular(frame, np.array([[100, 100], [140, 100]]))
    b = crop_periocular(frame, np.array([[140, 100], [100, 100]]))
    assert np.array_equal(a, b)


def test_extra_landmarks_are_ignored():
    frame = _frame_with_eyes((100, 100), (140, 100))
    five_points = np.array([[100, 100], [140, 100], [120, 130], [105, 150], [135, 150]])
    assert np.array_equal(crop_periocular(frame, five_points), crop_periocular(frame, five_points[:2]))


def test_output_size_resizes_the_crop():
    frame = _frame_with_eyes((100, 100), (140, 100))
    crop = crop_periocular(frame, np.array([[100, 100], [140, 100]]), output_size=(112, 56))
    assert crop.shape == (56, 112, 3)


def test_eyes_land_at_the_same_place_in_the_crop_whatever_the_head_tilt():
    frame = _frame_with_eyes((100, 100), (140, 130))  # head tilted
    crop = crop_periocular(frame, np.array([[100, 100], [140, 130]]), output_size=(160, 80))
    (lx, ly), (rx, ry) = _blob_center(crop, "left"), _blob_center(crop, "right")
    assert abs(ly - ry) < 2  # eyes level
    assert abs(ly - 40) < 2  # at mid height
    assert abs(lx - 40) < 3 and abs(rx - 120) < 3  # at 1/4 and 3/4 of the width


def test_does_not_modify_the_frame():
    frame = _frame_with_eyes((100, 100), (140, 100))
    before = frame.copy()
    crop_periocular(frame, np.array([[100, 100], [140, 100]]))
    assert np.array_equal(frame, before)


def test_returns_none_for_unusable_landmarks():
    frame = _frame_with_eyes((100, 100), (140, 100))
    assert crop_periocular(frame, np.array([[100, 100]])) is None  # one point
    assert crop_periocular(frame, np.empty((0, 2))) is None
    assert crop_periocular(frame, np.array([[100, 100], [104, 100]])) is None  # eyes too close
    assert crop_periocular(frame, np.array([[100, 100], [np.nan, 100]])) is None
    assert crop_periocular(frame, np.array([[100, 100], [500, 100]])) is None  # eye outside the frame
