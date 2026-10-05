import pytest

from src.tracking.hungarian import BipartiteGraph, associate, hungarian, iou


def _total(cost, pairs):
    return sum(cost[r][c] for r, c in pairs)


def test_iou_identical_disjoint_and_half_overlap():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area_box_is_zero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_hungarian_classic_3x3():
    cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
    pairs = hungarian(cost)
    assert _total(cost, pairs) == 5
    assert len(pairs) == 3


def test_hungarian_beats_greedy():
    # Greedy would take (0, 0)=1 and then be forced into (1, 1)=100
    cost = [[1, 2], [2, 100]]
    assert hungarian(cost) == [(0, 1), (1, 0)]


def test_hungarian_more_columns_than_rows():
    cost = [[9, 2, 7], [3, 6, 1]]
    assert hungarian(cost) == [(0, 1), (1, 2)]


def test_hungarian_more_rows_than_columns():
    cost = [[9, 3], [2, 6], [7, 1]]
    assert hungarian(cost) == [(1, 0), (2, 1)]


def test_hungarian_empty_inputs():
    assert hungarian([]) == []
    assert hungarian([[], []]) == []


def test_graph_rejects_ragged_matrix():
    with pytest.raises(ValueError):
        BipartiteGraph([[1, 2], [3]])


def test_graph_from_boxes_uses_one_minus_iou():
    graph = BipartiteGraph.from_boxes([(0, 0, 10, 10)], [(0, 0, 10, 10), (50, 50, 60, 60)])
    assert (graph.left_count, graph.right_count) == (1, 2)
    assert graph.weights == [[0.0, 1.0]]


def test_associate_matches_moved_boxes_and_reports_leftovers():
    tracks = [(0, 0, 10, 10), (100, 100, 110, 110), (200, 0, 210, 10)]
    detections = [(101, 100, 111, 110), (1, 0, 11, 10), (400, 400, 410, 410)]

    matches, unmatched_dets, unmatched_trks = associate(detections, tracks, iou_threshold=0.3)

    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched_dets == [2]
    assert unmatched_trks == [2]


def test_associate_drops_pairs_below_iou_threshold():
    matches, unmatched_dets, unmatched_trks = associate([(0, 0, 10, 10)], [(8, 0, 18, 10)], iou_threshold=0.3)
    assert matches == []
    assert unmatched_dets == [0]
    assert unmatched_trks == [0]


def test_associate_with_no_tracks_or_no_detections():
    assert associate([(0, 0, 1, 1)], [], 0.3) == ([], [0], [])
    assert associate([], [(0, 0, 1, 1)], 0.3) == ([], [], [0])
