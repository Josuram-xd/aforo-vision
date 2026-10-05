import random

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from src.tracking.hungarian import BipartiteGraph, hungarian, iou


def _scipy_cost(cost):
    rows, cols = linear_sum_assignment(np.array(cost, dtype=float))
    return float(np.array(cost, dtype=float)[rows, cols].sum()), len(rows)


def _own_cost(cost):
    pairs = hungarian(cost)
    rows = [r for r, _ in pairs]
    cols = [c for _, c in pairs]
    assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols)  # a valid matching
    return float(sum(cost[r][c] for r, c in pairs)), len(pairs)


@pytest.mark.parametrize("cost", [
    [[4, 1, 3], [2, 0, 5], [3, 2, 2]],
    [[1, 2], [2, 100]],  # greedy trap
    [[9, 2, 7], [3, 6, 1]],  # wide
    [[9, 3], [2, 6], [7, 1]],  # tall
    [[5]],
    [[0, 0], [0, 0]],  # all ties
    [[-3, -1], [-2, -8]],  # negative costs
    [[1.5, 2.25, 0.75], [0.5, 3.0, 2.0], [2.0, 1.0, 4.5]],
])
def test_known_cases_match_scipy(cost):
    assert _own_cost(cost) == pytest.approx(_scipy_cost(cost))


@pytest.mark.parametrize("rows,cols", [(1, 1), (2, 2), (5, 5), (8, 8), (3, 7), (7, 3), (1, 6), (6, 1)])
def test_random_matrices_match_scipy(rows, cols):
    rng = random.Random(rows * 100 + cols)
    for _ in range(50):
        cost = [[rng.uniform(0, 1) for _ in range(cols)] for _ in range(rows)]
        assert _own_cost(cost) == pytest.approx(_scipy_cost(cost))


def test_random_integer_matrices_with_many_ties_match_scipy():
    rng = random.Random(7)
    for _ in range(100):
        size = rng.randint(2, 7)
        cost = [[rng.randint(0, 3) for _ in range(size)] for _ in range(size)]
        assert _own_cost(cost) == pytest.approx(_scipy_cost(cost))


def test_random_box_graphs_match_scipy():
    rng = random.Random(42)

    def random_box():
        x, y = rng.uniform(0, 300), rng.uniform(0, 300)
        return (x, y, x + rng.uniform(20, 80), y + rng.uniform(40, 160))

    for _ in range(50):
        detections = [random_box() for _ in range(rng.randint(1, 8))]
        tracks = [random_box() for _ in range(rng.randint(1, 8))]
        graph = BipartiteGraph.from_boxes(detections, tracks)
        own = sum(graph.weights[d][t] for d, t in graph.min_cost_matching())
        expected, _ = _scipy_cost(graph.weights)
        assert own == pytest.approx(expected)
        assert all(0.0 <= iou(d, t) <= 1.0 for d in detections for t in tracks)
