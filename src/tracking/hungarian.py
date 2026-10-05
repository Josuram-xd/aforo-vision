"""Bipartite graph and Hungarian algorithm (own implementation) for detection-to-track association.

Left nodes are the detections of frame t, right nodes are the tracks from frame t-1. Every edge
costs 1 - IoU, so the minimum-cost perfect matching is the assignment with the most overlap.
"""

from collections.abc import Sequence

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

_INF = float("inf")


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0.0 when they do not overlap or one has no area."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _area(box: Box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


class BipartiteGraph:
    """Complete weighted bipartite graph stored as an adjacency matrix: weights[left][right]."""

    def __init__(self, weights: Sequence[Sequence[float]]):
        widths = {len(row) for row in weights}
        if len(widths) > 1:
            raise ValueError("all rows of the weight matrix must have the same length")
        self.weights = [list(row) for row in weights]
        self.left_count = len(self.weights)
        self.right_count = widths.pop() if widths else 0

    @classmethod
    def from_boxes(cls, detections: Sequence[Box], tracks: Sequence[Box]) -> "BipartiteGraph":
        """Build the graph with cost 1 - IoU between every detection and every track."""
        return cls([[1.0 - iou(det, trk) for trk in tracks] for det in detections])

    def min_cost_matching(self) -> list[tuple[int, int]]:
        """Return (left, right) pairs of the minimum total cost; min(left, right) pairs in total."""
        return hungarian(self.weights)


def hungarian(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Solve the rectangular assignment problem in O(n^2 * m) (Kuhn-Munkres with potentials).

    Returns (row, col) pairs, sorted by row, that minimize the total cost. Every row of the smaller
    side gets exactly one column of the larger side; the rest stay unassigned.
    """
    rows = len(cost)
    cols = len(cost[0]) if rows else 0
    if rows == 0 or cols == 0:
        return []
    if rows > cols:  # the algorithm needs rows <= cols, so solve the transposed problem
        transposed = [[cost[r][c] for r in range(rows)] for c in range(cols)]
        return sorted((r, c) for c, r in hungarian(transposed))

    # 1-indexed arrays; index 0 is a virtual node used to start each augmenting path
    row_potential = [0.0] * (rows + 1)
    col_potential = [0.0] * (cols + 1)
    col_owner = [0] * (cols + 1)  # col_owner[j] = row currently assigned to column j (0 = free)
    way = [0] * (cols + 1)  # previous column on the shortest augmenting path

    for row in range(1, rows + 1):
        col_owner[0] = row
        current_col = 0
        min_slack = [_INF] * (cols + 1)
        visited = [False] * (cols + 1)
        while True:
            visited[current_col] = True
            current_row = col_owner[current_col]
            delta = _INF
            next_col = 0
            for col in range(1, cols + 1):
                if visited[col]:
                    continue
                slack = cost[current_row - 1][col - 1] - row_potential[current_row] - col_potential[col]
                if slack < min_slack[col]:
                    min_slack[col] = slack
                    way[col] = current_col
                if min_slack[col] < delta:
                    delta = min_slack[col]
                    next_col = col
            for col in range(cols + 1):
                if visited[col]:
                    row_potential[col_owner[col]] += delta
                    col_potential[col] -= delta
                else:
                    min_slack[col] -= delta
            current_col = next_col
            if col_owner[current_col] == 0:
                break
        while current_col:  # flip the augmenting path
            previous_col = way[current_col]
            col_owner[current_col] = col_owner[previous_col]
            current_col = previous_col

    return sorted((col_owner[col] - 1, col - 1) for col in range(1, cols + 1) if col_owner[col])


def associate(
    detections: Sequence[Box], tracks: Sequence[Box], iou_threshold: float,
    similarity: Sequence[Sequence[float]] | None = None,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to tracks by maximum similarity, dropping pairs with IoU below the threshold.

    `similarity[detection][track]` (0..1) defaults to the IoU; pass a matrix to rank candidates with
    other cues (see occlusion.py). The IoU gate applies either way.

    Returns (matches, unmatched_detection_indices, unmatched_track_indices).
    """
    if similarity is None:
        graph = BipartiteGraph.from_boxes(detections, tracks)
    else:
        graph = BipartiteGraph([[1.0 - value for value in row] for row in similarity])
    matches = [
        (det, trk) for det, trk in graph.min_cost_matching() if iou(detections[det], tracks[trk]) >= iou_threshold
    ]
    matched_dets = {det for det, _ in matches}
    matched_trks = {trk for _, trk in matches}
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_trks = [i for i in range(len(tracks)) if i not in matched_trks]
    return matches, unmatched_dets, unmatched_trks
