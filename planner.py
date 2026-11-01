import heapq
import numpy as np
from collision import segment_aabb_hit


# 26-connected neighborhood offsets
_NEIGHBORS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if not (dx == 0 and dy == 0 and dz == 0)],
    dtype=int,
)
_NEIGHBOR_COSTS = np.linalg.norm(_NEIGHBORS, axis=1)  # 1.0, sqrt(2), sqrt(3)


def diagonal_heuristic(a, b, res):
    """3-D diagonal (octile-equivalent) distance, admissible for 26-conn grids."""
    d = np.abs(np.asarray(a) - np.asarray(b)) * res
    dx, dy, dz = sorted(d)
    # Cost of dx face moves, then (dy-dx) edge moves, then (dz-dy) corner moves.
    return (np.sqrt(3) - np.sqrt(2)) * dx + (np.sqrt(2) - 1.0) * dy + dz


class GridPlanner:
    def __init__(self, boundary, blocks, res=0.5, epsilon=2.0):
        boundary = np.asarray(boundary, dtype=float)
        if boundary.ndim == 2:
            boundary = boundary[0]  # Take first row if 2D array
        self.boundary = boundary
        self.bmin = self.boundary[0:3]
        self.bmax = self.boundary[3:6]
        if len(blocks) == 0:
            self.blocks = np.zeros((0, 6))
        else:
            self.blocks = np.asarray(blocks, dtype=float)[:, :6]
        self.res = float(res)
        self.epsilon = float(epsilon)
    def to_idx(self, p):
        return tuple(((np.asarray(p) - self.bmin) / self.res).round().astype(int))

    def to_pos(self, idx):
        return self.bmin + np.asarray(idx, dtype=float) * self.res

    def in_bounds(self, p):
        return bool(np.all(p >= self.bmin - 1e-9) and np.all(p <= self.bmax + 1e-9))

    def edge_free(self, p0, p1):
        if not (self.in_bounds(p0) and self.in_bounds(p1)):
            return False
        for b in self.blocks:
            if segment_aabb_hit(p0, p1, b[0:3], b[3:6]):
                return False
        return True

    def plan(self, start, goal, max_expansions=2_000_000):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        s_idx = self.to_idx(start)
        g_idx = self.to_idx(goal)

        if s_idx == g_idx:
            return np.array([start, goal]), 0.0, 0
        open_heap = []
        g_score = {s_idx: 0.0}
        parent = {s_idx: None}
        closed = set()
        counter = 0

        h0 = diagonal_heuristic(s_idx, g_idx, self.res)
        heapq.heappush(open_heap, (self.epsilon * h0, counter, s_idx))

        expansions = 0
        while open_heap and expansions < max_expansions:
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            expansions += 1

            cur_pos = self.to_pos(cur)
            if np.max(np.abs(np.array(cur) - np.array(g_idx))) <= 1:
                if self.edge_free(cur_pos, goal):
                    return self._reconstruct(parent, cur, start, goal), g_score[cur] + np.linalg.norm(goal - cur_pos), expansions

            for off, step_cost in zip(_NEIGHBORS, _NEIGHBOR_COSTS):
                nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
                if nxt in closed:
                    continue
                nxt_pos = self.to_pos(nxt)
                if not self.in_bounds(nxt_pos):
                    continue
                if not self.edge_free(cur_pos, nxt_pos):
                    continue

                tentative_g = g_score[cur] + step_cost * self.res
                if tentative_g < g_score.get(nxt, np.inf):
                    g_score[nxt] = tentative_g
                    parent[nxt] = cur
                    h = diagonal_heuristic(nxt, g_idx, self.res)
                    f = tentative_g + self.epsilon * h
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, nxt))

        return None, float('inf'), expansions

    def _reconstruct(self, parent, end_idx, start_pos, goal_pos):
        path_idx = [end_idx]
        while parent[path_idx[-1]] is not None:
            path_idx.append(parent[path_idx[-1]])
        path_idx.reverse()
        path = [self.to_pos(i) for i in path_idx]
        # Replace endpoints with true start/goal coords for a clean output.
        path[0] = np.asarray(start_pos, dtype=float)
        path.append(np.asarray(goal_pos, dtype=float))
        return np.array(path)

def shortcut_path(path, blocks, max_iter=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    path = [np.asarray(p, dtype=float) for p in path]
    blocks = np.asarray(blocks, dtype=float)[:, :6] if len(blocks) else np.zeros((0, 6))

    for _ in range(max_iter):
        if len(path) < 3:
            break
        i = rng.integers(0, len(path) - 2)
        j = rng.integers(i + 2, len(path))
        if any(segment_aabb_hit(path[i], path[j], b[0:3], b[3:6]) for b in blocks):
            continue
        path = path[:i + 1] + path[j:]
    return np.array(path)
