import numpy as np

EPS = 1e-12


def segment_aabb_hit(p0, p1, box_min, box_max):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0

    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < EPS:
            # Segment parallel to this pair of slabs. Must already be inside.
            if p0[i] < box_min[i] - EPS or p0[i] > box_max[i] + EPS:
                return False
            continue
        inv = 1.0 / d[i]
        t1 = (box_min[i] - p0[i]) * inv
        t2 = (box_max[i] - p0[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin = t1
        if t2 < tmax:
            tmax = t2
        if tmin > tmax:
            return False
    return True


def segment_hits_any(p0, p1, blocks):
    if len(blocks) == 0:
        return False
    blocks = np.asarray(blocks, dtype=float)
    for b in blocks:
        if segment_aabb_hit(p0, p1, b[0:3], b[3:6]):
            return True
    return False


def point_in_bounds(p, boundary):
    return (boundary[0] <= p[0] <= boundary[3] and
            boundary[1] <= p[1] <= boundary[4] and
            boundary[2] <= p[2] <= boundary[5])
def segment_hits_any_vec(p0, p1, blocks_min, blocks_max):
    """Vectorized slab test: one segment vs many boxes.

    blocks_min, blocks_max: (N, 3) arrays.
    Returns True if the segment hits any block.
    """
    if blocks_min.shape[0] == 0:
        return False

    p0 = np.asarray(p0, dtype=float)
    d = np.asarray(p1, dtype=float) - p0

    safe = np.where(np.abs(d) < EPS, EPS, d)
    inv = 1.0 / safe  # shape (3,)

    t1 = (blocks_min - p0) * inv  # (N, 3)
    t2 = (blocks_max - p0) * inv  # (N, 3)

    tmin_axis = np.minimum(t1, t2)
    tmax_axis = np.maximum(t1, t2)
    parallel = np.abs(d) < EPS  # (3,)
    if parallel.any():
        inside = (p0 >= blocks_min) & (p0 <= blocks_max)  # (N, 3)
        for i in np.where(parallel)[0]:
            tmin_axis[:, i] = np.where(inside[:, i], -np.inf,  np.inf)
            tmax_axis[:, i] = np.where(inside[:, i],  np.inf, -np.inf)

    tmin = np.maximum.reduce([tmin_axis[:, 0], tmin_axis[:, 1], tmin_axis[:, 2], np.zeros(blocks_min.shape[0])])
    tmax = np.minimum.reduce([tmax_axis[:, 0], tmax_axis[:, 1], tmax_axis[:, 2], np.ones(blocks_min.shape[0])])

    return bool(np.any(tmin <= tmax))
