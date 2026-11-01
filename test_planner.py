#!/usr/bin/env python3
"""Quick test of the planner implementation."""

import sys
import numpy as np
from pathlib import Path

# Test collision checking
print("Testing collision.py...")
from collision import segment_aabb_hit, segment_hits_any, point_in_bounds

# Test a simple segment-box intersection
p0 = np.array([0, 0, 0], dtype=float)
p1 = np.array([2, 2, 2], dtype=float)
box_min = np.array([0.5, 0.5, 0.5], dtype=float)
box_max = np.array([1.5, 1.5, 1.5], dtype=float)

hit = segment_aabb_hit(p0, p1, box_min, box_max)
print(f"  Segment-box intersection test: {hit} (expected True)")
assert hit, "Segment should intersect box"

# Test point in bounds
p = np.array([0.5, 0.5, 0.5])
boundary = np.array([0, 0, 0, 1, 1, 1])
inside = point_in_bounds(p, boundary)
print(f"  Point in bounds test: {inside} (expected True)")
assert inside, "Point should be in bounds"

print("[OK] collision.py works!\n")

# Test planner
print("Testing planner.py...")
from planner import GridPlanner

# Create a simple environment with no obstacles
boundary = np.array([0, 0, 0, 10, 10, 10])
blocks = np.zeros((0, 6))

planner = GridPlanner(boundary, blocks, res=0.5, epsilon=2.0)

start = np.array([1, 1, 1], dtype=float)
goal = np.array([9, 9, 9], dtype=float)

path, cost, expansions = planner.plan(start, goal)
print(f"  Planned path with {len(path)} waypoints")
print(f"  Path cost: {cost:.2f}")
print(f"  Nodes expanded: {expansions}")

assert path is not None, "Should find a path"
assert len(path) > 1, "Path should have multiple waypoints"
np.testing.assert_array_almost_equal(path[0], start, decimal=5)
np.testing.assert_array_almost_equal(path[-1], goal, decimal=1)
print("[OK] planner.py works!\n")

# Test replanner
print("Testing replanner.py...")
from replanner import ReplanningPlanner

replanner = ReplanningPlanner(boundary, blocks, res=0.5, epsilon=3.0)

# Plan to multiple goals
goals = [
    np.array([2, 2, 2], dtype=float),
    np.array([8, 8, 8], dtype=float),
    np.array([5, 5, 5], dtype=float),
]

times = []
for i, goal_i in enumerate(goals):
    import time
    t0 = time.perf_counter()
    path, cost, expansions = replanner.plan_to(start, goal_i)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  Goal {i+1}: {len(path) if path is not None else 0} waypoints, cost {cost:.2f}, {expansions} expansions, {dt*1000:.2f}ms")

print(f"[OK] replanner.py works!\n")

print("-" * 50)
print("All tests passed!")
print("-" * 50)
