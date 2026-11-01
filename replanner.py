import heapq
import time
import numpy as np

from planner import GridPlanner, diagonal_heuristic, _NEIGHBORS, _NEIGHBOR_COSTS


class ReplanningPlanner(GridPlanner):
    def __init__(self, boundary, blocks, res=0.5, epsilon=3.0):
        super().__init__(boundary, blocks, res=res, epsilon=epsilon)
        self._start_idx = None
        self._g_score = {}            # Persistent: cost-to-come from fixed start
        self._parent = {}             # Persistent: parent pointers
        self._frontier = set()        # Persistent: generated but not settled nodes
        self._incons = set()          # Persistent: nodes with out-of-date heuristics
        self.last_status = "initialized"  # Status of last plan_to() call

    def _reset_from_start(self, start):

        self._start_idx = self.to_idx(start)
        self._g_score = {self._start_idx: 0.0}
        self._parent = {self._start_idx: None}
        self._frontier = {self._start_idx}
        self._incons = set()

    def _make_open(self, goal_idx, eps):
        open_heap = []
        counter = 0

        # Insert all frontier nodes
        for idx in self._frontier:
            g_val = self._g_score.get(idx, np.inf)
            if g_val < np.inf:
                h = diagonal_heuristic(idx, goal_idx, self.res)
                f = g_val + eps * h
                heapq.heappush(open_heap, (f, counter, idx))
                counter += 1

        # Insert all inconsistent nodes
        for idx in self._incons:
            g_val = self._g_score.get(idx, np.inf)
            if g_val < np.inf:
                h = diagonal_heuristic(idx, goal_idx, self.res)
                f = g_val + eps * h
                heapq.heappush(open_heap, (f, counter, idx))
                counter += 1

        # If both frontier and incons are empty, fall back to all known g_score nodes
        if not open_heap:
            for idx, g_val in self._g_score.items():
                h = diagonal_heuristic(idx, goal_idx, self.res)
                f = g_val + eps * h
                heapq.heappush(open_heap, (f, counter, idx))
                counter += 1

        # Clear incons after inserting into OPEN
        self._incons.clear()

        return open_heap, counter

    def _closed_goal_candidate(self, g_idx, goal, candidate_nodes):
        for idx in candidate_nodes:
            if np.max(np.abs(np.array(idx) - np.array(g_idx))) <= 1:
                idx_pos = self.to_pos(idx)
                if self.edge_free(idx_pos, goal):
                    cost = self._g_score[idx] + np.linalg.norm(goal - idx_pos)
                    return idx, cost
        return None, np.inf

    def plan_to(self, start, goal, max_expansions=2_000_000, time_budget=None,
                epsilon_start=None, epsilon_min=1.0, epsilon_step=1.0):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        s_idx = self.to_idx(start)
        g_idx = self.to_idx(goal)

        # Reset if start changed
        if self._start_idx is None or s_idx != self._start_idx:
            self._reset_from_start(start)

        # Determine initial epsilon
        if epsilon_start is None:
            epsilon_start = self.epsilon

        # Fast goal check: scan all known g_score nodes for L_inf <= 1 candidate
        cand, cand_cost = self._closed_goal_candidate(g_idx, goal, self._g_score.keys())
        if cand is not None:
            self.last_status = "success"
            return self._finish(cand, start, goal, append_goal=True), cand_cost, 0

        # Per-query state
        closed_this_query = set()
        expansions = 0
        best_node = s_idx
        best_h = diagonal_heuristic(s_idx, g_idx, self.res)
        time_start = time.perf_counter() if time_budget else None

        # Build initial OPEN heap from persistent frontier and incons
        open_heap, counter = self._make_open(g_idx, epsilon_start)

        # Main search loop
        while open_heap and expansions < max_expansions:
            # Check time budget
            if time_budget is not None:
                elapsed = time.perf_counter() - time_start
                if elapsed > time_budget:
                    # Return best partial path without goal (may not reach)
                    if best_node is not None and best_node in self._g_score:
                        self.last_status = "partial_timeout"
                        return self._finish(best_node, start, goal, append_goal=False), \
                               self._g_score[best_node], expansions
                    self.last_status = "failed"
                    return None, float('inf'), expansions

            # Pop from OPEN
            _, _, cur = heapq.heappop(open_heap)
            if cur in closed_this_query:
                continue

            # Mark as closed in this query
            closed_this_query.add(cur)
            self._frontier.discard(cur)
            expansions += 1

            cur_pos = self.to_pos(cur)

            # Track best node by heuristic distance to goal
            cur_h = diagonal_heuristic(cur, g_idx, self.res)
            if cur_h < best_h:
                best_h = cur_h
                best_node = cur

            # Goal check: within L_inf distance 1 and collision-free edge
            if np.max(np.abs(np.array(cur) - np.array(g_idx))) <= 1:
                if self.edge_free(cur_pos, goal):
                    cost = self._g_score[cur] + np.linalg.norm(goal - cur_pos)
                    self.last_status = "success"
                    return self._finish(cur, start, goal, append_goal=True), cost, expansions

            # Expand neighbors
            for off, step_cost in zip(_NEIGHBORS, _NEIGHBOR_COSTS):
                nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])

                nxt_pos = self.to_pos(nxt)
                if not self.in_bounds(nxt_pos):
                    continue
                if not self.edge_free(cur_pos, nxt_pos):
                    continue

                # Calculate tentative g-value
                tentative_g = self._g_score[cur] + step_cost * self.res

                # Update if better g-value found
                if tentative_g < self._g_score.get(nxt, np.inf):
                    self._g_score[nxt] = tentative_g
                    self._parent[nxt] = cur

                    # If nxt was already closed in this query, add to INCONS
                    if nxt in closed_this_query:
                        self._incons.add(nxt)
                    else:
                        # Add to frontier and push to OPEN
                        self._frontier.add(nxt)
                        h = diagonal_heuristic(nxt, g_idx, self.res)
                        f = tentative_g + epsilon_start * h
                        heapq.heappush(open_heap, (f, counter, nxt))
                        counter += 1

        # OPEN exhausted or expansion limit reached
        if best_node is not None and best_node in self._g_score:
            self.last_status = "partial_exhausted"
            return self._finish(best_node, start, goal, append_goal=False), \
                   self._g_score[best_node], expansions

        self.last_status = "failed"
        return None, float('inf'), expansions

    def _finish(self, end_idx, start_pos, goal_pos, append_goal=True):
    
        # Reconstruct from parent pointers
        path_idx = [end_idx]
        while self._parent[path_idx[-1]] is not None:
            path_idx.append(self._parent[path_idx[-1]])
        path_idx.reverse()

        # Convert to continuous positions
        path = [self.to_pos(i) for i in path_idx]

        # Ensure exact start position
        path[0] = np.asarray(start_pos, dtype=float)

        # Optionally append true goal position
        if append_goal:
            end_pos = self.to_pos(end_idx)
            if self.edge_free(end_pos, goal_pos):
                path.append(np.asarray(goal_pos, dtype=float))
            # else: do NOT append goal if edge is not collision-free

        return np.array(path)
