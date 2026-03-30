"""nD Pruning Decomposition (dyadic)"""

from __future__ import annotations
import os, sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable, Set
from Tetris.merger import merge_empty_children_kd

# NDDecomposer (pruning, dyadic)
class NDDecomposer:
    """
    Pruning ND decomposer that takes:
        - an nD dyadic domain (bounds)
        - a list of points (tuples of ints)

    And recursively splits the domain along midpoints, but:
        - If a region has no points, it is emitted as a GAP and NOT subdivided.
        - If a region contains points and is bigger than min_cell_size,
          it is split into 2^n children and restricted recursively.

    Result:
        A list of dyadic GAP boxes (maximal empty regions).
    """

    def __init__(self,
                 bounds: List[Tuple[int, int]],
                 points: List[Tuple[int, ...]],
                 min_cell_size: int = 1):
        """
        Args:
            bounds: [(lo,hi)] per dimension; must define a dyadic domain (sizes powers of 2).
            points: list of nD integer tuples; dimension must match len(bounds).
            min_cell_size: smallest cell size in each dimension; default 1 (atomic voxels).
        """
        self.bounds = bounds
        self.points = points
        self.min_cell_size = min_cell_size
        self.gaps: List[List[Tuple[int, int]]] = []

        if points:
            dim_bounds = len(bounds)
            dim_points = len(points[0])
            assert dim_bounds == dim_points, (
                f"Dimension mismatch: bounds={dim_bounds}, points={dim_points}"
            )

    def build(self) -> List[List[Tuple[int, int]]]:
        """
        Run the recursive decomposition and return all GAP boxes.
        """
        self._recurse(self.bounds, self.points)
        return self.gaps

    def _recurse(self,
                 bounds: List[Tuple[int, int]],
                 pts: List[Tuple[int, ...]]) -> None:
        """
        Recursive helper:
            - bounds: current nD box
            - pts   : points lying inside this box
        """
        # 1) If no points → this region is a GAP; record and prune.
        if not pts:
            self.gaps.append(bounds)
            return

        # 2) If region is atomic-ish (all dimensions <= min_cell_size) → stop.
        if all(hi - lo <= self.min_cell_size for (lo, hi) in bounds):
            return  # treat this as FULL / boundary; no further splitting.

        k = len(bounds)
        # 3) Compute midpoints for dyadic split.
        mids = [(lo + hi) // 2 for (lo, hi) in bounds]

        # Precompute all child "bit patterns": tuples of 0/1 of length k
        # Each pattern corresponds to: 0=lower half, 1=upper half on that axis.
        child_bits_list = list(itertools.product([0, 1], repeat=k))
        num_children = len(child_bits_list)

        # 4) Initialize child bounds and child point buckets.
        child_bounds: List[List[Tuple[int, int]]] = []
        child_points: List[List[Tuple[int, ...]]] = [[] for _ in range(num_children)]

        # Build all child bounds in the same order as child_bits_list.
        for bits in child_bits_list:
            cb: List[Tuple[int, int]] = []
            for d, bit in enumerate(bits):
                lo, hi = bounds[d]
                mid = mids[d]
                if bit == 0:
                    cb.append((lo, mid))
                else:
                    cb.append((mid, hi))
            child_bounds.append(cb)

        # Mapping bits -> child index
        bits_to_idx = {bits: idx for idx, bits in enumerate(child_bits_list)}

        # 5) Assign each point to exactly one child using the same bits convention.
        for p in pts:
            bits = []
            for d, (lo, hi) in enumerate(bounds):
                mid = mids[d]
                bits.append(0 if p[d] < mid else 1)
            bits_tuple = tuple(bits)
            child_idx = bits_to_idx[bits_tuple]
            child_points[child_idx].append(p)

        # after bucketing points (child_points filled)...
        merged_boxes, consumed = merge_empty_children_kd(
            parent_bounds=bounds,
            child_bits_list=child_bits_list,
            child_bounds=child_bounds,
            child_points=child_points,
            bits_to_idx=bits_to_idx
        )

        # emit merged boxes immediately
        self.gaps.extend(merged_boxes)

        # 6) Recurse on each child
        def is_degenerate(cb):
            return any(lo == hi for (lo, hi) in cb)

        # Recurse / emit remaining empties
        for idx, (cb, cp) in enumerate(zip(child_bounds, child_points)):
            if is_degenerate(cb):
                continue
            if not cp:
                # If this empty leaf is already covered by a merged empty box, skip it
                if idx in consumed:
                    continue
                # Otherwise emit this empty child as a gap
                self.gaps.append(cb)
                continue

            # Non-empty: recurse
            self._recurse(cb, cp)