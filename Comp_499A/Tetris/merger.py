"""Merger logic for child boxes in a kD-tree split"""
from __future__ import annotations
import os, sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
Bit = Union[int, None]         
Bits = Tuple[Bit, ...]
Bounds = List[Tuple[int, int]]

def _is_degenerate(cb: Bounds) -> bool:
    return any(lo == hi for (lo, hi) in cb)

def _merge_patterns_once(patterns: Set[Bits], k: int) -> Tuple[Set[Bits], bool]:
    """
    One pass of sibling merging over wildcard patterns.

    Two patterns p and q can merge on axis a if:
      - p and q are identical except at axis a
      - p[a] and q[a] are 0/1 complements (not wildcard)
    Result is same pattern with axis a = None (wildcard).
    """
    merged: Set[Bits] = set()
    used: Set[Bits] = set()
    changed = False

    # For determinism: iterate axes in order; greedy merge within each axis.
    for a in range(k):
        # Group by "key": all coordinates except axis a
        buckets: Dict[Tuple[Bit, ...], List[Bits]] = {}
        for p in patterns:
            if p in used:
                continue
            if p[a] is None:
                continue  # can't merge further on this axis if wildcard already
            key = p[:a] + p[a+1:]
            buckets.setdefault(key, []).append(p)

        for key, ps in buckets.items():
            # We want to pair patterns that differ only on axis a: 0 with 1
            have0 = [p for p in ps if p[a] == 0 and p not in used]
            have1 = [p for p in ps if p[a] == 1 and p not in used]
            # Pair them greedily
            while have0 and have1:
                p0 = have0.pop()
                p1 = have1.pop()
                used.add(p0)
                used.add(p1)
                # Create wildcarded pattern
                newp = list(p0)
                newp[a] = None
                merged.add(tuple(newp))
                changed = True

    # Any patterns not used in a merge survive
    survivors = {p for p in patterns if p not in used}
    # New merged patterns + survivors
    out = survivors | merged
    return out, changed

def merge_empty_children_kd(
    parent_bounds: Bounds,
    child_bits_list: List[Tuple[int, ...]],
    child_bounds: List[Bounds],
    child_points: List[List[Tuple[int, ...]]],
    bits_to_idx: Dict[Tuple[int, ...], int],
) -> Tuple[List[Bounds], Set[int]]:
    """
    Given a single dyadic split at one parent node:
      - determine empty children
      - merge empty siblings into larger dyadic boxes (kD)
      - return (merged_boxes, consumed_child_indices)
    """
    k = len(parent_bounds)

    # Start with empty leaf patterns (0/1^k)
    empty_leaf_patterns: Set[Bits] = set()
    for bits in child_bits_list:
        idx = bits_to_idx[bits]
        if not child_points[idx] and not _is_degenerate(child_bounds[idx]):
            empty_leaf_patterns.add(tuple(bits))  # promote to Bits (ints only)

    if not empty_leaf_patterns:
        return [], set()

    # Iteratively merge using wildcards (None)
    patterns: Set[Bits] = empty_leaf_patterns
    while True:
        patterns, changed = _merge_patterns_once(patterns, k)
        if not changed:
            break

    # Convert wildcard patterns -> merged boxes, and compute which leaf children they cover
    merged_boxes: List[Bounds] = []
    consumed: Set[int] = set()

    # Precompute per-axis half-intervals from parent_bounds and child_bounds via bits.
    axis_half: List[Dict[int, Tuple[int, int]]] = [dict() for _ in range(k)]
    for d in range(k):
        # pick two children that differ only in axis d, others 0
        bits0 = [0] * k
        bits1 = [0] * k
        bits1[d] = 1
        b0 = tuple(bits0)
        b1 = tuple(bits1)
        if b0 in bits_to_idx and b1 in bits_to_idx:
            axis_half[d][0] = child_bounds[bits_to_idx[b0]][d]
            axis_half[d][1] = child_bounds[bits_to_idx[b1]][d]
        else:
            # fallback: derive from parent midpoint if needed
            lo, hi = parent_bounds[d]
            mid = (lo + hi) // 2
            axis_half[d][0] = (lo, mid)
            axis_half[d][1] = (mid, hi)

    def expand_pattern(p: Bits) -> Iterable[Tuple[int, ...]]:
        """Expand wildcard pattern to all leaf (0/1)^k bit tuples."""
        choices = []
        for b in p:
            if b is None:
                choices.append([0, 1])
            else:
                choices.append([b])
        # Cartesian product
        stack = [[]]
        for opts in choices:
            new_stack = []
            for pref in stack:
                for o in opts:
                    new_stack.append(pref + [o])
            stack = new_stack
        for bits in stack:
            yield tuple(bits)

    for p in patterns:
        # Build merged box bounds from pattern p
        box: Bounds = []
        for d, b in enumerate(p):
            if b is None:
                box.append(parent_bounds[d])            # wildcard => full axis span
            else:
                box.append(axis_half[d][int(b)])        # fixed bit => half span
        merged_boxes.append(box)

        # Mark covered leaf children as consumed
        for leaf_bits in expand_pattern(p):
            consumed.add(bits_to_idx[leaf_bits])
    return merged_boxes, consumed