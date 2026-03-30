from __future__ import annotations

import os, sys
from dataclasses import dataclass
from itertools import product
from typing import List, Optional, Sequence, Tuple

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Tetris.tetris import PrefixBox

# CNF basics

def infer_num_vars(clauses: List[List[int]]) -> int:
    return max((abs(lit) for c in clauses for lit in c), default=0)

def is_tautology(clause: List[int]) -> bool:
    s = set(clause)
    return any(-lit in s for lit in s)

# Clause -> (0,1,x)^n cube (x = None)
def clause_to_cube(
    clause: List[int],
    n: int,
    ordering: Optional[Sequence[int]] = None,
) -> Optional[List[Optional[int]]]:
    """
    Clause -> falsifying partial assignment cube over {0,1,x}^n.

    +v in clause => falsifier sets x_v = 0
    -v in clause => falsifier sets x_v = 1

    ordering:
        Option A representation:
            ordering[pos] = old_var
        If None:
            fixed ordering 1..n is used.
    """
    if len(clause) == 0:
        return [None] * n
    if is_tautology(clause):
        return None

    # Build old_var -> cube_index map
    if ordering is None:
        # identity mapping
        def var_to_index(v: int) -> int:
            return v - 1
    else:
        if len(ordering) != n:
            raise ValueError("ordering length must equal n")

        old_to_new = [0] * (n + 1)
        for idx, old_var in enumerate(ordering):
            old_to_new[old_var] = idx

        def var_to_index(v: int) -> int:
            return old_to_new[v]

    cube: List[Optional[int]] = [None] * n

    for lit in clause:
        v = abs(lit)
        if not (1 <= v <= n):
            raise ValueError(f"Literal var {v} out of range 1..{n}")

        idx = var_to_index(v)
        cube[idx] = 0 if lit > 0 else 1

    return cube

# Grouping as list of ints (bit widths per dimension)

def normalize_grouping(grouping: Sequence[int], n: int) -> List[int]:
    """
      - grouping is a list of positive ints (bit-widths)
      - if sum < n: append remainder as extra dimension
      - if sum > n: reduce/remove from the end until sum == n
    """
    g = [int(x) for x in grouping if int(x) > 0]
    s = sum(g)

    if s < n:
        g.append(n - s)
        return g

    if s > n:
        excess = s - n
        i = len(g) - 1
        while excess > 0 and i >= 0:
            take = min(g[i], excess)
            g[i] -= take
            excess -= take
            if g[i] == 0:
                g.pop(i)
            i -= 1
        if sum(g) != n:
            raise ValueError("Failed to normalize grouping to sum to n.")
        return g

    return g

@dataclass(frozen=True)
class GroupMap:
    """
    Maps POSITION p in 0..n-1 to (dim, bit) where bit is within that dim.
    Position order is the grouping layout order (bit 0 is first slot).
    """
    n: int
    widths: Tuple[int, ...]
    pos_to_dim_bit: Tuple[Tuple[int, int], ...]  # index by p (0..n-1)

    @property
    def dims(self) -> int:
        return len(self.widths)


def build_group_map(grouping: Sequence[int], n: int) -> GroupMap:
    widths = tuple(normalize_grouping(grouping, n))
    lut: List[Tuple[int, int]] = [(0, 0)] * n  # index by position p=0..n-1

    p = 0
    for d, w in enumerate(widths):
        for b in range(w):
            lut[p] = (d, b)
            p += 1

    if p != n:
        raise ValueError(f"Internal error: mapping filled positions up to {p-1}, expected {n-1}.")
    return GroupMap(n=n, widths=widths, pos_to_dim_bit=tuple(lut))


# Cube -> grouped masked strings (0/1/x per bit)
def cube_to_grouped_masks(cube: List[Optional[int]], gmap: GroupMap) -> List[str]:
    """
    Convert cube over {0,1,x}^n into per-dimension masks.
    Cube index p corresponds to position p in the chosen ordering.
    """
    if len(cube) != gmap.n:
        raise ValueError("cube length does not match mapping n")

    dim_bits: List[List[str]] = [list("x" * w) for w in gmap.widths]
    for p, val in enumerate(cube):
        if val is None:
            continue
        d, b = gmap.pos_to_dim_bit[p]
        dim_bits[d][b] = "1" if val == 1 else "0"
    return ["".join(bits) for bits in dim_bits]

# Prefix -> interval
def prefix_to_interval(prefix: str, width: int) -> Tuple[int, int]:
    """Half-open interval [lo, hi) inside [0, 2^width)."""
    if prefix == "":
        return (0, 1 << width)
    L = len(prefix)
    if L > width:
        raise ValueError(f"Prefix length {L} exceeds dimension width {width}")
    v = int(prefix, 2)
    shift = width - L
    lo = v << shift
    hi = (v + 1) << shift
    return (lo, hi)


# Mask -> dyadic intervals directly (no PrefixBox objects)
def mask_to_intervals(mask: str, width: int) -> List[Tuple[int, int]]:
    """
    Convert one grouped mask string over {0,1,x} into a list of DYADIC intervals [lo,hi).

    Same dyadic rule as mask_to_prefixes:
      - Let k = highest index with mask[k] in {0,1}.
      - If no fixed bits => full interval [(0,2^width))
      - If any 'x' appears in positions 0..k => split on those missing bits (all combos)
      - Intervals correspond to prefixes of length k+1
    """
    fixed_positions = [i for i, ch in enumerate(mask) if ch in ("0", "1")]
    if not fixed_positions:
        return [(0, 1 << width)]  # full dimension

    k = max(fixed_positions)
    missing = [i for i in range(0, k + 1) if mask[i] == "x"]

    base: List[Optional[str]] = []
    for i in range(0, k + 1):
        ch = mask[i]
        base.append(ch if ch in ("0", "1") else None)

    if not missing:
        prefix = "".join(base)  # type: ignore[arg-type]
        return [prefix_to_interval(prefix, width)]

    out: List[Tuple[int, int]] = []
    for bits in product("01", repeat=len(missing)):
        tmp = base[:]
        for idx, pos in enumerate(missing):
            tmp[pos] = bits[idx]
        prefix = "".join(tmp)  # type: ignore[arg-type]
        out.append(prefix_to_interval(prefix, width))
    return out

# Clause -> raw gapboxes directly
def clause_to_gapboxes_raw(
    clause: List[int],
    *,
    n: int,
    gmap: GroupMap,
    ordering: Optional[Sequence[int]] = None,
    drop_tautologies: bool = True,
) -> List[List[Tuple[int, int]]]:
    """
    Clause -> list of gapboxes in YOUR format:
      [
        [(lo0,hi0), (lo1,hi1), ...],
        ...
      ]
    Generates dyadic splitting directly as numeric intervals; no PrefixBox objects.
    """
    cube = clause_to_cube(clause, n, ordering=ordering)
    if cube is None:
        return [] if drop_tautologies else [[(0, 1 << w) for w in gmap.widths]]

    masks = cube_to_grouped_masks(cube, gmap)

    per_dim_intervals: List[List[Tuple[int, int]]] = [
        mask_to_intervals(masks[d], gmap.widths[d]) for d in range(gmap.dims)
    ]

    return [list(coords) for coords in product(*per_dim_intervals)]

# CNF -> raw gapboxes (with dedup)
def cnf_to_gapboxes_raw(
    clauses: List[List[int]],
    grouping: Sequence[int],
    *,
    ordering: Optional[Sequence[int]] = None,
    n: Optional[int] = None,
    drop_tautologies: bool = True,
    dedup: bool = True,
) -> Tuple[GroupMap, List[List[Tuple[int, int]]]]:
    if n is None:
        n = infer_num_vars(clauses)

    gmap = build_group_map(grouping, n)

    out: List[List[Tuple[int, int]]] = []
    seen: set[Tuple[Tuple[int, int], ...]] = set()

    for c in clauses:
        for gb in clause_to_gapboxes_raw(
            c, n=n, gmap=gmap, ordering=ordering, drop_tautologies=drop_tautologies
        ):
            if not dedup:
                out.append(gb)
                continue
            key = tuple(gb)
            if key not in seen:
                seen.add(key)
                out.append(gb)
    return gmap, out
