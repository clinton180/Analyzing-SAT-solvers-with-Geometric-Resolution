# utilities to convert between gap boxes (PrefixBox) and SAT CNF clauses
from __future__ import annotations
import os
import sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Sequence
from Tetris.tetris import PrefixBox

from typing import Sequence, Optional, List, Dict

def make_pos_to_var(n: int, ordering: Optional[Sequence[int]]) -> List[int]:
    """
    Returns pos_to_var of length n where pos_to_var[p] = DIMACS var id at position p.
    If ordering is None: pos_to_var[p] = p+1.
    Option A ordering: ordering[p] = old_var at new position p.
    """
    if ordering is None:
        return [p + 1 for p in range(n)]
    if len(ordering) != n:
        raise ValueError("ordering length must equal n")
    return list(ordering)

def make_var_to_pos(n: int, ordering: Optional[Sequence[int]]) -> List[int]:
    """
    Returns var_to_pos such that var_to_pos[var] = position p.
    If ordering is None: var_to_pos[var] = var-1.
    """
    if ordering is None:
        vt = [0] * (n + 1)
        for v in range(1, n + 1):
            vt[v] = v - 1
        return vt
    if len(ordering) != n:
        raise ValueError("ordering length must equal n")
    vt = [0] * (n + 1)
    for p, v in enumerate(ordering):
        vt[v] = p
    return vt

# Gap box -> CNF clause conversions
def var_id_by_offsets(
    offsets: Sequence[int],
    dim: int,
    bit: int,
    *,
    pos_to_var: Sequence[int],
) -> int:
    """
    Map (dim, bit) -> DIMACS var ID using block offsets over POSITIONS.
    """
    pos = offsets[dim] + bit  # 0-based position
    return pos_to_var[pos]


def compute_offsets(widths: Sequence[int]) -> List[int]:
    offsets: List[int] = []
    s = 0
    for w in widths:
        offsets.append(s)
        s += w
    return offsets

def gapbox_to_clause_by_widths(
    box: PrefixBox,
    widths: Sequence[int],
    *,
    ordering: Optional[Sequence[int]] = None,  # Option A
) -> List[int]:
    """
    Convert ONE PrefixBox into ONE CNF clause under a positional layout defined by widths,
    then map positions to DIMACS var ids via ordering.

    ordering=None => identity (position p uses DIMACS var p+1).
    ordering=OptionA => position p uses DIMACS var ordering[p].
    """
    if box.dim() != len(widths):
        raise ValueError(f"Box dim={box.dim()} != len(widths)={len(widths)}")

    n = sum(widths)
    pos_to_var = make_pos_to_var(n, ordering)
    offsets = compute_offsets(widths)

    clause: List[int] = []
    for dim, pref in enumerate(box.coords):
        if len(pref) > widths[dim]:
            raise ValueError(f"Prefix too long in dim {dim}: {pref!r} > width={widths[dim]}")
        for bit, ch in enumerate(pref):
            v = var_id_by_offsets(offsets, dim, bit, pos_to_var=pos_to_var)
            if ch == "0":
                clause.append(+v)
            elif ch == "1":
                clause.append(-v)
            else:
                raise ValueError(f"Invalid prefix char {ch!r} in dim {dim}, prefix={pref!r}")

    return clause

def gapboxes_to_clauses(B: List[PrefixBox], widths: Sequence[int], *, ordering: Optional[Sequence[int]] = None) -> List[List[int]]:
    if not B:
        return []
    D = B[0].dim()
    if D != len(widths):
        raise ValueError(f"Boxes have dim {D} but widths has len {len(widths)}")
    for b in B:
        if b.dim() != D:
            raise ValueError("All boxes must have same dimension.")
    return [gapbox_to_clause_by_widths(b, widths, ordering=ordering) for b in B]


# SAT model output -> unit PrefixBox conversion
def decode_sat_model_to_unit_box(
    model: List[int],
    bits_per_attr: int,
    *,
    ordering: Optional[Sequence[int]] = None,  # Option A
) -> PrefixBox:
    """
    Model -> unit PrefixBox with equal bits_per_attr per attribute, respecting ordering.
    ordering length defines n; if ordering is None, n inferred from max var in model.
    """
    if bits_per_attr <= 0:
        raise ValueError("bits_per_attr must be positive")

    assign: Dict[int, int] = {}
    max_var = 0
    for lit in model:
        v = abs(lit)
        max_var = max(max_var, v)
        assign[v] = 1 if lit > 0 else 0

    n = len(ordering) if ordering is not None else max_var
    if n % bits_per_attr != 0:
        raise ValueError(f"n={n} not divisible by bits_per_attr={bits_per_attr}")

    num_attrs = n // bits_per_attr
    widths = [bits_per_attr] * num_attrs
    return decode_model_to_unit_box_by_widths(model, widths, ordering=ordering)


def decode_model_to_unit_box_by_widths(
    model: List[int],
    widths: List[int],
    *,
    ordering: Optional[Sequence[int]] = None,  # Option A
) -> PrefixBox:
    """
    Model (DIMACS lits) -> unit PrefixBox where coord d has length widths[d].

    widths defines a positional layout of length n=sum(widths).
    ordering maps positions -> DIMACS var ids.

    ordering=None => dim blocks are vars 1..w0, w0+1..w0+w1, ...
    ordering=OptionA => dim blocks are vars ordering[offset+bit].
    """
    assign: Dict[int, int] = {}
    print(model)
    for lit in model:
        v = abs(lit)
        assign[v] = 1 if lit > 0 else 0

    n = sum(widths)
    pos_to_var = make_pos_to_var(n, ordering)

    # Ensure every var used by these positions is assigned
    missing = [pos_to_var[p] for p in range(n) if pos_to_var[p] not in assign]
    if missing:
        raise ValueError(f"Model missing assignments for variables: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    coords: List[str] = []
    pos = 0
    for w in widths:
        bits: List[str] = []
        for _ in range(w):
            var_id = pos_to_var[pos]
            bits.append("1" if assign[var_id] == 1 else "0")
            pos += 1
        coords.append("".join(bits))

    return PrefixBox(tuple(coords))


def write_dimacs_cnf(clauses: Sequence[Sequence[int]], path: str, num_vars: int | None = None) -> int:
    """
    clauses: list of clauses, each clause is a list of ints (e.g., [1, -5, 9])
    path: output .cnf file path
    num_vars: optional; if None, inferred as max abs literal
    returns: inferred/used num_vars
    """
    if num_vars is None:
        num_vars = max((abs(lit) for cl in clauses for lit in cl), default=0)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for cl in clauses:
            if len(cl) == 0:
                # empty clause => immediate UNSAT in DIMACS is "0" on a line
                f.write("0\n")
            else:
                f.write(" ".join(map(str, cl)) + " 0\n")
    return num_vars

def read_cnf(path):
    num_vars = 0
    clauses = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            if line.startswith("p"):
                # Example: p cnf 63 64
                _, _, num_vars, _ = line.split()
                num_vars = int(num_vars)
                continue

            clause = list(map(int, line.split()))
            if clause[-1] == 0:
                clause = clause[:-1]
            clauses.append(clause)

    return num_vars, clauses
