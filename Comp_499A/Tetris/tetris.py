from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from Tetris.cds import MultilevelCDS

LAMBDA = ""  # λ is the empty prefix


@dataclass(frozen=True)
class PrefixBox:
    """
    A box is an n-tuple of prefixes over {0,1} plus λ = "".
    Example (2D): <λ, 0>  is ("", "0")
    """
    coords: Tuple[str, ...]

    def dim(self) -> int:
        return len(self.coords)

    def is_unit(self, widths: Sequence[int]) -> bool:
        """Unit box = point: in each dimension d, prefix has full length widths[d]."""
        if len(widths) != self.dim():
            raise ValueError(f"widths length mismatch: {len(widths)} vs dim {self.dim()}")
        return all(len(p) == widths[d] for d, p in enumerate(self.coords))

    def contains(self, other: "PrefixBox") -> bool:
        """
        a ⊇ b  iff  for all i, a_i is a prefix of b_i
        (λ is prefix of everything).
        """
        if self.dim() != other.dim():
            return False
        for a, b in zip(self.coords, other.coords):
            if not b.startswith(a):
                return False
        return True

    def __repr__(self) -> str:
        parts = [("λ" if p == "" else p) for p in self.coords]
        return f"<{', '.join(parts)}>"



# JSONL tracer (buffered) + iter support
class JsonlTracer:
    """
    Writes one JSON object per line (JSONL) to a file.
    Uses a monotone counter 't' for stable ordering.
    Adds optional fields: iter (outer loop), depth (recursion depth).
    """

    def __init__(self, path: str, enabled: bool = True, flush_every: int = 1000) -> None:
        self.path = path
        self.enabled = enabled
        self.flush_every = max(1, flush_every)
        self.t = 0
        self.iter = 0  # outer loop iteration in tetris()
        self._fh = open(path, "w", encoding="utf-8") if enabled else None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def emit(self, event: str, **data: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        rec: Dict[str, Any] = {"t": self.t, "event": event, "iter": self.iter}
        rec.update(data)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.t += 1
        if (self.t % self.flush_every) == 0:
            self._fh.flush()


def _box_str(b: Optional[PrefixBox]) -> Optional[str]:
    return None if b is None else repr(b)


# Knowledge base A (global set of boxes)
class KnowledgeBase:
    def __init__(self, num_dims: int) -> None:
        self._num_dims = num_dims
        self._cds = MultilevelCDS(num_dims=num_dims)
        self._size = 0

    def add_many(self, boxes: Iterable["PrefixBox"]) -> None:
        for b in boxes:
            self.add(b)

    def add(self, box: "PrefixBox") -> None:
        coords = list(box.coords)
        if len(coords) != self._num_dims:
            raise ValueError(f"Box dim mismatch: expected {self._num_dims}, got {len(coords)}")
        self._cds.insert_box_prefixes(coords)
        self._size += 1

    def find_covering_box(self, b: "PrefixBox") -> Optional["PrefixBox"]:
        """
        Return some a in A such that a ⊇ b, if it exists. Otherwise return None.
        """
        coords = list(b.coords)
        if len(coords) != self._num_dims:
            raise ValueError(f"Query dim mismatch: expected {self._num_dims}, got {len(coords)}")
        hits, active = self._cds.any_box_containing_prefix_box(coords)
        # print(active)
        return PrefixBox(tuple(hits)) if hits is not None else None

    def __len__(self) -> int:
        return self._size


# Split-First-Thick-Dimension(b)
def split_first_thick_dimension(
    b: PrefixBox,
    sao: Sequence[int],
    widths: Sequence[int],
) -> Tuple[PrefixBox, PrefixBox, int]:
    """
    Cut b into two equal halves by extending the first (per SAO) dimension
    whose prefix length is < widths[d].

    Returns (b1, b2, split_dim).
    """
    n = b.dim()
    if len(widths) != n:
        raise ValueError(f"widths length mismatch: {len(widths)} vs dim {n}")
    if sorted(sao) != list(range(n)):
        raise ValueError("SAO must be a permutation of [0..n-1].")

    coords = list(b.coords)
    for d in sao:
        if len(coords[d]) < widths[d]:
            p = coords[d]
            coords1 = coords.copy()
            coords2 = coords.copy()
            coords1[d] = p + "0"
            coords2[d] = p + "1"
            return PrefixBox(tuple(coords1)), PrefixBox(tuple(coords2)), d

    raise ValueError("No thick dimension found to split (already unit).")


# Resolve(w1, w2)  (ordered geometric resolution)
def resolve(
    w1: PrefixBox,
    w2: PrefixBox,
    split_dim: int,
) -> PrefixBox:
    """
    Ordered geometric resolution along split_dim.
    Precondition: w1 and w2 are siblings in split_dim.
    """
    if w1.dim() != w2.dim():
        raise ValueError("Resolve: dimension mismatch")

    n = w1.dim()
    a = w1.coords[split_dim]
    b = w2.coords[split_dim]

    # sibling check in split_dim: same length, differ only in last bit, share parent
    if len(a) != len(b) or len(a) == 0:
        raise ValueError(f"Resolve: not siblings in split_dim={split_dim}: {a!r}, {b!r}")
    if a[:-1] != b[:-1]:
        raise ValueError(f"Resolve: not siblings in split_dim={split_dim}: {a!r}, {b!r}")
    if {a[-1], b[-1]} != {"0", "1"}:
        raise ValueError(f"Resolve: not siblings in split_dim={split_dim}: {a!r}, {b!r}")

    parent = a[:-1]

    out: List[str] = []
    for j in range(n):
        if j == split_dim:
            out.append(parent)
            continue
        p1 = w1.coords[j]
        p2 = w2.coords[j]
        # comparable by prefix order
        if p1.startswith(p2):
            out.append(p1)  # p1 more specific
        elif p2.startswith(p1):
            out.append(p2)  # p2 more specific
        else:
            raise ValueError(f"Resolve: non-comparable prefixes in dim={j}: {p1!r} vs {p2!r}")
    return PrefixBox(tuple(out))


# Algorithm 1: TetrisSkeleton(b)   (traced + depth)
def tetris_skeleton(
    KB: "KnowledgeBase",
    sao: Sequence[int],
    widths: Sequence[int],
    b: "PrefixBox",
    tracer: Optional[JsonlTracer] = None,
    depth: int = 0,
) -> Tuple[bool, "PrefixBox"]:
    """
    Output: (v, w)
      - if v is TRUE:  w is a cover box for b
      - if v is FALSE: w is an uncovered point (unit box) inside b
    """
    # 1: cover query
    a = KB.find_covering_box(b)
    if tracer is not None:
        tracer.emit("QUERY_COVER", depth=depth, b=_box_str(b), result=_box_str(a))
    if a is not None:
        return True, a

    # 2: uncovered unit
    if b.is_unit(widths):
        if tracer is not None:
            tracer.emit("UNCOVERED_POINT", depth=depth, w=_box_str(b))
        return False, b

    # 3: split
    b1, b2, split_dim = split_first_thick_dimension(b, sao, widths)
    if tracer is not None:
        tracer.emit(
            "SPLIT",
            depth=depth,
            b=_box_str(b),
            dim=split_dim,
            left=_box_str(b1),
            right=_box_str(b2),
        )

    # 4
    v1, w1 = tetris_skeleton(KB, sao, widths, b1, tracer=tracer, depth=depth + 1)
    if v1 is False:
        return False, w1
    if w1.contains(b):
        if tracer is not None:
            tracer.emit("RETURN_COVER", depth=depth, b=_box_str(b), w=_box_str(w1))
        return True, w1

    # 5
    v2, w2 = tetris_skeleton(KB, sao, widths, b2, tracer=tracer, depth=depth + 1)
    if v2 is False:
        return False, w2
    if w2.contains(b):
        if tracer is not None:
            tracer.emit("RETURN_COVER", depth=depth, b=_box_str(b), w=_box_str(w2))
        return True, w2

    # 6: resolve + add to KB
    w = resolve(w1, w2, split_dim)
    if tracer is not None:
        tracer.emit("RESOLVE", depth=depth, dim=split_dim, w1=_box_str(w1), w2=_box_str(w2), out=_box_str(w))
        tracer.emit("ADD_BOX", depth=depth, w=_box_str(w), source="resolve", kb_size_before=len(KB))
    KB.add(w)
    if tracer is not None:
        tracer.emit("KB_SIZE", depth=depth, kb_size_after=len(KB))
    return True, w


# Algorithm 2: Tetris(B)   
def tetris(
    B: Sequence["PrefixBox"],
    widths: Sequence[int],
    sao: Sequence[int],
    init_A: Optional[Iterable["PrefixBox"]] = None,
    oracle: Optional[object] = None,
    trace_path: str = "tetris_trace.jsonl",
    trace_enabled: bool = True,
    trace_flush_every: int = 1000,
    halt_first: bool = False,  
) -> List["PrefixBox"]:
    """
    Tracing:
      - writes JSONL to trace_path automatically when trace_enabled=True
      - each record includes:
          - t: global event counter
          - iter: outer loop iteration counter
          - depth: recursion depth for skeleton events (when applicable)

    halt_first:
      - if True, stop at the first OUTPUT and return [w]
    """
     
    if len(B) == 0:
        return []

    n = B[0].dim()
    for b in B:
        if b.dim() != n:
            raise ValueError("All boxes in B must have the same dimension.")
    if sao is None: 
        sao = list(range(n)) 

    if len(widths) != n:
        raise ValueError(f"widths length mismatch: {len(widths)} vs dim {n}")
    if any(w < 0 for w in widths):
        raise ValueError(f"widths must be nonnegative: {widths}")
    if sorted(sao) != list(range(n)):
        raise ValueError("SAO must be a permutation of [0..n-1].")

    tracer = JsonlTracer(trace_path, enabled=trace_enabled, flush_every=trace_flush_every)
    try:
        KB = KnowledgeBase(num_dims=n)
        tracer.emit("START", n=n, widths=list(widths), sao=list(sao), B_size=len(B), halt_first=halt_first)

        # 1: Initialize(A)
        if init_A is not None:
            init_list = list(init_A)
            KB.add_many(init_list)
            # for bb in init_list:
            #     tracer.emit("ADD_BOX", depth=None, w=_box_str(bb), source="init_A", kb_size_before=None)
            tracer.emit("KB_SIZE", depth=None, kb_size_after=len(KB))

        full = PrefixBox(tuple(LAMBDA for _ in range(n)))

        v, w = tetris_skeleton(KB, sao, widths, full, tracer=tracer, depth=0)

        outputs: List[PrefixBox] = []

        outer = 0
        while v is False:
            outer += 1
            tracer.iter = outer  # all subsequent events tagged with this outer iteration

            # B' <- {b in B | b ⊇ w}
            if oracle is None:
                B_prime = [b for b in B if b.contains(w)]
            else:
                B_prime = oracle(w)

            tracer.emit(
                "ORACLE",
                depth=None,
                w=_box_str(w),
                B_prime_size=len(B_prime),
                used_oracle=(oracle is not None),
            )

            if len(B_prime) == 0:
                outputs.append(w)
                tracer.emit("OUTPUT", depth=None, w=_box_str(w))

                if halt_first:
                    tracer.emit("DONE", depth=None, outputs=len(outputs), kb_size=len(KB), halted_first=True)
                    return outputs  

                B_prime = [w]

            KB.add_many(B_prime)
            for bb in B_prime:
                tracer.emit("ADD_BOX", depth=None, w=_box_str(bb), source="B_prime", kb_size_before=None)
            tracer.emit("KB_SIZE", depth=None, kb_size_after=len(KB))

            v, w = tetris_skeleton(KB, sao, widths, full, tracer=tracer, depth=0)

        tracer.emit("DONE", depth=None, outputs=len(outputs), kb_size=len(KB), halted_first=False)
        return outputs
    finally:
        tracer.close()
