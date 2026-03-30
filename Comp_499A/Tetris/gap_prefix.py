import os,sys
# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import math
from typing import List, Sequence, Tuple

LAMBDA = ""  

def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0

def interval_to_prefix(lo: int, hi: int, domain_size: int) -> str:
    """
    Convert dyadic half-open interval [lo,hi) within [0,domain_size) to a prefix.
    domain_size must be power of two.
    """
    if not is_power_of_two(domain_size):
        raise ValueError(f"domain_size must be power of two, got {domain_size}")
    if not (0 <= lo < hi <= domain_size):
        raise ValueError(f"interval [{lo},{hi}) out of domain [0,{domain_size})")

    length = hi - lo
    if not is_power_of_two(length):
        raise ValueError(f"interval length must be power of two, got {length} for [{lo},{hi})")
    if lo % length != 0:
        raise ValueError(f"interval must be dyadically aligned: lo % length == 0 violated for [{lo},{hi})")

    if lo == 0 and hi == domain_size:
        return LAMBDA  # λ

    k = int(math.log2(domain_size))
    t = int(math.log2(length))
    prefix_len = k - t

    lo_bits = format(lo, f"0{k}b")          # k-bit binary string
    return lo_bits[:prefix_len]             # take first prefix_len bits

def gap_box_to_prefix_box(
    box: Sequence[Tuple[int, int]],
    domain_sizes: Sequence[int],
) -> Tuple[str, ...]:
    """
    Convert one n-dim numeric interval box to an n-tuple of prefixes.
    domain_sizes[i] is the domain size for dimension i.
    """
    if len(box) != len(domain_sizes):
        raise ValueError("box arity != domain_sizes arity")

    return tuple(interval_to_prefix(lo, hi, domain_sizes[i]) for i, (lo, hi) in enumerate(box))

def convert_gap_boxes(
    boxes: Sequence[Sequence[Tuple[int, int]]],
    domain_sizes: Sequence[int],
) -> List[Tuple[str, ...]]:
    return [gap_box_to_prefix_box(b, domain_sizes) for b in boxes]