import sys
import os

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np 
import pandas as pd
from typing import List, Sequence, Tuple, Optional
from data_layer.search_space import SearchSpace
from Tetris.cds import  MultilevelCDS
from Tetris. n_decomposition import NDDecomposer
from Tetris.gap_prefix import convert_gap_boxes
from Tetris.tetris import PrefixBox, tetris
from cnfgen.families import pigeonhole, randomformulas, pebbling, graphisomorphism, tseitin
from pysat.formula import CNF as PyCNF

n, m = 3,4

def cnf_prep():
    CNF = pigeonhole.PigeonholePrinciple(n, m)
    formula = PyCNF(from_string=CNF.to_dimacs())
    return formula.clauses, formula.nv

def bits_from_domain_size(domain_size: int) -> int:
    if domain_size <= 0 or (domain_size & (domain_size - 1)) != 0:
        raise ValueError(f"domain_size must be a positive power of two, got {domain_size}")
    return domain_size.bit_length() - 1  # since domain_size = 2^bits


def prefix_point_to_ints(p: PrefixBox, widths: Sequence[int]) -> Tuple[int, ...]:
    if not p.is_unit(widths):
        raise ValueError(f"Not a unit point: {p}")
    if len(widths) != p.dim():
        raise ValueError(f"widths length mismatch: {len(widths)} vs dim {p.dim()}")
    # coords are already bitstrings; int(s,2) works regardless of width,
    # but this asserts each coord has the expected length via is_unit().
    return tuple(int(s, 2) for s in p.coords)


def prefix_point_to_ints_uniform(p: PrefixBox, bits_per_attr: int) -> Tuple[int, ...]:
    widths = [bits_per_attr] * p.dim()
    return prefix_point_to_ints(p, widths)

def get_relation_attributes(matrix, attributes):
    relations = []
    for row in matrix:
        attrs = [attr for attr, bit in zip(attributes, row) if bit == 1]
        relations.append(attrs)
    return relations


def make_space(attrs, domain_lengths=None):
    dims = [domain_lengths[a] for a in attrs]
    space = SearchSpace(dims=dims)

    space.add_hard_box([(13, 16), (0, 3)])
    space.add_attractor(center=(2, 2), weight=4.5, spread=3.0)
    space.add_repeller(center=(3, 12), strength=3.0, spread=2.0)

    return space

def compute_local_gaps(df, relation_attrs, domain_lengths):
    """
    df: dataframe of sampled tuples for this relation
    relation_attrs: ["v1","v2"] etc.
    domain_lengths: global map of domain sizes per attribute
    """

    # Build domain box for NDNode
    domain_box = [(0, domain_lengths[attr]) for attr in relation_attrs]

    # Build list of points
    points = [tuple(row[attr] for attr in relation_attrs) 
            for idx, row in df.iterrows()]

    # Initialize ND decomposition root
    root = NDDecomposer(domain_box, points)


    # build
    gaps = root.build()

    return gaps     # list of gap boxes in local attribute order


def lift_gap_box(local_gap, relation_attrs, global_attrs, domain_lengths):
    """
    local_gap: list of intervals in the order of relation_attrs
    relation_attrs: attributes in this relation (e.g. ["v1","v2"])
    global_attrs: all attributes (e.g. ["v1","v2","v3"])
    """
    local_map = dict(zip(relation_attrs, local_gap))
    lifted = []

    for g in global_attrs:
        if g in local_map:
            lifted.append(local_map[g])
        else:
            lifted.append((0, domain_lengths[g]))  # full domain for missing attr
    assert len(local_gap) == len(relation_attrs)
    assert all(a in domain_lengths for a in global_attrs)
    return lifted


def next_pow2(n):
        return 1 << (n - 1).bit_length()

def canon_box(box):
    return tuple(tuple(iv) for iv in box)


if __name__ == "__main__":
    test = "test"
    print(test)