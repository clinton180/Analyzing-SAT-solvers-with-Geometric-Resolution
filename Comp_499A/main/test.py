"""
    Tests for generating different CSP families.
    We convert a CSP instance to forms acceptable by Tetris and SAT solvers,
    and verify that both solvers return the same set of solutions.
    For the Tetris solver, we use gap boxes derived from CNF formulas.
    For SAT solvers, we use CNF clauses.

"""
from __future__ import annotations

import json
import os
import sys

# add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sample import * 
from main.utils import *
from typing import Dict, Iterable, Iterator, List, Tuple, Optional
from SAT.sat import *
from SAT.sat_gap import cnf_to_gapboxes_raw
# from Tetris.d_tetris import PrefixBox, tetris
from Tetris.gap_prefix import convert_gap_boxes

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Protocol, Any
import matplotlib.pyplot as plt
import pandas as pd

Box = List[Tuple[int, int]]

def run_tetris_on_gaps(
    global_gaps: Sequence[Sequence[Tuple[int, int]]],
    dyadic_dims: Sequence[int],
    seed_A: bool = True,
):
    """
    sao:
      - None -> default static SAO [0..n-1]
      - list/tuple of dims -> static SAO
      - SAOPolicy object -> dynamic SAO (watched-literals, KB-based, etc.)
    """

    global B

    def _is_pow2(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    if not all(_is_pow2(int(s)) for s in dyadic_dims):
        raise ValueError(f"dyadic_dims must be powers of 2 (domain sizes). got {dyadic_dims}")

    D = len(dyadic_dims)
    for b in global_gaps:
        if len(b) != D:
            raise ValueError(f"gapbox dim {len(b)} != domain dim {D}: {b}")
        for d, (lo, hi) in enumerate(b):
            if not (0 <= lo < hi <= dyadic_dims[d]):
                raise ValueError(f"gap interval {(lo,hi)} out of range [0,{dyadic_dims[d]}] at dim {d}")

    bits_per_dim = [bits_from_domain_size(s) for s in dyadic_dims]
    prefix_tuples = convert_gap_boxes(global_gaps, dyadic_dims)
    B = [PrefixBox(tuple(t)) for t in prefix_tuples]
    if not B:
        return []

    cds = MultilevelCDS(num_dims=len(B[0].coords))
    for b in B:
        cds.insert_box_prefixes(list(b.coords))

    def oracle(w: PrefixBox):
        tuples = cds.boxes_containing_prefix_box(list(w.coords))
        return [PrefixBox(t) for t in set(tuples)]

    n = len(dyadic_dims)

    # init_A selection
    init_A = None
    if seed_A:
        init_A = [b for b in B if "" in b.coords]
    init_A = B  # your current choice

    # Call a tetris variant that accepts a policy
    return tetris(
        B=B,
        sao = None,  # default static SAO (can be customized if needed) 
        widths=bits_per_dim,
        init_A=init_A,
        oracle=oracle,
        halt_first=True,
    )


def compute_global_gaps_from_csv(
    csv_path: str,
    relation_attrs: Sequence[str],
    global_attrs: Sequence[str],
    domain_lengths: Dict[str, int],
    *,
    out_png: str = "relation_global_space.png",
    deduplicate: bool = True,
) -> tuple[list[Box], list[int]]:
    
    csv_path = str(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    # --- 1. load relation ---
    df = pd.read_csv(csv_path)

    # keep only requested columns, in requested order
    missing = [a for a in relation_attrs if a not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[list(relation_attrs)].copy()

    # ensure plain ints if possible
    for col in relation_attrs:
        df[col] = df[col].astype(int)

    # compute local gaps 
    local_gaps = compute_local_gaps(df, relation_attrs, domain_lengths)

    # lift to global dimensionality 
    global_gaps: list[Box] = []
    for g in local_gaps:
        lifted = lift_gap_box(g, relation_attrs, global_attrs, domain_lengths)
        global_gaps.append(lifted)

    # deduplicate
    if deduplicate:
        gap_set = set(canon_box(b) for b in global_gaps)
        global_gaps = [list(map(tuple, b)) for b in gap_set]

    print(f"{len(global_gaps)} global gaps after lifting")

    # dyadic dimensions + visualization
    dyadic_dims = [next_pow2(domain_lengths[a]) for a in global_attrs]
    global_space = SearchSpace(dyadic_dims)
    global_space.add_hard_boxes(global_gaps)

    fig = global_space.visualize()
    if fig is not None:
        print(f"Saving visualization to {out_png}")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return global_gaps, dyadic_dims


def tetris_prep():
    structure_matrix = np.array([
        [1, 1, 0],   # h1
        [0, 1, 1],   # h2
        [1, 0, 1]    # h3
    ])
    num_attrs = structure_matrix.shape[1]
    attributes = [f"v{i+1}" for i in range(num_attrs)]
    relations = get_relation_attributes(structure_matrix, attributes)
    domain_lengths = {f"v{i+1}": 16 for i in range(num_attrs)}
    relation_dfs = []    # <-- Store each relation dataframe here
    global_gaps = []     # <-- Store all lifted global gaps here
    num_samples = 7

    for i, attrs in enumerate(relations, start=1):
        print(f"Generating relation_{i} with attributes {attrs}")
        space = make_space(attrs, domain_lengths=domain_lengths)
        samples = space.sample_gibbs(num_samples, temperature=1)
        df = pd.DataFrame(samples, columns=attrs)
        df.to_csv(f"relation_{i}.csv", index=False)
        relation_dfs.append(df)
        print(f"Saved relation_{i}.csv")

    for relation_attrs, df in zip(relations, relation_dfs):
        local_gaps = compute_local_gaps(df, relation_attrs, domain_lengths)

        for g in local_gaps:
            lifted = lift_gap_box(g, relation_attrs, attributes, domain_lengths)
            global_gaps.append(lifted)

    # Deduplicate 
    gap_set = set(canon_box(b) for b in global_gaps)
    global_gaps = [list(map(tuple, b)) for b in gap_set]  # deduplicate

    # Pad each attribute to next power of 2 (for dyadic decomposition)
    print(len(global_gaps), "global gaps after lifting and deduplication")
    dyadic_dims = [next_pow2(domain_lengths[a]) for a in attributes]
    global_space = SearchSpace(dyadic_dims)
    global_space.add_hard_boxes(global_gaps)
    fig = global_space.visualize()
    if fig is not None:
        print("Saving global space visualization to global_space.png")
        fig.savefig("global_space.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return global_gaps, dyadic_dims


def sample_from_cnf():
    clauses, nv = cnf_prep()
    grouping = [1]*nv  
    ordering = None  # identity ordering (or any permutation of [1..nv])
    gmap, gapboxes = cnf_to_gapboxes_raw(clauses, grouping=grouping, ordering=ordering)
    
    # domain sizes derived from widths
    domain = [1 << w for w in gmap.widths]

    outputs = run_tetris_on_gaps(gapboxes, domain, seed_A=True)
    models = []
    model = SAT_test(clauses)   
    models.append(model if model is not None else [])
    widths = list(gmap.widths)
    
    decoded = [decode_model_to_unit_box_by_widths(m, widths) for m in models]
    print("SAT models decoded to unit boxes:", decoded)

    if len(models) > 0:
        out_set = sorted([x.coords for x in outputs])
        print(len(out_set), "solutions")
    
    if len(models) == 0:
        assert len(outputs) == 0, "CNF is UNSAT but Tetris returned nonempty outputs"
        print("OK: CNF and Tetris both UNSAT.")
    return [prefix_point_to_ints(p, widths) for p in outputs]


def sample_from_tetris():
    global_gaps, dyadic_dims = tetris_prep()
    widths = [bits_from_domain_size(s) for s in dyadic_dims]
    outputs = run_tetris_on_gaps(global_gaps, dyadic_dims, seed_A=True)
    clauses = gapboxes_to_clauses(B, widths)
    nv = write_dimacs_cnf(clauses, "instance.cnf")
    print("wrote instance.cnf with nv =", nv, "and m =", len(clauses))
    models = []
    model = SAT_test(clauses)   
    models.append(model if model is not None else [])
    models = [decode_model_to_unit_box_by_widths(m, widths) for m in models]

    if len(outputs) > 0:
        out_set = sorted([x.coords for x in outputs])
        print(len(out_set), "solutions")

    if not outputs:
        print("No outputs from Tetris.")

    # return integers per coord (generic)
    return [tuple(int(s, 2) for s in p.coords) for p in outputs]

def main():
    domain_lengths = {"v1": 16, "v2": 16}
    global_attrs = ["v1", "v2"]

    # suppose relation_1.csv has columns v1,v2
    global_gaps, dyadic_dims = compute_global_gaps_from_csv(
        csv_path="relation_1.csv",
        relation_attrs=["v1", "v2"],
        global_attrs=global_attrs,
        domain_lengths=domain_lengths,
        out_png="relation_1_global_space.png",
    )
    widths = [bits_from_domain_size(s) for s in dyadic_dims]
    outputs = run_tetris_on_gaps(global_gaps, dyadic_dims, seed_A=True)

    clauses = gapboxes_to_clauses(B, widths)
    nv = write_dimacs_cnf(clauses, "instance.cnf")
    print("wrote instance.cnf with nv =", nv, "and m =", len(clauses))
    print("Dyadic dims:", dyadic_dims)

if __name__ == "__main__":
    # test = sample_from_tetris()
    # if len(test) > 0:
    #     print("Sample output:")
    #     print(test[0])
    # else:
    #     print("No points sampled.")
    main()