# =========================
# Instance/Run Generator
# =========================
from __future__ import annotations

import os, sys, json, time, subprocess
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Optional, Dict, Any, Sequence, Tuple, List
import numpy as np
import pandas as pd
# from Comp_499A.data_layer.search_space import SearchSpace

from Comp_499A.data_layer.search_space_mod import SearchSpace
from Comp_499A.Tetris.n_decomposition import NDDecomposer
from Comp_499A.Tetris.gap_prefix import convert_gap_boxes
from Comp_499A.Tetris.tetris import PrefixBox, tetris
from Comp_499A.main.utils import *
from Comp_499A.Tetris.cds import MultilevelCDS
rng = np.random.default_rng()
# Default query
DEFAULT_STRUCTURE_MATRIX = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1]
], dtype=int)

# RUN DIRECTORY RESOLUTION
def resolve_run_directory(base_dir: str, run_id: int) -> tuple[int, str]:
    """
    If join_instances/run_<run_id> exists and is non-empty, increment run_id until
    a missing or empty directory is found. Creates the directory if missing.
    Returns (final_run_id, run_path).
    """
    os.makedirs(base_dir, exist_ok=True)

    while True:
        run_path = os.path.join(base_dir, f"run_{run_id}")
        if not os.path.exists(run_path):
            os.makedirs(run_path, exist_ok=False)
            return run_id, run_path

        if len(os.listdir(run_path)) == 0:
            return run_id, run_path

        print(f"[INFO] run_{run_id} exists and is non-empty. Incrementing RUN.")
        run_id += 1


def write_run_meta_txt(
    run_dir: str,
    run_id: int,
    structure_matrix: Sequence[Sequence[int]],
    attributes: List[str],
    relations: List[List[str]],
    distribution_pool: dict,
    run_distribution_choice: Dict[str, Any],
    domain_choices: Sequence[int],
    run_domain_lengths: Dict[str, int],
    run_dyadic_dims: List[int],
    samples_range: Tuple[int, int],
    seed0: Optional[int],
    n_instances: int,
    generator_version: str = "tetris_instance_gen_v2",
) -> str:
    """
    Writes: join_instances/run_<RUN>/run_<RUN>_meta.txt (atomic).
    """
    meta_name = f"run_{run_id}_meta.txt"
    path = os.path.join(run_dir, meta_name)

    def j(x) -> str:
        return json.dumps(x, separators=(",", ":"))

    created_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    content = "\n".join([
        "# run metadata",
        f"run_id: {run_id}",
        f'created_utc: {j(created_utc)}',
        f'generator_version: {j(generator_version)}',
        "",
        "# query",
        f"structure_matrix: {j(structure_matrix)}",
        f"attributes: {j(attributes)}",
        f"relations: {j(relations)}",
        "",
        "# run distribution pool (instance-level choice drawn from this)",
        f"distribution_pool: {j(distribution_pool)}",
        "",
        "# generation knobs",
        f"domain_choices: {j(list(domain_choices))}",
        f"samples_range: {j(list(samples_range))}",
        f"seed0: {j(seed0)}",
        f"n_instances: {n_instances}",
        "# resolved run configuration",
        f"run_distribution_choice: {j(run_distribution_choice)}",
        f"run_domain_lengths: {j(run_domain_lengths)}",
        f"run_dyadic_dims: {j(run_dyadic_dims)}",
        "",
    ])

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)
    return path


def random_center_by_attr(
    attributes: List[str],
    domain_lengths: Dict[str, int],
    rng: np.random.Generator,
) -> Dict[str, float]:
    return {a: float(rng.integers(0, int(domain_lengths[a]))) for a in attributes}


def project_center_to_attrs(
    center_by_attr: Dict[str, float],
    attrs: List[str],
) -> List[float]:
    return [float(center_by_attr[a]) for a in attrs]


def apply_components_to_space(
    space: "SearchSpace",
    components: List[Dict[str, Any]],
    attrs: List[str],
) -> None:
    """
    components format:
      {
        "kind": "attractor" | "repeller",
        "center": {"v1": ..., "v2": ..., ...},   # GLOBAL, keyed by attr name
        "weight": float,
        "spread": float
      }
    """
    for comp in components:
        kind = comp["kind"]
        center = project_center_to_attrs(comp["center"], attrs)
        weight = float(comp["weight"])
        spread = float(comp["spread"])

        if kind == "attractor":
            space.add_attractor(center=center, weight=weight, spread=spread)
        elif kind == "repeller":
            space.add_repeller(center=center, strength=weight, spread=spread)
        else:
            raise ValueError(f"Unknown component kind: {kind}")


def make_random_components(
    attributes: List[str],
    domain_lengths: Dict[str, int],
    rng: np.random.Generator,
    n_attractors: int,
    n_repellers: int,
    spread_range: Tuple[float, float],
    weight_range: Tuple[float, float],
) -> List[Dict[str, Any]]:
    comps: List[Dict[str, Any]] = []

    for _ in range(n_attractors):
        comps.append({
            "kind": "attractor",
            "center": random_center_by_attr(attributes, domain_lengths, rng),
            "weight": float(rng.uniform(weight_range[0], weight_range[1])),
            "spread": float(rng.uniform(spread_range[0], spread_range[1])),
        })

    for _ in range(n_repellers):
        comps.append({
            "kind": "repeller",
            "center": random_center_by_attr(attributes, domain_lengths, rng),
            "weight": float(rng.uniform(weight_range[0], weight_range[1])),
            "spread": float(rng.uniform(spread_range[0], spread_range[1])),
        })

    return comps


def resolve_run_configuration(
    attributes: List[str],
    domain_choices: Sequence[int],
    distribution_pool: dict,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Resolve everything that should stay fixed within a run:
      - domain_lengths
      - dyadic_dims
      - distribution_choice
    """
    run_domain_lengths = {a: int(rng.choice(domain_choices)) for a in attributes}
    run_dist_choice = pick_distribution(distribution_pool["choices"], rng, force_type="random")

    dims = [int(run_domain_lengths[a]) for a in attributes]
    typ = run_dist_choice.get("type")
    params = run_dist_choice.get("params", {}) or {}

    resolved_distribution: Dict[str, Any] = {
        "type": typ,
        "params": dict(params),
    }

    if typ == "normal":
        spread_frac = float(params.get("spread_frac", 0.20))
        depth = float(params.get("depth", 5.0))

        resolved_distribution["resolved_center"] = {
            a: float((int(run_domain_lengths[a])) / 2.0) for a in attributes
        }
        resolved_distribution["resolved_spread"] = float(_sigma_from_frac(dims, spread_frac))
        resolved_distribution["resolved_depth"] = depth

    elif typ == "uniform":
        pass

    elif typ == "custom":
        components = params.get("components", [])
        if not components:
            raise ValueError("custom distribution requires params['components']")
        # Expect centers keyed by global attribute name
        resolved_distribution["components"] = list(components)

    elif typ == "random":
        n_attractors = int(params.get("n_attractors", 2))
        n_repellers = int(params.get("n_repellers", 1))
        spread_range = tuple(params.get("spread_range", [2.0, 8.0]))
        weight_range = tuple(params.get("weight_range", [2.0, 8.0]))

        resolved_distribution["components"] = make_random_components(
            attributes=attributes,
            domain_lengths=run_domain_lengths,
            rng=rng,
            n_attractors=n_attractors,
            n_repellers=n_repellers,
            spread_range=spread_range,
            weight_range=weight_range,
        )

    else:
        raise ValueError(f"Unknown distribution type: {typ}")

    return {
        "domain_lengths": run_domain_lengths,
        "dyadic_dims": [next_pow2(int(run_domain_lengths[a])) for a in attributes],
        "distribution_choice": resolved_distribution,
    }

# INSTANCE FILE
def write_instance_txt(
    run_dir: str,
    run_id: int,
    instance_id: int,
    instance_seed: int,
    run_meta_file: str,
    structure_matrix: Sequence[Sequence[int]],
    attributes: List[str],
    relations: List[List[str]],
    domain_lengths: Dict[str, int],
    samples_per_relation: Dict[int, int],
    distribution_choice: Dict[str, Any],
    temperature: float,
    dyadic_dims: List[int],
    global_gaps: List[List[Tuple[int, int]]],
    unit_dyadic_dims: List[int],
    unit_global_gaps: List[List[Tuple[int, int]]],
    append_manifest: bool = True,
) -> str:
    """
    Writes: join_instances/run_<RUN>/run_<RUN>_instance_<J>.txt (atomic)
    plus appends to manifest.jsonl.
    """
    fname = f"run_{run_id}_instance_{instance_id}.txt"
    path = os.path.join(run_dir, fname)

    def j(x) -> str:
        return json.dumps(x, separators=(",", ":"))

    spr = {str(k): int(v) for k, v in samples_per_relation.items()}

    content = "\n".join([
        "# instance",
        f"run_id: {run_id}",
        f"instance_id: {instance_id}",
        f"instance_seed: {instance_seed}",
        f'run_meta_file: {j(run_meta_file)}',
        "",
        "# query",
        f"structure_matrix: {j(structure_matrix)}",
        f"attributes: {j(attributes)}",
        f"relations: {j(relations)}",
        "",
        "# domains",
        f"domain_lengths: {j(domain_lengths)}",
        "",
        "# sampling",
        f"samples_per_relation: {j(spr)}",
        f"distribution: {j(distribution_choice)}",
        f"temperature: {float(temperature)}",
        "",
        "# outputs",
        f"dyadic_dims: {j(dyadic_dims)}",
        f"global_gaps: {j(global_gaps)}",
        "",
        "# unit-partition outputs",
        f"unit_dyadic_dims: {j(unit_dyadic_dims)}",
        f"unit_global_gaps: {j(unit_global_gaps)}",
        "",
    ])

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

    if append_manifest:
        manifest_path = os.path.join(run_dir, "manifest.jsonl")
        rec = {
            "run_id": run_id,
            "instance_id": instance_id,
            "instance_seed": instance_seed,
            "distribution_type": distribution_choice.get("type"),
            "file": fname,
            "run_meta": run_meta_file,
        }
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec) + "\n")

    return path


# QUERY STRUCTURE HELPERS
def get_relation_attributes(matrix: np.ndarray, attributes: List[str]) -> List[List[str]]:
    relations = []
    for row in matrix:
        attrs = [attr for attr, bit in zip(attributes, row.tolist()) if int(bit) == 1]
        relations.append(attrs)
    return relations


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def canon_box(box):
    return tuple(tuple(iv) for iv in box)

def bits_from_domain_size(domain_size: int) -> int:
    if domain_size <= 0 or (domain_size & (domain_size - 1)) != 0:
        raise ValueError(f"domain_size must be a positive power of two, got {domain_size}")
    return domain_size.bit_length() - 1  # since domain_size = 2^bits

def interval_to_prefix(lo: int, hi: int, domain_size: int) -> str:
    """
    Convert one dyadic interval [lo, hi) inside [0, domain_size) into its binary prefix.
    Assumes domain_size is a power of 2 and interval is dyadic.
    Examples in domain 16:
      [0,16) -> ""
      [12,14) -> "110"
      [3,4) -> "0011"
      [8,12) -> "10"
    """
    if hi <= lo:
        raise ValueError(f"Invalid interval [{lo},{hi})")
    if lo < 0 or hi > domain_size:
        raise ValueError(f"Interval [{lo},{hi}) out of range for domain {domain_size}")

    length = hi - lo
    if (length & (length - 1)) != 0:
        raise ValueError(f"Interval length must be power of 2, got {length} for [{lo},{hi})")

    if lo % length != 0:
        raise ValueError(f"Interval [{lo},{hi}) is not dyadic-aligned")

    bits = bits_from_domain_size(domain_size)
    free_bits = bits_from_domain_size(length)
    prefix_len = bits - free_bits

    if prefix_len == 0:
        return ""

    return format(lo, f"0{bits}b")[:prefix_len]


def prefix_to_unit_box(prefix: str, total_bits: int) -> List[Tuple[int, int]]:
    """
    Convert a binary prefix to a unit-partition box over total_bits binary dimensions.
    '0' -> (0,1)
    '1' -> (1,2)
    lambda/free suffix -> (0,2)
    """
    if len(prefix) > total_bits:
        raise ValueError(f"prefix '{prefix}' longer than total_bits={total_bits}")

    out: List[Tuple[int, int]] = []
    for ch in prefix:
        if ch == "0":
            out.append((0, 1))
        elif ch == "1":
            out.append((1, 2))
        else:
            raise ValueError(f"Invalid prefix character: {ch!r}")

    out.extend((0, 2) for _ in range(total_bits - len(prefix)))
    return out


def grouped_gap_to_unit_gap(
    gap_box: Sequence[Sequence[int]],
    dyadic_dims: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Convert one grouped dyadic gap box, e.g.
      [[0,64],[12,14],[3,4]]
    under dyadic_dims=[64,16,16]
    into its unit-partition representation over [2,2,...,2].

    Example output shape length = sum(log2(dyadic_dims)).
    """
    if len(gap_box) != len(dyadic_dims):
        raise ValueError("gap_box and dyadic_dims must have same length")

    unit_box: List[Tuple[int, int]] = []
    for (lo, hi), dom in zip(gap_box, dyadic_dims):
        dom = int(dom)
        b = bits_from_domain_size(dom)
        pfx = interval_to_prefix(int(lo), int(hi), dom)
        unit_box.extend(prefix_to_unit_box(pfx, b))
    return unit_box


def transform_global_gaps_to_unit_partition(
    global_gaps: Sequence[Sequence[Sequence[int]]],
    dyadic_dims: Sequence[int],
) -> Tuple[List[int], List[List[Tuple[int, int]]]]:
    """
    Convert grouped gap boxes into unit-partition gap boxes.

    Returns:
      unit_dyadic_dims = [2] * total_bits
      unit_global_gaps = list of length-total_bits boxes with intervals in {(0,1),(1,2),(0,2)}
    """
    total_bits = sum(bits_from_domain_size(int(d)) for d in dyadic_dims)
    unit_dyadic_dims = [2] * total_bits

    unit_global_gaps = [grouped_gap_to_unit_gap(b, dyadic_dims) for b in global_gaps]

    # Dedup for safety
    unit_gap_set = set(canon_box(b) for b in unit_global_gaps)
    unit_global_gaps = [list(map(tuple, b)) for b in unit_gap_set]

    return unit_dyadic_dims, unit_global_gaps


def pick_distribution(choices, rng, force_type=None):
    if force_type is not None:
        for c in choices:
            if c.get("type") == force_type:
                return dict(c)
        raise ValueError(f"type {force_type} not found")

    return dict(choices[int(rng.integers(len(choices)))])


# DIM-AWARE DISTRIBUTION BUILDERS
def _midpoint(dims: Sequence[int]) -> Tuple[float, ...]:
    return tuple(int(d) / 2.0 for d in dims)

def _sigma_from_frac(dims: Sequence[int], spread_frac: float) -> float:
    scale = float(np.mean(dims))
    return max(1.0, float(spread_frac) * scale)


def make_space(
    attrs: List[str],
    domain_lengths: Dict[str, int],
    dist_choice: Dict[str, Any],
):
    dims = [int(domain_lengths[a]) for a in attrs]
    typ = dist_choice.get("type")

    if typ == "uniform":
        return None

    space = SearchSpace(dims=dims)

    if typ == "normal":
        center_by_attr = dist_choice.get(
            "resolved_center",
            {a: float((int(domain_lengths[a]) - 1) / 2.0) for a in domain_lengths}
        )
        center = [float(center_by_attr[a]) for a in attrs]
        spread = float(dist_choice.get("resolved_spread", _sigma_from_frac(dims, 0.20)))
        depth = float(dist_choice.get("resolved_depth", 5.0))
        space.add_attractor(center=center, weight=depth, spread=spread)

    elif typ == "custom":
        components = dist_choice.get("components", [])
        apply_components_to_space(space, components, attrs)

    elif typ == "random":
        components = dist_choice.get("components", [])
        apply_components_to_space(space, components, attrs)

    else:
        raise ValueError(f"Unknown distribution type: {typ}")

    return space


# GAP BOX PIPELINE
def compute_local_gaps(df, relation_attrs, domain_lengths):
    domain_box = [(0, int(domain_lengths[attr])) for attr in relation_attrs]
    points = [tuple(int(row[attr]) for attr in relation_attrs) for _, row in df.iterrows()]
    root = NDDecomposer(domain_box, points)
    gaps = root.build()
    return gaps

def lift_gap_box(local_gap, relation_attrs, global_attrs, domain_lengths):
    local_map = dict(zip(relation_attrs, local_gap))
    lifted = []
    for g in global_attrs:
        if g in local_map:
            lifted.append(local_map[g])
        else:
            lifted.append((0, int(domain_lengths[g])))
    return lifted



# ONE INSTANCE GENERATION
def tetris_prep_one_instance(
    structure_matrix: np.ndarray,
    domain_lengths: Dict[str, int],
    samples_per_relation: Dict[int, int],
    dist_choice: Dict[str, Any],
    temperature: float,
    rng: np.random.Generator,
):
    num_attrs = int(structure_matrix.shape[1])
    attributes = [f"v{i+1}" for i in range(num_attrs)]
    relations = get_relation_attributes(structure_matrix, attributes)

    relation_dfs = []
    global_gaps = []

    for i, attrs in enumerate(relations):
        num_samples = int(samples_per_relation[i])
        dims = [int(domain_lengths[a]) for a in attrs]

        if dist_choice.get("type") == "uniform":
            samples = [
                tuple(int(rng.integers(0, d)) for d in dims)
                for _ in range(num_samples)
            ]
        else:
            space = make_space(attrs, domain_lengths=domain_lengths, dist_choice=dist_choice)
            samples = space.sample_gibbs(
                num_samples,
                temperature=float(temperature),
                rng=rng,
            )

        df = pd.DataFrame(list(samples), columns=attrs)
        relation_dfs.append(df)

    for relation_attrs, df in zip(relations, relation_dfs):
        local_gaps = compute_local_gaps(df, relation_attrs, domain_lengths)
        for g in local_gaps:
            lifted = lift_gap_box(g, relation_attrs, attributes, domain_lengths)
            global_gaps.append(lifted)

    gap_set = set(canon_box(b) for b in global_gaps)
    global_gaps = [list(map(tuple, b)) for b in gap_set]

    dyadic_dims = [next_pow2(int(domain_lengths[a])) for a in attributes]

    global_space = SearchSpace(dims=dyadic_dims)
    global_space.add_hard_boxes(global_gaps)

    return attributes, relations, global_gaps, dyadic_dims

distribution_pools = {
    "temperature": 1.0,
    "choices": [
        {
            "type": "normal",
            "params": {"spread_frac": 0.20, "depth": 5.0}
        },
        {
            "type": "uniform"
        },
        {
            "type": "custom",
            "params": {
                "components": [
            {
                "kind": "attractor",
                "center": {"v1": 16.0, "v2": 16.0, "v3": 16.0},
                "weight": 5.0,
                "spread": 1.0
            },
            {
                "kind": "repeller",
                "center": {"v1": 48.0, "v2": 32.0, "v3": 20.0},
                "weight": 3.0,
                "spread": 8.0
            },
        ]
            }
        },
        {
            "type": "random",
            "params": {
                "n_attractors": int(rng.integers(1, 4)),
                "n_repellers": int(rng.integers(1, 3)),
                "spread_range": [2.0, 10.0],
                "weight_range": [2.0, 7.0]
            }
        },
    ],
}


# TOP-LEVEL: GENERATE RUN + N INSTANCES AND WRITE .txt FILES
def generate_instances_to_txt(
    n: int,
    run_id: int,
    out_root: str = "join_instances",
    structure_matrix: Optional[np.ndarray] = None,
    distribution_pool: Optional[dict] = None,
    domain_choices: Sequence[int] = (64,64),
    samples_range: Tuple[int, int] = (5, 20),
    seed0: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Creates:
      join_instances/run_<RUN>/run_<RUN>_meta.txt
      join_instances/run_<RUN>/run_<RUN>_instance_<j>.txt
      join_instances/run_<RUN>/manifest.jsonl

    Returns summary dict with resolved run_id, run_dir, and list of written files.
    """
    if structure_matrix is None:
        structure_matrix = DEFAULT_STRUCTURE_MATRIX.copy()

    if distribution_pool is None:
        distribution_pool = distribution_pools


    # resolve run dir
    run_id, run_dir = resolve_run_directory(out_root, run_id)

    num_attrs = int(structure_matrix.shape[1])
    attributes = [f"v{i+1}" for i in range(num_attrs)]
    relations = get_relation_attributes(structure_matrix, attributes)

    temperature = float(distribution_pool.get("temperature", 1.0))
    rng_run = np.random.default_rng(seed0)

    run_cfg = resolve_run_configuration(
        attributes=attributes,
        domain_choices=domain_choices,
        distribution_pool=distribution_pool,
        rng=rng_run,
    )

    run_domain_lengths = run_cfg["domain_lengths"]
    run_dist_choice = run_cfg["distribution_choice"]
    run_dyadic_dims = run_cfg["dyadic_dims"]

    # write run meta AFTER run_cfg is known
    run_meta_path = write_run_meta_txt(
        run_dir=run_dir,
        run_id=run_id,
        structure_matrix=structure_matrix.tolist(),
        attributes=attributes,
        relations=relations,
        distribution_pool=distribution_pool,
        run_distribution_choice=run_dist_choice,
        domain_choices=domain_choices,
        run_domain_lengths=run_domain_lengths,
        run_dyadic_dims=run_dyadic_dims,
        samples_range=samples_range,
        seed0=seed0,
        n_instances=n,
    )
    run_meta_file = os.path.basename(run_meta_path)

    written = []
    lo, hi = int(samples_range[0]), int(samples_range[1])

    for j in range(1, int(n) + 1):
        instance_seed = int(rng_run.integers(0, 2**31 - 1))
        rng_inst = np.random.default_rng(instance_seed)

        # fixed per run
        dist_choice = run_dist_choice
        domain_lengths = run_domain_lengths

        # randomized per instance
        samples_per_relation = {
            i: int(rng_inst.integers(lo, hi + 1))
            for i in range(len(relations))
        }

        attrs, rels, global_gaps, dyadic_dims = tetris_prep_one_instance(
            structure_matrix=structure_matrix,
            domain_lengths=domain_lengths,
            samples_per_relation=samples_per_relation,
            dist_choice=dist_choice,
            temperature=temperature,
            rng=rng_inst,
        )

        # optional sanity check
        if list(dyadic_dims) != list(run_dyadic_dims):
            raise ValueError(
                f"dyadic_dims changed within run: expected {run_dyadic_dims}, got {dyadic_dims}"
            )

        unit_dyadic_dims, unit_global_gaps = transform_global_gaps_to_unit_partition(
            global_gaps=global_gaps,
            dyadic_dims=dyadic_dims,
        )

        path = write_instance_txt(
            run_dir=run_dir,
            run_id=run_id,
            instance_id=j,
            instance_seed=instance_seed,
            run_meta_file=run_meta_file,
            structure_matrix=structure_matrix.tolist(),
            attributes=attrs,
            relations=rels,
            domain_lengths=domain_lengths,
            samples_per_relation=samples_per_relation,
            distribution_choice=dist_choice,
            temperature=temperature,
            dyadic_dims=dyadic_dims,
            global_gaps=global_gaps,
            unit_dyadic_dims=unit_dyadic_dims,
            unit_global_gaps=unit_global_gaps,
            append_manifest=True,
        )
        written.append(path)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_meta": run_meta_path,
        "instance_files": written,
    }

# LOADERS (run meta + instance)
def load_kv_txt(path: str) -> Dict[str, Any]:
    """
    Reads key: value lines where value is JSON for known keys; else parses ints/floats.
    Works for both run_meta and instance files created above.
    """
    json_keys = {
        "created_utc", "generator_version",
        "structure_matrix", "attributes", "relations",
        "distribution_pool", "domain_choices", "samples_range", "seed0",
        "run_meta_file",
        "domain_lengths", "samples_per_relation",
        "distribution", "dyadic_dims", "global_gaps",
        "unit_dyadic_dims", "unit_global_gaps",
        "run_distribution_choice", "run_domain_lengths", "run_dyadic_dims",
    }
    out: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k in json_keys:
                out[k] = json.loads(v)
            else:
                # numeric fallbacks
                if k in {"run_id", "instance_id", "instance_seed", "n_instances"}:
                    out[k] = int(v)
                elif k == "temperature":
                    out[k] = float(v)
                else:
                    out[k] = v
    return out


def load_instance_txt(path: str) -> dict:
    json_keys = {
        "structure_matrix","attributes","relations",
        "domain_lengths","samples_per_relation",
        "distribution","dyadic_dims","global_gaps","run_meta_file",
        "unit_dyadic_dims","unit_global_gaps",
    }
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in json_keys:
                out[k] = json.loads(v)
            else:
                if k in {"run_id","instance_id","instance_seed"}:
                    out[k] = int(v)
                elif k == "temperature":
                    out[k] = float(v)
                else:
                    out[k] = v
    return out

def write_dimacs_cnf(path: str, clauses: list[list[int]]) -> None:
    maxvar = max((abs(l) for c in clauses for l in c), default=0)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"p cnf {maxvar} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
    os.replace(tmp, path)

def convert_run_txt_to_cnf(run_dir: str, ordering=None):
    """
    For every run_<RUN>_instance_<J>.txt in run_dir, produce run_<RUN>_instance_<J>.cnf
    """
    for name in sorted(os.listdir(run_dir)):
        if not (name.endswith(".txt") and "_instance_" in name):
            continue

        txt_path = os.path.join(run_dir, name)
        rec = load_instance_txt(txt_path)

        global_gaps = rec["global_gaps"]
        dyadic_dims = rec["dyadic_dims"]

        bits_per_dim = [bits_from_domain_size(s) for s in dyadic_dims]
        prefix_tuples = convert_gap_boxes(global_gaps, dyadic_dims)
        B = [PrefixBox(tuple(t)) for t in prefix_tuples]
        clauses = gapboxes_to_clauses(B, bits_per_dim, ordering=ordering)

        cnf_path = os.path.splitext(txt_path)[0] + ".cnf"
        write_dimacs_cnf(cnf_path, clauses)


def run_tetris_on_gaps(
    global_gaps: Sequence[Sequence[Tuple[int, int]]],
    dyadic_dims: Sequence[int],
    *,
    seed_A: bool = True,
    sao=None,
    ordering=None,
    trace_dir: str = "tetris_traces",
    run_id: Optional[int] = None,
    instance_id: Optional[int] = None,
    suffix: Optional[str] = None,
    trace_enabled: bool = True,
    trace_flush_every: int = 1000,
    halt_first: bool = True,
):
    def _is_pow2(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0

    if not all(_is_pow2(int(s)) for s in dyadic_dims):
        raise ValueError(f"dyadic_dims must be powers of 2, got {dyadic_dims}")

    D = len(dyadic_dims)
    for b in global_gaps:
        if len(b) != D:
            raise ValueError(f"gapbox dim {len(b)} != domain dim {D}: {b}")
        for d, (lo, hi) in enumerate(b):
            if not (0 <= lo < hi <= dyadic_dims[d]):
                raise ValueError(f"gap interval {(lo,hi)} out of range [0,{dyadic_dims[d]}] at dim {d}")

    bits_per_dim = [bits_from_domain_size(int(s)) for s in dyadic_dims]
    prefix_tuples = convert_gap_boxes(global_gaps, dyadic_dims)
    B = [PrefixBox(tuple(t)) for t in prefix_tuples]
    if not B:
        return [], None

    cds = MultilevelCDS(num_dims=len(B[0].coords))
    for b in B:
        cds.insert_box_prefixes(list(b.coords))

    def oracle(w: PrefixBox):
        tuples = cds.boxes_containing_prefix_box(list(w.coords))
        return [PrefixBox(t) for t in set(tuples)]

    init_A = None
    if seed_A:
        init_A = [b for b in B if "" in b.coords]
    init_A = B

    os.makedirs(trace_dir, exist_ok=True)
    rid = "NA" if run_id is None else str(run_id)
    iid = "NA" if instance_id is None else str(instance_id)
    suf = "" if not suffix else f"_{suffix}"
    trace_path = os.path.join(trace_dir, f"tetris_run_{rid}_inst_{iid}{suf}.jsonl")

    outputs = tetris(
        B=B,
        sao=sao,
        widths=bits_per_dim,
        init_A=init_A,
        oracle=oracle,
        trace_path=trace_path,
        trace_enabled=trace_enabled,
        trace_flush_every=trace_flush_every,
        halt_first=halt_first,
    )
    return outputs, trace_path


def run_tetris_traces_for_run_dir(
    run_dir: str,
    *,
    trace_root: str = "join_tetris_traces",
    halt_first: bool = True,
    trace_flush_every: int = 1000,
):
    """
    For each run_<RUN>_instance_<J>.txt in run_dir:
      - load global_gaps + dyadic_dims
      - run tetris
      - write trace to: tetris_traces/run_<RUN>/tetris_run_<RUN>_inst_<J>.jsonl
    """
    # infer run_id from folder name if possible
    base = os.path.basename(os.path.normpath(run_dir))
    run_id = int(base.split("_")[1]) if base.startswith("run_") else None

    out_dir = os.path.join(trace_root, f"run_{run_id}" if run_id is not None else "run_NA")
    os.makedirs(out_dir, exist_ok=True)

    produced = []
    for name in sorted(os.listdir(run_dir)):
        if not (name.endswith(".txt") and "_instance_" in name):
            continue

        rec = load_instance_txt(os.path.join(run_dir, name))
        iid = rec.get("instance_id", None)

        global_gaps = rec["global_gaps"]
        dyadic_dims = rec["dyadic_dims"]

        _, trace_path = run_tetris_on_gaps(
            global_gaps=global_gaps,
            dyadic_dims=dyadic_dims,
            trace_dir=out_dir,
            run_id=run_id,
            instance_id=iid,
            halt_first=halt_first,
            trace_flush_every=trace_flush_every,
        )
        produced.append(trace_path)

    return produced

def run_tetris_unit_traces_for_run_dir(
    run_dir: str,
    *,
    trace_root: str = "join_tetris_unit",
    halt_first: bool = True,
    trace_flush_every: int = 1000,
):
    """
    For each run_<RUN>_instance_<J>.txt in run_dir:
      - load unit_global_gaps + unit_dyadic_dims
      - run tetris on the unit-partition representation
      - write trace to: tetris_unit/run_<RUN>/tetris_run_<RUN>_inst_<J>_unit.jsonl
    """
    base = os.path.basename(os.path.normpath(run_dir))
    run_id = int(base.split("_")[1]) if base.startswith("run_") else None

    out_dir = os.path.join(trace_root, f"run_{run_id}" if run_id is not None else "run_NA")
    os.makedirs(out_dir, exist_ok=True)

    produced = []
    for name in sorted(os.listdir(run_dir)):
        if not (name.endswith(".txt") and "_instance_" in name):
            continue

        rec = load_instance_txt(os.path.join(run_dir, name))
        iid = rec.get("instance_id", None)

        unit_global_gaps = rec["unit_global_gaps"]
        unit_dyadic_dims = rec["unit_dyadic_dims"]

        _, trace_path = run_tetris_on_gaps(
            global_gaps=unit_global_gaps,
            dyadic_dims=unit_dyadic_dims,
            trace_dir=out_dir,
            run_id=run_id,
            instance_id=iid,
            suffix="unit",
            halt_first=halt_first,
            trace_flush_every=trace_flush_every,
        )
        produced.append(trace_path)

    return produced


CADICAL_BIN = "cadical"         

# CADICAL configuration. 
CADICAL_FLAGS = [
    "--plain",
    "--no-binary",
    "--lrat=true",
    "--lucky=false", "--luckyearly=false", "--luckylate=false", "--luckyassumptions=false",
    "--walk=false", "--warmup=false",
    "--preprocesslight=false",
    "--inprocessing=false", "--inprobing=false", "--probe=false",
    "--elim=false", "--subsume=false", "--deduplicate=false", "--sweep=false", "--vivify=false",
    "--ternary=false", "--transred=false", "--congruence=false", "--decompose=false",
    "--reduce=false",
    "--restart=false", "--reluctant=false", "--rephase=0", "--stabilize=false", "--target=0",
    "--chrono=0", "--chronoreusetrail=false",
    "--otfs=false", "--minimize=false", "--shrink=0",
    "--randec=false", "--seed=0",
    "--shuffle=false", "--shufflequeue=false", "--shufflescores=false",
    "--compact=false",
]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _write_text(path: str, s: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    os.replace(tmp, path)

def _write_json(path: str, obj: Any) -> None:
    _write_text(path, json.dumps(obj, indent=2, sort_keys=True))

def save_cadical_config(run_dir: str, run_id: int) -> Dict[str, str]:
    cfg = {
        "cadical_bin": CADICAL_BIN,
        "flags": CADICAL_FLAGS,
        "note": (
            "stdout+stderr captured to action_run_<RUN>_instance_<J>.log; "
            "lrat proof per instance; SAT trace extracted from stdout into trace_run_<RUN>_instance_<J>."
        ),
    }
    json_path = os.path.join(run_dir, f"cadical_run_{run_id}_config.json")
    txt_path  = os.path.join(run_dir, f"cadical_run_{run_id}_config.txt")
    _write_json(json_path, cfg)
    _write_text(txt_path, " ".join([CADICAL_BIN] + CADICAL_FLAGS) + " <cnf> <proof.lrat>\n")
    return {"json": json_path, "txt": txt_path}

def extract_sat_trace_from_stdout(text: str) -> str:
    """
    Extract only the CaDiCaL trace event lines from stdout.
    (These are the lines you want to treat as the SAT trace.)
    """
    prefixes = (
        "MARK_FIXED",
        "UNMARK_FIXED",
        "ASSIGN ",
        "DECIDE ",
        "CONFLICT ",
        "CONFLICT_AFTER_PROP",
        "AFTER_ANALYZE",
        "AFTER_ANALYZE",
        "ANALYZE_BEGIN",
        "UIP_FOUND",
        "LEARNED_METRICS",
        "LEARNED_FINAL",
        "LEARNED_LIT ",
        "DRIVING_ID",
        "BACKTRACK",
        "ASSERT_ASSIGN",
        "AFTER_ANALYZE",
        "RESTART",
        "REDUCE",
    )
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line and line.startswith(prefixes):
            out.append(line)
    return "\n".join(out) + ("\n" if out else "")

def run_cadical_on_one_cnf(
    cnf_path: str,
    *,
    run_id: int,
    instance_id: int,
    out_dir: str,
    cwd: Optional[str] = None,
) -> Dict[str, str]:
    _ensure_dir(out_dir)

    proof_path = os.path.join(out_dir, f"proof_run_{run_id}_instance_{instance_id}.lrat")
    log_path   = os.path.join(out_dir, f"action_run_{run_id}_instance_{instance_id}.log")
    trace_path = os.path.join(out_dir, f"trace_run_{run_id}_instance_{instance_id}")  # no extension

    argv = [CADICAL_BIN] + CADICAL_FLAGS + [cnf_path, proof_path]

    proc = subprocess.run(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Full stdout+stderr (banner + stats + trace events)
    _write_text(log_path, proc.stdout)

    # Extract and save SAT trace events as a separate file (this is the key fix)
    trace_text = extract_sat_trace_from_stdout(proc.stdout)
    if trace_text:
        _write_text(trace_path, trace_text)
    else:
        # Keep a marker in the action log for debugging.
        _write_text(
            log_path,
            proc.stdout + "\n[WARN] no SAT trace events found in stdout (no ASSIGN/CONFLICT/etc).\n"
        )

    return {
        "cnf": cnf_path,
        "proof": proof_path,
        "log": log_path,
        "trace": trace_path,
        "returncode": str(proc.returncode),
    }

def run_cadical_for_run_dir(
    run_dir: str,
    *,
    cnf_dir: Optional[str] = None,      # default: run_dir
    outputs_subdir: str = "join_sat_outputs",# folder at project root
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    base = os.path.basename(os.path.normpath(run_dir))
    if not base.startswith("run_"):
        raise ValueError(f"run_dir must look like .../run_<RUN>, got: {run_dir}")
    run_id = int(base.split("_")[1])

    cnf_dir = cnf_dir or run_dir

    # Put outputs in <project_root>/sat_outputs/run_<RUN>/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    out_dir = os.path.join(project_root, outputs_subdir, f"run_{run_id}")
    _ensure_dir(out_dir)

    cfg_paths = save_cadical_config(run_dir, run_id)

    results = []
    for name in sorted(os.listdir(cnf_dir)):
        if not (name.endswith(".cnf") and "_instance_" in name):
            continue

        inst_str = name.split("_instance_")[1].split(".cnf")[0]
        instance_id = int(inst_str)

        cnf_path = os.path.join(cnf_dir, name)
        results.append(
            run_cadical_on_one_cnf(
                cnf_path,
                run_id=run_id,
                instance_id=instance_id,
                out_dir=out_dir,
                cwd=cwd,
            )
        )

    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "config": cfg_paths,
        "outputs_dir": out_dir,
        "jobs": results,
    }


def main():
    structure_matrix = np.array([
        [1, 1, 0],
        [0, 1, 1],
        [1, 0, 1],
        ])

       
    def generate_instances_and_cnf(*, n: int, run_id: int, seed0: int = 0, **gen_kwargs):
        result = generate_instances_to_txt(n=n, run_id=run_id, seed0=seed0, structure_matrix=structure_matrix, **gen_kwargs)
        convert_run_txt_to_cnf(result["run_dir"])
        run_tetris_traces_for_run_dir(result["run_dir"])
        run_tetris_unit_traces_for_run_dir(result["run_dir"])
        run_cadical_for_run_dir(result["run_dir"])    # runs CaDiCaL on all .cnf in that run

    generate_instances_and_cnf(n=50, run_id=1, seed0=42)

if __name__ == "__main__":
    main()
# ============================================================
