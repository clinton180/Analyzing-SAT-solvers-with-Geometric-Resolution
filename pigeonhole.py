# pigeonhole_generate.py
from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from join_queries import resolve_run_directory
from join_queries import (
    run_cadical_on_one_cnf,
    save_cadical_config,
    _ensure_dir,
    _write_json,
    run_tetris_on_gaps,
)
from Comp_499A.SAT.sat_gap import cnf_to_gapboxes_raw
from cnfgen.families import pigeonhole
from pysat.formula import CNF as PyCNF


PH_ROOT = "pigeonhole_cnf"
SAT_ROOT = Path("pigeonhole_sat").resolve()
TETRIS_ROOT = Path("pigeonhole_tetris").resolve()


def write_text_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2, sort_keys=True))


def clauses_to_dimacs(clauses: List[List[int]], nv: int) -> str:
    lines: List[str] = [f"p cnf {nv} {len(clauses)}"]
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")
    lines.append("")
    return "\n".join(lines)


def cnf_prep(n: int, m: int) -> Tuple[List[List[int]], int]:
    """
    Pigeonhole instance:
      - SAT when n <= m
      - UNSAT when n > m
    """
    cnf_obj = pigeonhole.PigeonholePrinciple(n, m)
    formula = PyCNF(from_string=cnf_obj.to_dimacs())
    return formula.clauses, formula.nv


def instance_status(n: int, m: int) -> str:
    return "sat" if n <= m else "unsat"


def all_candidate_specs(max_x: int, mode: str) -> List[Dict[str, Any]]:
    """
    Generate all unique (n, m) with 1 <= n,m < max_x, filtered by mode.
    """
    if max_x <= 1:
        raise ValueError("max_x must be > 1 because we require 1 <= n,m < max_x")

    specs: List[Dict[str, Any]] = []
    for n in range(1, max_x):
        for m in range(1, max_x):
            status = instance_status(n, m)

            if mode == "sat" and status != "sat":
                continue
            if mode == "unsat" and status != "unsat":
                continue
            if mode == "mixed" and status not in {"sat", "unsat"}:
                continue

            specs.append({
                "n": n,
                "m": m,
                "status": status,
            })

    return specs


def choose_unique_specs(
    *,
    num_instances: int,
    max_x: int,
    mode: str,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Choose unique specs for one run.
    For mixed mode, try to include both SAT and UNSAT when possible.
    """
    candidates = all_candidate_specs(max_x=max_x, mode=mode)

    if len(candidates) < num_instances:
        raise ValueError(
            f"Requested {num_instances} instances but only {len(candidates)} "
            f"unique instances exist for mode={mode!r} with max_x={max_x}"
        )

    if mode != "mixed":
        return rng.sample(candidates, num_instances)

    sat_specs = [s for s in candidates if s["status"] == "sat"]
    unsat_specs = [s for s in candidates if s["status"] == "unsat"]

    # If the user asked for mixed and num_instances >= 2, force both types when possible.
    chosen: List[Dict[str, Any]] = []
    remaining_pool = list(candidates)

    if num_instances >= 2 and sat_specs and unsat_specs:
        s1 = rng.choice(sat_specs)
        s2 = rng.choice(unsat_specs)
        chosen.extend([s1, s2])

        used = {(s1["n"], s1["m"]), (s2["n"], s2["m"])}
        remaining_pool = [x for x in remaining_pool if (x["n"], x["m"]) not in used]

        need = num_instances - 2
        if need > 0:
            chosen.extend(rng.sample(remaining_pool, need))
        rng.shuffle(chosen)
        return chosen

    # Fallback: just sample from union if forcing both types is impossible.
    return rng.sample(candidates, num_instances)


def write_run_meta(
    run_dir: str,
    run_id: int,
    num_instances: int,
    max_x: int,
    mode: str,
    seed0: Optional[int],
) -> str:
    path = os.path.join(run_dir, f"run_{run_id}_meta.txt")
    content = {
        "run_id": run_id,
        "generator": "pigeonhole_generator_v1",
        "n_instances": num_instances,
        "max_x_exclusive": max_x,
        "mode": mode,
        "seed0": seed0,
        "instance_rule": "all instances satisfy 1 <= n,m < max_x",
        "sat_rule": "SAT iff n <= m",
        "unsat_rule": "UNSAT iff n > m",
        "uniqueness_rule": "no duplicate (n,m) pair within a run",
        "cnf_behavior": "keep formula exactly as produced by cnfgen/pysat",
    }
    write_json_atomic(path, content)
    return path


def write_instance_meta(
    run_dir: str,
    run_id: int,
    instance_id: int,
    instance_seed: int,
    spec: Dict[str, Any],
    cnf_filename: str,
) -> str:
    path = os.path.join(run_dir, f"run_{run_id}_instance_{instance_id}.txt")
    rec = {
        "run_id": run_id,
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "n": spec["n"],
        "m": spec["m"],
        "status": spec["status"],
        "cnf_file": cnf_filename,
    }
    write_json_atomic(path, rec)
    return path


def append_manifest(run_dir: str, rec: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "manifest.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def generate_pigeonhole_instances(
    *,
    num_instances: int,
    max_x: int,
    mode: str,
    run_id: int,
    out_root: str = PH_ROOT,
    seed0: Optional[int] = None,
) -> Dict[str, Any]:
    """
    mode in {"sat", "unsat", "mixed"}
    """
    mode = mode.lower().strip()
    if mode not in {"sat", "unsat", "mixed"}:
        raise ValueError("mode must be one of {'sat', 'unsat', 'mixed'}")

    rng = random.Random(seed0)

    run_id, run_dir = resolve_run_directory(out_root, run_id)
    run_meta = write_run_meta(
        run_dir=run_dir,
        run_id=run_id,
        num_instances=num_instances,
        max_x=max_x,
        mode=mode,
        seed0=seed0,
    )

    chosen_specs = choose_unique_specs(
        num_instances=num_instances,
        max_x=max_x,
        mode=mode,
        rng=rng,
    )

    written_cnf: List[str] = []
    written_meta: List[str] = []

    for instance_id, spec in enumerate(chosen_specs, start=1):
        instance_seed = rng.randint(0, 2**31 - 1)

        clauses, nv = cnf_prep(spec["n"], spec["m"])
        dimacs_text = clauses_to_dimacs(clauses, nv)

        cnf_name = f"run_{run_id}_instance_{instance_id}.cnf"
        cnf_path = os.path.join(run_dir, cnf_name)
        write_text_atomic(cnf_path, dimacs_text)

        meta_path = write_instance_meta(
            run_dir=run_dir,
            run_id=run_id,
            instance_id=instance_id,
            instance_seed=instance_seed,
            spec=spec,
            cnf_filename=cnf_name,
        )

        append_manifest(
            run_dir,
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "instance_seed": instance_seed,
                "n": spec["n"],
                "m": spec["m"],
                "status": spec["status"],
                "cnf_file": cnf_name,
                "meta_file": os.path.basename(meta_path),
            },
        )

        written_cnf.append(cnf_path)
        written_meta.append(meta_path)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "run_meta": run_meta,
        "cnf_files_written": len(written_cnf),
        "instance_meta_files_written": len(written_meta),
    }


def load_instance_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_dimacs_cnf(cnf_path: Path) -> Tuple[List[List[int]], int]:
    clauses: List[List[int]] = []
    nv = 0
    current: List[int] = []

    with open(cnf_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue

            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    nv = int(parts[2])
                continue

            for tok in line.split():
                lit = int(tok)
                if lit == 0:
                    if current:
                        clauses.append(current)
                        current = []
                else:
                    current.append(lit)

    if current:
        raise ValueError(f"Incomplete clause at end of file: {cnf_path}")
    if nv <= 0:
        raise ValueError(f"Missing or invalid DIMACS header in: {cnf_path}")

    return clauses, nv


def cnf_file_to_gapboxes(cnf_path: Path):
    clauses, nv = parse_dimacs_cnf(cnf_path)
    grouping = [1] * nv
    ordering = None

    gmap, gapboxes = cnf_to_gapboxes_raw(
        clauses,
        grouping=grouping,
        ordering=ordering,
    )
    dyadic_dims = [1 << w for w in gmap.widths]
    return gapboxes, dyadic_dims


def tetris_safe_name(path: Path) -> str:
    return path.name if path.exists() else path.name


def process_pigeonhole_run(run_dir: str) -> Dict[str, Any]:
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(run_path)

    base = run_path.name
    if not base.startswith("run_"):
        raise ValueError(f"Expected run_<id> directory, got {run_path}")

    run_id = int(base.split("_", 1)[1])

    sat_run_dir = SAT_ROOT / f"run_{run_id}"
    tetris_run_dir = TETRIS_ROOT / f"run_{run_id}"
    _ensure_dir(str(sat_run_dir))
    _ensure_dir(str(tetris_run_dir))

    save_cadical_config(str(sat_run_dir), run_id)

    stats = {
        "run_id": run_id,
        "source_run_dir": str(run_path),
        "processed_this_run": 0,
        "jobs": [],
    }

    cnf_files = sorted(run_path.glob("*.cnf"), key=lambda p: p.name.lower())

    for local_idx, cnf_path in enumerate(cnf_files, start=1):
        base_name = cnf_path.stem
        meta_path = run_path / f"{base_name}.txt"
        meta = load_instance_json(meta_path) if meta_path.exists() else {}

        # CaDiCaL
        sat_result = run_cadical_on_one_cnf(
            str(cnf_path),
            run_id=run_id,
            instance_id=local_idx,
            out_dir=str(sat_run_dir),
        )

        sat_default_proof = sat_run_dir / f"proof_run_{run_id}_instance_{local_idx}.lrat"
        sat_default_log = sat_run_dir / f"action_run_{run_id}_instance_{local_idx}.log"
        sat_default_trace = sat_run_dir / f"trace_run_{run_id}_instance_{local_idx}"

        # Tetris 
        gapboxes, dyadic_dims = cnf_file_to_gapboxes(cnf_path)

        _, produced_trace_path = run_tetris_on_gaps(
            global_gaps=gapboxes,
            dyadic_dims=dyadic_dims,
            trace_dir=str(tetris_run_dir),
            run_id=run_id,
            instance_id=local_idx,
            halt_first=True,
            trace_flush_every=1000,
        )

        produced_trace = Path(produced_trace_path)

        job = {
            "local_index": local_idx,
            "source_cnf": str(cnf_path),
            "base_name": base_name,
            "instance_meta": meta,
            "sat": {
                "proof": sat_default_proof.name,
                "log": sat_default_log.name,
                "trace": tetris_safe_name(sat_default_trace),
                "returncode": sat_result.get("returncode"),
            },
            "tetris": {
                "trace": tetris_safe_name(produced_trace),
            },
        }

        stats["jobs"].append(job)
        stats["processed_this_run"] = local_idx

    stats_path = run_path / f"run_{run_id}_solve_stats.json"
    _write_json(str(stats_path), stats)

    return {
        "run_id": run_id,
        "source_run_dir": str(run_path),
        "sat_run_dir": str(sat_run_dir),
        "tetris_run_dir": str(tetris_run_dir),
        "stats_file": str(stats_path),
        "processed_this_run": stats["processed_this_run"],
    }


if __name__ == "__main__":
    generated = generate_pigeonhole_instances(
        num_instances=3,
        max_x=5,     # all instances satisfy 1 <= n,m < 10
        mode="unsat", # "sat", "unsat", or "mixed"
        run_id=1,
        seed0=0,
    )
    solved = process_pigeonhole_run(generated["run_dir"])
    print(generated)
    print(solved)