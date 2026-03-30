# pebbling_generate.py
from __future__ import annotations

import os
import sys
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from join_queries import resolve_run_directory

CNFGEN_BIN = "cnfgen"
PEBBLING_ROOT = "pebbling_cnf"

from join_queries import (
    run_cadical_on_one_cnf,
    save_cadical_config,
    _ensure_dir,
    _write_json,
    run_tetris_on_gaps, run_tetris_unit_traces_for_run_dir
)
from Comp_499A.SAT.sat_gap import cnf_to_gapboxes_raw


SAT_ROOT = Path("pebbling_sat").resolve()
TETRIS_ROOT = Path("pebbling_tetris").resolve()


def write_text_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(obj, indent=2, sort_keys=True))


def sample_instance_spec(rng: random.Random) -> Dict[str, Any]:
    height = rng.randint(3, 13)
    transformed = rng.choice([False, True])

    lift_k: Optional[int]
    if transformed:
        lift_k = 1
    else:
        lift_k = None

    return {
        "height": height,
        "transformed": transformed,
        "lift_k": lift_k,
    }


def run_cnfgen_pyramid(height: int, transformed: bool, lift_k: Optional[int]) -> str:
    cmd = [CNFGEN_BIN, "peb", "pyramid", str(height)]
    if transformed:
        cmd += ["-T", "lift", str(lift_k)]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout


def write_run_meta(
    run_dir: str,
    run_id: int,
    n: int,
    seed0: Optional[int],
) -> str:
    path = os.path.join(run_dir, f"run_{run_id}_meta.txt")
    content = {
        "run_id": run_id,
        "generator": "pebbling_pyramid_generator_v1",
        "cnfgen_bin": CNFGEN_BIN,
        "n_instances": n,
        "seed0": seed0,
        "height_range_inclusive": [5, 10],
        "transform_choice": "uniform over {False, True}",
        "lift_k_choice_if_transformed": "uniform over {1,2,3}",
        "untransformed_behavior": "remove positive source unit clauses",
        "transformed_behavior": "keep cnfgen output exactly as produced",
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
    cnfgen_command: List[str],
) -> str:
    path = os.path.join(run_dir, f"run_{run_id}_instance_{instance_id}.txt")
    rec = {
        "run_id": run_id,
        "instance_id": instance_id,
        "instance_seed": instance_seed,
        "height": spec["height"],
        "transformed": spec["transformed"],
        "lift_k": spec["lift_k"],
        "cnf_file": cnf_filename,
        "cnfgen_command": cnfgen_command,
    }
    write_json_atomic(path, rec)
    return path


def append_manifest(run_dir: str, rec: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, "manifest.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def generate_pebbling_instances(
    *,
    n: int,
    run_id: int,
    out_root: str = PEBBLING_ROOT,
    seed0: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed0)

    run_id, run_dir = resolve_run_directory(out_root, run_id)
    run_meta = write_run_meta(run_dir, run_id, n, seed0)

    written_cnf: List[str] = []
    written_meta: List[str] = []

    for instance_id in range(1, n + 1):
        instance_seed = rng.randint(0, 2**31 - 1)
        inst_rng = random.Random(instance_seed)

        spec = sample_instance_spec(inst_rng)

        cnfgen_cmd = [CNFGEN_BIN, "peb", "pyramid", str(spec["height"])]
        if spec["transformed"]:
            cnfgen_cmd += ["-T", "lift", str(spec["lift_k"])]

        raw_dimacs = run_cnfgen_pyramid(
            height=spec["height"],
            transformed=spec["transformed"],
            lift_k=spec["lift_k"],
        )
        final_dimacs = raw_dimacs
        cnf_name = f"run_{run_id}_instance_{instance_id}.cnf"
        cnf_path = os.path.join(run_dir, cnf_name)
        write_text_atomic(cnf_path, final_dimacs)

        meta_path = write_instance_meta(
            run_dir=run_dir,
            run_id=run_id,
            instance_id=instance_id,
            instance_seed=instance_seed,
            spec=spec,
            cnf_filename=cnf_name,
            cnfgen_command=cnfgen_cmd,
        )

        append_manifest(
            run_dir,
            {
                "run_id": run_id,
                "instance_id": instance_id,
                "instance_seed": instance_seed,
                "height": spec["height"],
                "transformed": spec["transformed"],
                "lift_k": spec["lift_k"],
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
        "cnf_files written:": len(written_cnf),
        "instance_meta_files written:": len(written_meta),
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


def process_pebbling_run(run_dir: str) -> Dict[str, Any]:
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

        # ---- SAT ----
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
                "trace": sat_default_trace.name,
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


def tetris_safe_name(path: Path) -> str:
    return path.name if path.exists() else path.name


if __name__ == "__main__":
    generated = generate_pebbling_instances(n=20, run_id=1, seed0=20)
    solved = process_pebbling_run(generated["run_dir"])

    print(json.dumps({
        "generation": generated,
        "solving": solved,
    }, indent=2))
