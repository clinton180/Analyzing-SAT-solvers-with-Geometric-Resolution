from __future__ import annotations

import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Iterator, Optional
from join_queries import run_cadical_on_one_cnf, save_cadical_config, _ensure_dir, _write_json
from pathlib import Path

CNF_ROOT = Path("HMV_cnf").resolve()
SAT_ROOT = Path("hmv_sat").resolve()

# manual cap for how many CNFs this distinct run should process
MAX_FILES_THIS_RUN = 50


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj: dict) -> None:
    _write_json(str(path), obj)


def read_cnf_var_count(path: Path) -> int:
    """
    Read DIMACS header and return the declared number of variables.
    Expects a line like: p cnf <num_vars> <num_clauses>
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    return int(parts[2])
                raise ValueError(f"Malformed DIMACS header in {path}: {line}")

    raise ValueError(f"No DIMACS header found in {path}")


def next_run_id(root: Path) -> int:
    if not root.exists():
        return 1

    max_id = 0
    for name in os.listdir(root):
        if not name.startswith("run_"):
            continue
        try:
            rid = int(name.split("_", 1)[1])
            max_id = max(max_id, rid)
        except Exception:
            pass

    return max_id + 1


def iter_cnf_dfs(root: Path) -> Iterator[Path]:
    """
    Deterministic DFS traversal over CNF tree.
    Only yields .cnf files.
    """
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]

    for d in dirs:
        yield from iter_cnf_dfs(d)

    for f in sorted(files, key=lambda p: p.name.lower()):
        if f.suffix.lower() == ".cnf":
            yield f


def make_prefix(idx: int) -> str:
    return f"{idx:03d}"


def main() -> None:
    if not CNF_ROOT.exists():
        raise FileNotFoundError(f"CNF root does not exist: {CNF_ROOT}")

    _ensure_dir(str(SAT_ROOT))

    # -------- global stats across all runs --------
    global_stats_path = SAT_ROOT / "hmv_sat_stats.json"
    global_stats = load_json(
        global_stats_path,
        {
            "source_root": str(CNF_ROOT),
            "processed_count": 0,
            "runs_created": 0,
            "last_run_id": 0,
            "history": [],
        },
    )

    already_done = int(global_stats.get("processed_count", 0))

    # -------- distinct new run --------
    run_id = next_run_id(SAT_ROOT)
    run_dir = SAT_ROOT / f"run_{run_id}"
    _ensure_dir(str(run_dir))

    save_cadical_config(str(run_dir), run_id)

    run_stats_path = run_dir / f"run_{run_id}_stats.json"
    run_stats = {
        "run_id": run_id,
        "source_root": str(CNF_ROOT),
        "global_processed_count_at_start": already_done,
        "processed_this_run": 0,
        "jobs": [],
    }
    save_json(run_stats_path, run_stats)

    seen = 0
    processed_this_run = 0

    for cnf_path in iter_cnf_dfs(CNF_ROOT):
        seen += 1

        # skip everything already consumed by earlier runs
        if seen <= already_done:
            continue

        # stop this distinct run after its manual batch size
        if MAX_FILES_THIS_RUN is not None and processed_this_run >= MAX_FILES_THIS_RUN:
            break

        global_idx = seen
        local_idx = processed_this_run + 1

        prefix = make_prefix(local_idx)
        base = cnf_path.stem

        proof_name = f"{prefix}_{base}.lrat"
        log_name = f"{prefix}_{base}.log"
        trace_name = f"{prefix}_{base}"

        if (
            (run_dir / proof_name).exists()
            or (run_dir / log_name).exists()
            or (run_dir / trace_name).exists()
        ):
            raise FileExistsError(
                f"Output collision for local index {local_idx} and base '{base}' in {run_dir}"
            )

        result = run_cadical_on_one_cnf(
            str(cnf_path),
            run_id=run_id,
            instance_id=local_idx,
            out_dir=str(run_dir),
        )

        # rename from generate.py default naming into your flat run naming
        default_proof = run_dir / f"proof_run_{run_id}_instance_{local_idx}.lrat"
        default_log = run_dir / f"action_run_{run_id}_instance_{local_idx}.log"
        default_trace = run_dir / f"trace_run_{run_id}_instance_{local_idx}"
        processed_this_run += 1

        # update per-run stats 
        var_count = read_cnf_var_count(cnf_path)

        job_record = {
            "local_index": local_idx,
            "global_dfs_index": global_idx,
            "source_cnf": str(cnf_path),
            "base_name": base,
            "var_count": var_count,
            "proof": default_proof.name,
            "log": default_log.name,
            "trace": default_trace.name,
            "returncode": result.get("returncode"),
        }

        run_stats["processed_this_run"] = processed_this_run
        run_stats["last_processed_cnf"] = str(cnf_path)
        run_stats["jobs"].append(job_record)
        save_json(run_stats_path, run_stats)

        # update global stats 
        global_stats["processed_count"] = global_idx
        global_stats["runs_created"] = max(int(global_stats.get("runs_created", 0)), run_id)
        global_stats["last_run_id"] = run_id
        save_json(global_stats_path, global_stats)

        print(f"[run_{run_id} | {prefix}] processed {cnf_path.name}")

    # finalize global history
    global_stats = load_json(global_stats_path, global_stats)
    global_stats.setdefault("history", []).append(
        {
            "run_id": run_id,
            "processed_this_run": processed_this_run,
            "global_processed_count_after_run": global_stats.get("processed_count", already_done),
            "run_dir": str(run_dir),
        }
    )
    save_json(global_stats_path, global_stats)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "global_processed_before_run": already_done,
                "processed_this_run": processed_this_run,
                "global_processed_after_run": global_stats.get("processed_count", already_done),
                "run_dir": str(run_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()