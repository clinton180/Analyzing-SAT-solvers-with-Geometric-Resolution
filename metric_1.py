from __future__ import annotations
import os, sys, json, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# Parsing helpers
DECISION_RE = re.compile(r"\bASSIGN\s+kind=DECISION\s+lit=([+-]?\d+)\b")
EVENT_RE = re.compile(r'"event"\s*:\s*"([^"]+)"')


@dataclass
class TetrisTrace:
    n_attrs: int
    widths: List[int]
    sao: List[int]
    split_dims_all: List[int]
    split_dims_forced: List[int]  # optional; can be empty if you don't compute it

def parse_tetris_jsonl(path: str) -> TetrisTrace:
    n_attrs: Optional[int] = None
    widths: Optional[List[int]] = None
    sao: Optional[List[int]] = None
    split_dims_all: List[int] = []

    # If you haven't implemented forced-split detection yet, keep this empty.
    split_dims_forced: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            ev = obj.get("event")
            if ev == "START":
                n_attrs = int(obj["n"])
                widths = list(obj["widths"])
                sao = list(obj.get("sao", []))
            elif ev == "SPLIT":
                d = obj.get("dim", None)
                if d is None:
                    continue
                split_dims_all.append(int(d))

    if n_attrs is None or widths is None:
        raise ValueError(f"{path}: missing START line (n/widths).")

    if sao is None:
        sao = list(range(n_attrs))

    return TetrisTrace(
        n_attrs=n_attrs,
        widths=widths,
        sao=sao,
        split_dims_all=split_dims_all,
        split_dims_forced=split_dims_forced,
    )

def parse_sat_decision_attrs(trace_path: str, widths: List[int]) -> List[int]:
    """
    Returns attribute ids (0..n-1) for each DECISION assignment.
    Uses var->(attr,bit) mapping induced by widths.
    """
    # Build var -> attr map: var 1..sum(widths)
    var_to_attr: Dict[int, int] = {}
    v = 1
    for a, w in enumerate(widths):
        for _ in range(w):
            var_to_attr[v] = a
            v += 1

    out: List[int] = []

    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Example: ASSIGN kind=DECISION lit=12 level=1 ...
            if line.startswith("ASSIGN") and "kind=DECISION" in line:
                m = re.search(r"\blit=(-?\d+)\b", line)
                if not m:
                    continue
                lit = int(m.group(1))
                var = abs(lit)
                a = var_to_attr.get(var)
                if a is not None:
                    out.append(a)

    return out

def parse_sat_full_assignment_vars(trace_path: str) -> List[int]:
    """
    Returns the full SAT assignment sequence as raw variable ids (positive indices only).

    Included:
      - MARK_FIXED
      - ASSIGN kind=DECISION
      - ASSIGN kind=IMPLIED

    Excluded as duplicates:
      - ASSIGN kind=UNIT   (duplicates preceding MARK_FIXED)
      - ASSERT_ASSIGN      (duplicates following ASSIGN kind=IMPLIED)

    Output is absolute variable ids in chronological order.
    """

    out: List[int] = []

    # Track literals already emitted by MARK_FIXED so we can suppress the
    # immediately duplicated ASSIGN kind=UNIT for the same literal.
    pending_fixed_lits: List[int] = []

    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # ---- MARK_FIXED: include ----
            if line.startswith("MARK_FIXED"):
                m = re.search(r"\blit=(-?\d+)\b", line)
                if not m:
                    continue
                lit = int(m.group(1))
                out.append(abs(lit))
                pending_fixed_lits.append(lit)
                continue

            # ---- ASSIGN lines ----
            if line.startswith("ASSIGN"):
                m = re.search(r"\blit=(-?\d+)\b", line)
                if not m:
                    continue
                lit = int(m.group(1))

                # Skip UNIT if it is duplicating a prior MARK_FIXED
                if "kind=UNIT" in line:
                    if pending_fixed_lits and pending_fixed_lits[-1] == lit:
                        pending_fixed_lits.pop()
                        continue
                    # Fallback: if a UNIT appears without matching MARK_FIXED,
                    # ignore it anyway since your rule is not to include UNIT.
                    continue

                # Include decisions and implications
                if "kind=DECISION" in line or "kind=IMPLIED" in line:
                    out.append(abs(lit))
                    continue

            # ASSERT_ASSIGN: skip
            # It is duplicated by the following ASSIGN kind=IMPLIED.
            if line.startswith("ASSERT_ASSIGN"):
                continue
    return out


def build_var_to_attr(widths: List[int]) -> Dict[int, int]:
    """
    Build var -> attr map assuming variables are laid out contiguously by widths.
    """
    var_to_attr: Dict[int, int] = {}
    v = 1
    for a, w in enumerate(widths):
        for _ in range(w):
            var_to_attr[v] = a
            v += 1
    return var_to_attr


def vars_to_attrs(var_seq: List[int], widths: List[int]) -> List[int]:
    """
    Convert raw SAT variable ids to attribute ids using widths.
    """
    var_to_attr = build_var_to_attr(widths)
    out: List[int] = []
    for v in var_seq:
        a = var_to_attr.get(v)
        if a is not None:
            out.append(a)
    return out


def raw_vars_to_reversed_ranks(var_seq: List[int], total_vars: int) -> List[int]:
    """
    Convert raw SAT vars to reversed raw-variable ranks:
    """
    out: List[int] = []
    for v in var_seq:
        if 1 <= v <= total_vars:
            out.append(total_vars - v - 1)
    return out

# Windowing
def window_slices(length: int, W: int) -> List[Tuple[int, int]]:
    """
    Returns W slices [lo, hi) that partition indices 0..length.
    Uses floor boundaries; last window closes at length.
    """
    out = []
    for w in range(W):
        lo = (w * length) // W
        hi = ((w + 1) * length) // W
        out.append((lo, hi))
    return out


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_text(path: str, s: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    os.replace(tmp, path)

def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def plot_mean_std_band(xs: List[float], mean_curve: List[float], std_curve: List[float],
                       ylabel: str, title: str, out_png: str) -> None:
    lo = [max(0.0, mean_curve[i] - std_curve[i]) if not math.isnan(mean_curve[i]) else float("nan")
          for i in range(len(mean_curve))]
    hi = [mean_curve[i] + std_curve[i] if not math.isnan(mean_curve[i]) else float("nan")
          for i in range(len(mean_curve))]

    plt.figure()
    plt.plot(xs, mean_curve, marker="o")
    plt.fill_between(xs, lo, hi, alpha=0.25)
    plt.xlabel("Time (window midpoint; percentile of trace)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_per_instance_attr_scatter(
    sat_attrs: List[int],
    tetris_attrs: List[int],
    out_png: str,
    *,
    inst_id: int,
) -> None:
    """
    Per-instance scatterplot:
      - SAT decision attribute trace
      - Tetris split attribute trace

    x-axis: raw trace index
    y-axis: attribute id

    SAT and Tetris points are given a small vertical offset so that
    identical coordinates do not fully cover each other.
    """

    if not sat_attrs and not tetris_attrs:
        return

    max_x = max(
        len(sat_attrs) if sat_attrs else 0,
        len(tetris_attrs) if tetris_attrs else 0,
    )
    max_y = max(
        max(sat_attrs) if sat_attrs else 0,
        max(tetris_attrs) if tetris_attrs else 0,
    )

    # Dynamic figure size
    width = min(max(8.0, max_x * 0.08), 20.0)
    height = min(max(3.0, (max_y + 1) * 1.15), 8.0)

    fig, ax = plt.subplots(figsize=(width, height))

    # Small vertical offsets so overlaps remain visible
    sat_offset = -1
    tet_offset = 0

    if sat_attrs:
        xs_sat = list(range(len(sat_attrs)))
        ys_sat = [a + sat_offset for a in sat_attrs]
        ax.scatter(
            xs_sat,
            ys_sat,
            s=85,
            color="navy",
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            label="SAT decisions",
            zorder=2,
        )

    if tetris_attrs:
        xs_tet = list(range(len(tetris_attrs)))
        ys_tet = [a + tet_offset for a in tetris_attrs]
        ax.scatter(
            xs_tet,
            ys_tet,
            s=85,
            color="darkred",
            alpha=0.95,
            edgecolors="black",
            linewidths=0.6,
            label="Tetris splits",
            zorder=3,
        )

    # Integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Tight axis limits
    ax.set_ylim(-0.35, max_y + 0.35)
    ax.set_xlim(-0.5, max_x - 0.5 if max_x > 0 else 0.5)

    ax.set_xlabel("Trace index")
    ax.set_ylabel("Attribute id")
    ax.set_title(f"Instance {inst_id}: SAT vs Tetris attribute traces")

    # Leave room on the right for legend
    plt.legend()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    
    plt.close(fig)


def sat_only():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_i", type=int, help="run index i (folder run_i)")
    ap.add_argument("--sat_root", default="/home/catalyst/project/hmv_sat", help="root directory for sat traces")
    ap.add_argument("--mode", choices=["all_splits", "forced_splits"], default="all_splits")
    args = ap.parse_args()

    sat_dir = os.path.join(args.sat_root, f"run_{args.run_i}")

    if not os.path.isdir(sat_dir):
        raise FileNotFoundError(f"Missing sat dir: {sat_dir}")

    sat_stats_path = os.path.join(sat_dir, f"run_{args.run_i}_stats.json")
    if not os.path.isfile(sat_stats_path):
        raise FileNotFoundError(f"Missing SAT stats file: {sat_stats_path}")

    sat_files = sorted(glob.glob(os.path.join(sat_dir, f"trace_run_{args.run_i}_instance_*")))
    if not sat_files:
        raise FileNotFoundError(f"No SAT traces matching trace_run_{args.run_i}_instance_* in {sat_dir}")

    def extract_instance_id(p: str) -> Optional[int]:
        b = os.path.basename(p)
        m = re.search(r"(?:instance|inst)_(\d+)", b)
        return int(m.group(1)) if m else None

    # load run stats and build map: instance_id -> var_count
    with open(sat_stats_path, "r", encoding="utf-8") as f:
        sat_stats = json.load(f)

    var_count_map: Dict[int, int] = {}
    for job in sat_stats.get("jobs", []):
        inst_id = job.get("local_index")
        var_count = job.get("var_count")

        if inst_id is None:
            continue
        if var_count is None:
            raise ValueError(
                f"Missing 'var_count' for local_index={inst_id} in {sat_stats_path}"
            )

        var_count_map[int(inst_id)] = int(var_count)

    s_map: Dict[int, str] = {}
    for p in sat_files:
        j = extract_instance_id(p)
        if j is not None:
            s_map[j] = p

    # each pair = (instance_id, sat_trace_path, var_count)
    pairs: List[Tuple[int, str, int]] = []
    if s_map:
        common = sorted(set(s_map.keys()))
        for j in common:
            if j not in var_count_map:
                raise ValueError(
                    f"No var_count entry found in stats json for SAT trace instance {j}"
                )
            pairs.append((j, s_map[j], var_count_map[j]))

    if not pairs:
        raise RuntimeError("No paired instances found between SAT traces and run stats")

    results_root = os.path.join(os.getcwd(), "visualization", f"run_{args.run_i}")
    plots_dir = os.path.join(results_root, "per_instance_attr_scatter")
    ensure_dir(plots_dir)

    used = 0
    skipped = 0

    for j, spath, total_vars in pairs:
        widths = [1] * total_vars

        sat_attrs = parse_sat_full_assignment_vars(spath)

        if len(sat_attrs) == 0:
            print(f"[SKIP] inst={j} sat_decisions=0")
            skipped += 1
            continue

        sat_attrs = vars_to_attrs(sat_attrs, widths=widths)
        sat_raw_rank = raw_vars_to_reversed_ranks(sat_attrs, total_vars)

        out_png = os.path.join(plots_dir, f"instance_{j}_sat_vs_tetris_attr_scatter.png")
        plot_per_instance_attr_scatter(
            sat_attrs=sat_raw_rank,
            tetris_attrs=[],
            out_png=out_png,
            inst_id=j,
        )
        used += 1
        print(f"[OK] inst={j} -> {out_png}")

    print()
    print(f"Run: run_{args.run_i}")
    print(f"Mode: {args.mode}")
    print(f"Paired instances: {len(pairs)}")
    print(f"Plots written: {used}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {plots_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_i", type=int, help="run index i (folder run_i)")
    ap.add_argument("--tetris_root", default="/home/catalyst/project/pigeonhole_tetris", help="root directory for tetris traces")
    ap.add_argument("--sat_root", default="/home/catalyst/project/pigeonhole_sat", help="root directory for sat traces")
    ap.add_argument("--mode", choices=["all_splits", "forced_splits"], default="all_splits")
    args = ap.parse_args()

    tetris_dir = os.path.join(args.tetris_root, f"run_{args.run_i}")
    sat_dir = os.path.join(args.sat_root, f"run_{args.run_i}")

    if not os.path.isdir(tetris_dir):
        raise FileNotFoundError(f"Missing tetris dir: {tetris_dir}")
    if not os.path.isdir(sat_dir):
        raise FileNotFoundError(f"Missing sat dir: {sat_dir}")

    tetris_files = sorted(glob.glob(os.path.join(tetris_dir, "*.jsonl")))
    if not tetris_files:
        raise FileNotFoundError(f"No .jsonl tetris traces in {tetris_dir}")

    sat_files = sorted(glob.glob(os.path.join(sat_dir, f"trace_run_{args.run_i}_instance_*")))
    if not sat_files:
        raise FileNotFoundError(f"No SAT traces matching trace_run_{args.run_i}_instance_* in {sat_dir}")

    def extract_instance_id(p: str) -> Optional[int]:
        b = os.path.basename(p)
        m = re.search(r"(?:instance|inst)_(\d+)", b)
        return int(m.group(1)) if m else None

    t_map: Dict[int, str] = {}
    for p in tetris_files:
        j = extract_instance_id(p)
        if j is not None:
            t_map[j] = p

    s_map: Dict[int, str] = {}
    for p in sat_files:
        j = extract_instance_id(p)
        if j is not None:
            s_map[j] = p

    pairs: List[Tuple[int, str, str]] = []
    if t_map and s_map:
        common = sorted(set(t_map.keys()) & set(s_map.keys()))
        for j in common:
            pairs.append((j, t_map[j], s_map[j]))
    else:
        L = min(len(tetris_files), len(sat_files))
        for idx in range(L):
            pairs.append((idx + 1, tetris_files[idx], sat_files[idx]))

    if not pairs:
        raise RuntimeError("No paired instances found between Tetris and SAT traces")

    results_root = os.path.join(os.getcwd(), "visualization", f"run_{args.run_i}")
    plots_dir = os.path.join(results_root, "per_instance_attr_scatter")
    ensure_dir(plots_dir)

    used = 0
    skipped = 0

    for j, tpath, spath in pairs:
        t = parse_tetris_jsonl(tpath)
        total_vars = sum(t.widths)

        sat_attrs = parse_sat_full_assignment_vars(spath)
        t_seq = t.split_dims_all if args.mode == "all_splits" else t.split_dims_forced

        if len(sat_attrs) == 0 or len(t_seq) == 0:
            print(
                f"[SKIP] inst={j} "
                f"sat_decisions={len(sat_attrs)} "
                f"tetris_splits={len(t_seq)}"
            )
            skipped += 1
            continue

        
        sat_attrs = vars_to_attrs(sat_attrs, widths=t.widths)
        sat_raw_rank = raw_vars_to_reversed_ranks(sat_attrs, total_vars)
        out_png = os.path.join(plots_dir, f"instance_{j}_sat_vs_tetris_attr_scatter.png")
        out_reversed_png = os.path.join(plots_dir, f"instance_{j}_sat_vs_tetris_attr_scatter_reversed.png")

        plot_per_instance_attr_scatter(
            sat_attrs=sat_raw_rank,
            tetris_attrs=t_seq,
            out_png=out_png,
            inst_id=j,
        )

        plot_per_instance_attr_scatter(
            sat_attrs=sat_attrs,
            tetris_attrs=t_seq,
            out_png=out_reversed_png,
            inst_id=j,
        )
        used += 1
        print(f"[OK] inst={j}")

    print()
    print(f"Run: run_{args.run_i}")
    print(f"Mode: {args.mode}")
    print(f"Paired instances: {len(pairs)}")
    print(f"Plots written: {used}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {plots_dir}")

if __name__ == "__main__":
    main()