from __future__ import annotations
import os, sys, json, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import glob
import json
import math
import os
import re
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from metric_1 import *
import matplotlib.pyplot as plt


# Parsing helpers

DECISION_RE = re.compile(r"\bASSIGN\s+kind=DECISION\s+lit=([+-]?\d+)\b")

def parse_sat_decision_vars(trace_path: str) -> List[int]:
    """
    Returns raw SAT variable ids (positive indices only) for each DECISION assignment.
    """
    out: List[int] = []

    with open(trace_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("ASSIGN") and "kind=DECISION" in line:
                m = re.search(r"\blit=(-?\d+)\b", line)
                if not m:
                    continue
                lit = int(m.group(1))
                out.append(abs(lit))

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


def sao_rank_map(sao: List[int], n_attrs: int) -> List[int]:
    """
    Returns rank_of_attr[a] = rank in SAO order (0..n-1).
    If sao is empty/invalid, fall back to identity.
    """
    if not sao or len(sao) != n_attrs:
        return list(range(n_attrs))

    rank_of_attr = [0] * n_attrs
    for r, a in enumerate(sao):
        if 0 <= a < n_attrs:
            rank_of_attr[a] = r
        else:
            return list(range(n_attrs))
    return rank_of_attr


def raw_vars_to_reversed_ranks(var_seq: List[int], total_vars: int) -> List[int]:
    """
    Convert raw SAT vars to reversed raw-variable ranks:
        var = total_vars   -> rank 0
        var = total_vars-1 -> rank 1
        ...
        var = 1            -> rank total_vars-1
    """
    out: List[int] = []
    for v in var_seq:
        if 1 <= v <= total_vars:
            out.append(total_vars - v)
    return out


def attrs_to_reversed_sao_ranks(
    attr_seq: List[int],
    rank_of_attr: List[int],
    n_attrs: int,
) -> List[int]:
    """
    Convert attribute ids to reversed SAO ranks:
        rev_rank(a) = (n_attrs - 1) - rank_of_attr[a]
    """
    out: List[int] = []
    for a in attr_seq:
        if 0 <= a < len(rank_of_attr):
            out.append((n_attrs - 1) - rank_of_attr[a])
    return out


# Illegal forward metrics

def illegal_forward_metrics_stream(rank_seq: List[int]) -> Dict[str, Any]:
    """
    Measure forward behavior on an already-inverted rank sequence.

    Rules:
      - curr <= prev : allowed (stay / backtrack)
      - curr == prev + 1 : legal forward step
      - curr > prev + 1  : illegal forward jump

    Returns:
      total_steps
      forward_steps
      illegal_forward_steps
      forward_density
      illegal_forward_rate
      illegal_forward_burden
    """
    total_steps = max(0, len(rank_seq) - 1)

    if len(rank_seq) <= 1:
        return {
            "total_steps": total_steps,
            "forward_steps": 0,
            "illegal_forward_steps": 0,
            "forward_density": 0.0,
            "illegal_forward_rate": 0.0,
            "illegal_forward_burden": 0.0,
        }

    forward_steps = 0
    illegal_forward_steps = 0

    prev = rank_seq[0]
    for curr in rank_seq[1:]:
        if curr > prev:
            forward_steps += 1
            if curr > prev + 1:
                illegal_forward_steps += 1
        prev = curr

    forward_density = forward_steps / total_steps if total_steps > 0 else 0.0
    illegal_forward_rate = (
        illegal_forward_steps / forward_steps if forward_steps > 0 else 0.0
    )
    illegal_forward_burden = (
        illegal_forward_steps / total_steps if total_steps > 0 else 0.0
    )

    return {
        "total_steps": total_steps,
        "forward_steps": forward_steps,
        "illegal_forward_steps": illegal_forward_steps,
        "forward_density": forward_density,
        "illegal_forward_rate": illegal_forward_rate,
        "illegal_forward_burden": illegal_forward_burden,
    }

def windowed_illegal_forward_curves(rank_seq: List[int], W: int) -> Dict[str, List[float]]:
    """
    Window the sequence and compute metrics independently per window.
    """
    slices = window_slices(len(rank_seq), W)

    forward_density_curve: List[float] = []
    illegal_forward_rate_curve: List[float] = []
    illegal_forward_burden_curve: List[float] = []

    for lo, hi in slices:
        win = rank_seq[lo:hi]
        m = illegal_forward_metrics_stream(win)
        forward_density_curve.append(float(m["forward_density"]))
        illegal_forward_rate_curve.append(float(m["illegal_forward_rate"]))
        illegal_forward_burden_curve.append(float(m["illegal_forward_burden"]))

    return {
        "forward_density": forward_density_curve,
        "illegal_forward_rate": illegal_forward_rate_curve,
        "illegal_forward_burden": illegal_forward_burden_curve,
    }


# Utilities
def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(v)


def plot_two_mean_lines(
    xs: List[float],
    y1: List[float],
    label1: str,
    ylabel: str,
    title: str,
    out_png: str,
) -> None:
    plt.figure()
    plt.plot(xs, y1, marker="o", label=label1)
    plt.xlabel("Time (window midpoint; percentile of trace)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_two_instance_lines(
    xs: List[int],
    y1: List[float],
    label1: str,
    ylabel: str,
    title: str,
    out_png: str,
) -> None:
    plt.figure()
    plt.plot(xs, y1, marker="o", label=label1)
    plt.xlabel("Instance id")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def aggregate_curves(curves_list: List[List[float]], W: int) -> Tuple[List[float], List[float]]:
    mean_c, std_c = [], []
    for w in range(W):
        vals = [c[w] for c in curves_list if w < len(c)]
        mean_c.append(mean(vals))
        std_c.append(std(vals))
    return mean_c, std_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_i", type=int, help="run index i (folder run_i)")
    ap.add_argument("--tetris_root", default="/home/catalyst/project/join_tetris_unit",
                    help="root directory for tetris traces")
    ap.add_argument("--sat_root", default="/home/catalyst/project/join_sat_outputs",
                    help="root directory for sat traces")
    ap.add_argument("--W", type=int, default=3, help="number of percentile windows")
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

    # Temporal containers across instances
    main_forward_density_curves: List[List[float]] = []
    main_illegal_forward_rate_curves: List[List[float]] = []
    main_illegal_forward_burden_curves: List[List[float]] = []

    reverse_forward_density_curves: List[List[float]] = []
    reverse_illegal_forward_rate_curves: List[List[float]] = []
    reverse_illegal_forward_burden_curves: List[List[float]] = []

    # Instance-level rows
    per_instance_rows: List[Dict[str, Any]] = []

    for j, tpath, spath in pairs:
        t = parse_tetris_jsonl(tpath)
        total_vars = sum(t.widths)

        sat_vars = parse_sat_full_assignment_vars(spath)
        if len(sat_vars) == 0:
            print(f"[SKIP] inst={j} sat_decisions=0 file={os.path.basename(spath)}")
            continue

        # MAIN = direct
        sat_main_rank = sat_vars

        # REVERSE = existing reversed form
        sat_reverse_rank = raw_vars_to_reversed_ranks(sat_vars, total_vars)

        # Global full-run metrics
        main_global = illegal_forward_metrics_stream(sat_main_rank)
        reverse_global = illegal_forward_metrics_stream(sat_reverse_rank)

        # Temporal curves
        main_windowed = windowed_illegal_forward_curves(sat_main_rank, args.W)
        reverse_windowed = windowed_illegal_forward_curves(sat_reverse_rank, args.W)

        main_forward_density_curves.append(main_windowed["forward_density"])
        main_illegal_forward_rate_curves.append(main_windowed["illegal_forward_rate"])
        main_illegal_forward_burden_curves.append(main_windowed["illegal_forward_burden"])

        reverse_forward_density_curves.append(reverse_windowed["forward_density"])
        reverse_illegal_forward_rate_curves.append(reverse_windowed["illegal_forward_rate"])
        reverse_illegal_forward_burden_curves.append(reverse_windowed["illegal_forward_burden"])

        per_instance_rows.append({
            "instance_id": j,
            "tetris_file": os.path.basename(tpath),
            "sat_file": os.path.basename(spath),

            # MAIN
            "total_steps": main_global["total_steps"],
            "forward_steps": main_global["forward_steps"],
            "illegal_forward_steps": main_global["illegal_forward_steps"],
            "forward_density": main_global["forward_density"],
            "illegal_forward_rate": main_global["illegal_forward_rate"],
            "illegal_forward_burden": main_global["illegal_forward_burden"],

            # REVERSE
            "total_steps_reverse": reverse_global["total_steps"],
            "forward_steps_reverse": reverse_global["forward_steps"],
            "illegal_forward_steps_reverse": reverse_global["illegal_forward_steps"],
            "forward_density_reverse": reverse_global["forward_density"],
            "illegal_forward_rate_reverse": reverse_global["illegal_forward_rate"],
            "illegal_forward_burden_reverse": reverse_global["illegal_forward_burden"],
        })

    results_root = os.path.join(os.getcwd(), "depth_first", f"run_{args.run_i}")
    plots_dir = os.path.join(results_root, "plots")
    ensure_dir(plots_dir)

    if per_instance_rows:
        W = args.W
        xs_temporal = [(k + 0.5) / W for k in range(W)]

        main_fd_mu, main_fd_sd = aggregate_curves(main_forward_density_curves, W)
        main_ir_mu, main_ir_sd = aggregate_curves(main_illegal_forward_rate_curves, W)
        main_ib_mu, main_ib_sd = aggregate_curves(main_illegal_forward_burden_curves, W)

        reverse_fd_mu, reverse_fd_sd = aggregate_curves(reverse_forward_density_curves, W)
        reverse_ir_mu, reverse_ir_sd = aggregate_curves(reverse_illegal_forward_rate_curves, W)
        reverse_ib_mu, reverse_ib_sd = aggregate_curves(reverse_illegal_forward_burden_curves, W)

        main_fd_global = [r["forward_density"] for r in per_instance_rows]
        main_ir_global = [r["illegal_forward_rate"] for r in per_instance_rows]
        main_ib_global = [r["illegal_forward_burden"] for r in per_instance_rows]

        reverse_fd_global = [r["forward_density_reverse"] for r in per_instance_rows]
        reverse_ir_global = [r["illegal_forward_rate_reverse"] for r in per_instance_rows]
        reverse_ib_global = [r["illegal_forward_burden_reverse"] for r in per_instance_rows]

        summary = {
            "run_i": args.run_i,
            "metric": "metric_2c_cdcl_only_illegal_forward_consistency",
            "W": args.W,
            "pairs_used": len(per_instance_rows),
            "global_instance_metrics": {
                "forward_density": {
                    "main_mean": mean(main_fd_global),
                    "main_std": std(main_fd_global),
                    "reverse_mean": mean(reverse_fd_global),
                    "reverse_std": std(reverse_fd_global),
                },
                "illegal_forward_rate": {
                    "main_mean": mean(main_ir_global),
                    "main_std": std(main_ir_global),
                    "reverse_mean": mean(reverse_ir_global),
                    "reverse_std": std(reverse_ir_global),
                },
                "illegal_forward_burden": {
                    "main_mean": mean(main_ib_global),
                    "main_std": std(main_ib_global),
                    "reverse_mean": mean(reverse_ib_global),
                    "reverse_std": std(reverse_ib_global),
                },
            },
            "temporal_window_metrics": {
                "forward_density": {
                    "main_mean": main_fd_mu,
                    "main_std": main_fd_sd,
                    "reverse_mean": reverse_fd_mu,
                    "reverse_std": reverse_fd_sd,
                },
                "illegal_forward_rate": {
                    "main_mean": main_ir_mu,
                    "main_std": main_ir_sd,
                    "reverse_mean": reverse_ir_mu,
                    "reverse_std": reverse_ir_sd,
                },
                "illegal_forward_burden": {
                    "main_mean": main_ib_mu,
                    "main_std": main_ib_sd,
                    "reverse_mean": reverse_ib_mu,
                    "reverse_std": reverse_ib_sd,
                },
            },
            "per_instance": [
                {
                    "instance_id": r["instance_id"],

                    "total_steps": r["total_steps"],
                    "forward_steps": r["forward_steps"],
                    "illegal_forward_steps": r["illegal_forward_steps"],
                    "forward_density": r["forward_density"],
                    "illegal_forward_rate": r["illegal_forward_rate"],
                    "illegal_forward_burden": r["illegal_forward_burden"],

                    "total_steps_reverse": r["total_steps_reverse"],
                    "forward_steps_reverse": r["forward_steps_reverse"],
                    "illegal_forward_steps_reverse": r["illegal_forward_steps_reverse"],
                    "forward_density_reverse": r["forward_density_reverse"],
                    "illegal_forward_rate_reverse": r["illegal_forward_rate_reverse"],
                    "illegal_forward_burden_reverse": r["illegal_forward_burden_reverse"],
                }
                for r in per_instance_rows
            ],
        }
        write_json(os.path.join(results_root, "metric_2c_cdcl_only_summary.json"), summary)

        csv_path = os.path.join(results_root, "metric_2c_cdcl_only_instances.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            header = [
                "instance_id",
                "tetris_file",
                "sat_file",

                "total_steps",
                "forward_steps",
                "illegal_forward_steps",
                "forward_density",
                "illegal_forward_rate",
                "illegal_forward_burden",

                "total_steps_reverse",
                "forward_steps_reverse",
                "illegal_forward_steps_reverse",
                "forward_density_reverse",
                "illegal_forward_rate_reverse",
                "illegal_forward_burden_reverse",
            ]
            w.writerow(header)

            for r in per_instance_rows:
                row = [
                    r["instance_id"],
                    r["tetris_file"],
                    r["sat_file"],

                    r["total_steps"],
                    r["forward_steps"],
                    r["illegal_forward_steps"],
                    r["forward_density"],
                    r["illegal_forward_rate"],
                    r["illegal_forward_burden"],

                    r["total_steps_reverse"],
                    r["forward_steps_reverse"],
                    r["illegal_forward_steps_reverse"],
                    r["forward_density_reverse"],
                    r["illegal_forward_rate_reverse"],
                    r["illegal_forward_burden_reverse"],
                ]
                w.writerow(row)

        inst_ids = [r["instance_id"] for r in per_instance_rows]

        # MAIN temporal plot: burden
        plot_two_mean_lines(
            xs_temporal,
            main_ib_mu,
            label1="CDCL main vars",
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time",
            out_png=os.path.join(plots_dir, "metric_2c_burden_temporal.png"),
        )

        # REVERSE temporal plot: burden
        plot_two_mean_lines(
            xs_temporal,
            reverse_ib_mu,
            label1="CDCL reverse vars",
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time (reverse)",
            out_png=os.path.join(plots_dir, "metric_2c_burden_temporal_reverse.png"),
        )

        # MAIN instance-level plot: burden
        plot_two_instance_lines(
            inst_ids,
            main_ib_global,
            label1="CDCL main vars",
            ylabel="Global illegal forward burden",
            title="CDCL global illegal forward burden by instance",
            out_png=os.path.join(plots_dir, "metric_2c_burden_by_instance.png"),
        )

        # REVERSE instance-level plot: burden
        plot_two_instance_lines(
            inst_ids,
            reverse_ib_global,
            label1="CDCL reverse vars",
            ylabel="Global illegal forward burden",
            title="CDCL global illegal forward burden by instance (reverse)",
            out_png=os.path.join(plots_dir, "metric_2c_burden_by_instance_reverse.png"),
        )

        # MAIN mean±std bands for burden
        plot_mean_std_band(
            xs_temporal,
            main_ib_mu,
            main_ib_sd,
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time: mean ± std",
            out_png=os.path.join(plots_dir, "metric_2c_burden_mean_std.png"),
        )

        # REVERSE mean±std bands for burden
        plot_mean_std_band(
            xs_temporal,
            reverse_ib_mu,
            reverse_ib_sd,
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time: mean ± std (reverse)",
            out_png=os.path.join(plots_dir, "metric_2c_burden_mean_std_reverse.png"),
        )

    print(f"Wrote results to {results_root}")


def sat_only():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_i", type=int, help="run index i (folder run_i)")
    ap.add_argument("--tetris_root", default="/home/catalyst/project/", help="root directory for tetris traces")
    ap.add_argument("--sat_root", default="/home/catalyst/project/hmv_sat", help="root directory for sat traces")
    ap.add_argument("--W", type=int, default=3, help="number of percentile windows")
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

    results_root = os.path.join(os.getcwd(), "depth_first", f"run_{args.run_i}")
    plots_dir = os.path.join(results_root, "plots")
    ensure_dir(plots_dir)

    skipped = 0

    main_forward_density_curves: List[List[float]] = []
    main_illegal_forward_rate_curves: List[List[float]] = []
    main_illegal_forward_burden_curves: List[List[float]] = []

    reverse_forward_density_curves: List[List[float]] = []
    reverse_illegal_forward_rate_curves: List[List[float]] = []
    reverse_illegal_forward_burden_curves: List[List[float]] = []

    per_instance_rows: List[Dict[str, Any]] = []

    for j, spath, total_vars in pairs:
        sat_vars = parse_sat_full_assignment_vars(spath)

        if len(sat_vars) == 0:
            print(f"[SKIP] inst={j} sat_decisions=0")
            skipped += 1
            continue

        # MAIN = direct
        sat_main_rank = sat_vars

        # REVERSE = existing reversed form
        sat_reverse_rank = raw_vars_to_reversed_ranks(sat_vars, total_vars)

        main_global = illegal_forward_metrics_stream(sat_main_rank)
        reverse_global = illegal_forward_metrics_stream(sat_reverse_rank)

        main_windowed = windowed_illegal_forward_curves(sat_main_rank, args.W)
        reverse_windowed = windowed_illegal_forward_curves(sat_reverse_rank, args.W)

        main_forward_density_curves.append(main_windowed["forward_density"])
        main_illegal_forward_rate_curves.append(main_windowed["illegal_forward_rate"])
        main_illegal_forward_burden_curves.append(main_windowed["illegal_forward_burden"])

        reverse_forward_density_curves.append(reverse_windowed["forward_density"])
        reverse_illegal_forward_rate_curves.append(reverse_windowed["illegal_forward_rate"])
        reverse_illegal_forward_burden_curves.append(reverse_windowed["illegal_forward_burden"])

        per_instance_rows.append({
            "instance_id": j,
            "sat_file": os.path.basename(spath),

            "total_steps": main_global["total_steps"],
            "forward_steps": main_global["forward_steps"],
            "illegal_forward_steps": main_global["illegal_forward_steps"],
            "forward_density": main_global["forward_density"],
            "illegal_forward_rate": main_global["illegal_forward_rate"],
            "illegal_forward_burden": main_global["illegal_forward_burden"],

            "total_steps_reverse": reverse_global["total_steps"],
            "forward_steps_reverse": reverse_global["forward_steps"],
            "illegal_forward_steps_reverse": reverse_global["illegal_forward_steps"],
            "forward_density_reverse": reverse_global["forward_density"],
            "illegal_forward_rate_reverse": reverse_global["illegal_forward_rate"],
            "illegal_forward_burden_reverse": reverse_global["illegal_forward_burden"],
        })

    if per_instance_rows:
        W = args.W
        xs_temporal = [(k + 0.5) / W for k in range(W)]

        main_fd_mu, main_fd_sd = aggregate_curves(main_forward_density_curves, W)
        main_ir_mu, main_ir_sd = aggregate_curves(main_illegal_forward_rate_curves, W)
        main_ib_mu, main_ib_sd = aggregate_curves(main_illegal_forward_burden_curves, W)

        reverse_fd_mu, reverse_fd_sd = aggregate_curves(reverse_forward_density_curves, W)
        reverse_ir_mu, reverse_ir_sd = aggregate_curves(reverse_illegal_forward_rate_curves, W)
        reverse_ib_mu, reverse_ib_sd = aggregate_curves(reverse_illegal_forward_burden_curves, W)

        main_fd_global = [r["forward_density"] for r in per_instance_rows]
        main_ir_global = [r["illegal_forward_rate"] for r in per_instance_rows]
        main_ib_global = [r["illegal_forward_burden"] for r in per_instance_rows]

        reverse_fd_global = [r["forward_density_reverse"] for r in per_instance_rows]
        reverse_ir_global = [r["illegal_forward_rate_reverse"] for r in per_instance_rows]
        reverse_ib_global = [r["illegal_forward_burden_reverse"] for r in per_instance_rows]

        summary = {
            "run_i": args.run_i,
            "metric": "metric_2c_cdcl_only_illegal_forward_consistency",
            "W": args.W,
            "pairs_used": len(per_instance_rows),
            "skipped": skipped,
            "global_instance_metrics": {
                "forward_density": {
                    "main_mean": mean(main_fd_global),
                    "main_std": std(main_fd_global),
                    "reverse_mean": mean(reverse_fd_global),
                    "reverse_std": std(reverse_fd_global),
                },
                "illegal_forward_rate": {
                    "main_mean": mean(main_ir_global),
                    "main_std": std(main_ir_global),
                    "reverse_mean": mean(reverse_ir_global),
                    "reverse_std": std(reverse_ir_global),
                },
                "illegal_forward_burden": {
                    "main_mean": mean(main_ib_global),
                    "main_std": std(main_ib_global),
                    "reverse_mean": mean(reverse_ib_global),
                    "reverse_std": std(reverse_ib_global),
                },
            },
            "temporal_window_metrics": {
                "forward_density": {
                    "main_mean": main_fd_mu,
                    "main_std": main_fd_sd,
                    "reverse_mean": reverse_fd_mu,
                    "reverse_std": reverse_fd_sd,
                },
                "illegal_forward_rate": {
                    "main_mean": main_ir_mu,
                    "main_std": main_ir_sd,
                    "reverse_mean": reverse_ir_mu,
                    "reverse_std": reverse_ir_sd,
                },
                "illegal_forward_burden": {
                    "main_mean": main_ib_mu,
                    "main_std": main_ib_sd,
                    "reverse_mean": reverse_ib_mu,
                    "reverse_std": reverse_ib_sd,
                },
            },
            "per_instance": [
                {
                    "instance_id": r["instance_id"],
                    "total_steps": r["total_steps"],
                    "forward_steps": r["forward_steps"],
                    "illegal_forward_steps": r["illegal_forward_steps"],
                    "forward_density": r["forward_density"],
                    "illegal_forward_rate": r["illegal_forward_rate"],
                    "illegal_forward_burden": r["illegal_forward_burden"],

                    "total_steps_reverse": r["total_steps_reverse"],
                    "forward_steps_reverse": r["forward_steps_reverse"],
                    "illegal_forward_steps_reverse": r["illegal_forward_steps_reverse"],
                    "forward_density_reverse": r["forward_density_reverse"],
                    "illegal_forward_rate_reverse": r["illegal_forward_rate_reverse"],
                    "illegal_forward_burden_reverse": r["illegal_forward_burden_reverse"],
                }
                for r in per_instance_rows
            ],
        }

        write_json(os.path.join(results_root, "metric_2c_cdcl_only_summary.json"), summary)

        csv_path = os.path.join(results_root, "metric_2c_cdcl_only_instances.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = [
                "instance_id",
                "sat_file",

                "total_steps",
                "forward_steps",
                "illegal_forward_steps",
                "forward_density",
                "illegal_forward_rate",
                "illegal_forward_burden",

                "total_steps_reverse",
                "forward_steps_reverse",
                "illegal_forward_steps_reverse",
                "forward_density_reverse",
                "illegal_forward_rate_reverse",
                "illegal_forward_burden_reverse",
            ]
            w.writerow(header)

            for r in per_instance_rows:
                row = [
                    r["instance_id"],
                    r["sat_file"],

                    r["total_steps"],
                    r["forward_steps"],
                    r["illegal_forward_steps"],
                    r["forward_density"],
                    r["illegal_forward_rate"],
                    r["illegal_forward_burden"],

                    r["total_steps_reverse"],
                    r["forward_steps_reverse"],
                    r["illegal_forward_steps_reverse"],
                    r["forward_density_reverse"],
                    r["illegal_forward_rate_reverse"],
                    r["illegal_forward_burden_reverse"],
                ]
                w.writerow(row)

        inst_ids = [r["instance_id"] for r in per_instance_rows]

        plot_two_mean_lines(
            xs_temporal,
            main_ib_mu,
            label1="CDCL main vars",
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time",
            out_png=os.path.join(plots_dir, "metric_2c_burden_temporal.png"),
        )

        plot_two_mean_lines(
            xs_temporal,
            reverse_ib_mu,
            label1="CDCL reverse vars",
            ylabel="Illegal forward burden",
            title="CDCL illegal forward burden over time (reverse)",
            out_png=os.path.join(plots_dir, "metric_2c_burden_temporal_reverse.png"),
        )

        plot_two_instance_lines(
            inst_ids,
            main_ib_global,
            label1="CDCL main vars",
            ylabel="Global illegal forward burden",
            title="CDCL global illegal forward burden by instance",
            out_png=os.path.join(plots_dir, "metric_2c_burden_by_instance.png"),
        )

        plot_two_instance_lines(
            inst_ids,
            reverse_ib_global,
            label1="CDCL reverse vars",
            ylabel="Global illegal forward burden",
            title="CDCL global illegal forward burden by instance (reverse)",
            out_png=os.path.join(plots_dir, "metric_2c_burden_by_instance_reverse.png"),
        )

        

    print(f"Wrote results to {results_root}")

if __name__ == "__main__":
    main()