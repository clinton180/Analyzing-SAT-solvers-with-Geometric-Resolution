import subprocess, sys
LINUX_SCRIPT = "/home/catalyst/SAT/sat_test.py"
args = sys.argv[1:]
cmd = ["wsl", "python3", LINUX_SCRIPT, *args]
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr, file=sys.stderr)
sys.exit(result.returncode)


import json
from collections import Counter, defaultdict
import math

def load_events(path):
    evs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evs.append(json.loads(line))
    return evs

def summarize(evs):
    counts = Counter(e["event"] for e in evs)
    outputs = counts.get("OUTPUT", 0)

    # KB sizes: try to infer initial and final
    kb_sizes = [e.get("kb_size_after") for e in evs if e["event"] == "KB_SIZE" and e.get("kb_size_after") is not None]
    kb_init = kb_sizes[0] if kb_sizes else None
    kb_final = kb_sizes[-1] if kb_sizes else None

    # cover hits
    cover_total = counts.get("QUERY_COVER", 0)
    cover_hits = sum(1 for e in evs if e["event"] == "QUERY_COVER" and e.get("result") is not None)
    cover_hit_rate = (cover_hits / cover_total) if cover_total else None

    # depth stats by event
    depth_by_event = defaultdict(list)
    for e in evs:
        d = e.get("depth")
        if d is not None:
            depth_by_event[e["event"]].append(d)

    def depth_stats(xs):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        mean = sum(xs_sorted) / n
        p90 = xs_sorted[math.floor(0.9 * (n - 1))]
        return {"n": n, "max": xs_sorted[-1], "mean": mean, "p90": p90}

    depth_summary = {k: depth_stats(v) for k, v in depth_by_event.items()}

    # resolve dim histogram
    resolve_dims = Counter(e.get("dim") for e in evs if e["event"] == "RESOLVE")
    # per-output episode lengths
    episode_lengths = []
    last_out_idx = None
    out_indices = [i for i, e in enumerate(evs) if e["event"] == "OUTPUT"]
    for idx in out_indices:
        if last_out_idx is None:
            episode_lengths.append(idx + 1)
        else:
            episode_lengths.append(idx - last_out_idx)
        last_out_idx = idx

    def basic_list_stats(xs):
        if not xs:
            return None
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        mean = sum(xs_sorted) / n
        p90 = xs_sorted[math.floor(0.9 * (n - 1))]
        return {"n": n, "min": xs_sorted[0], "max": xs_sorted[-1], "mean": mean, "p90": p90}

    episode_stats = basic_list_stats(episode_lengths)

    # normalized ratios
    def per_output(x):
        return (x / outputs) if outputs else None

    ratios = {
        "splits_per_output": per_output(counts.get("SPLIT", 0)),
        "queries_per_output": per_output(counts.get("QUERY_COVER", 0)),
        "resolves_per_output": per_output(counts.get("RESOLVE", 0)),
        "adds_per_output": per_output(counts.get("ADD_BOX", 0)),
        "uncov_per_output": per_output(counts.get("UNCOVERED_POINT", 0)),
        "oracle_per_output": per_output(counts.get("ORACLE", 0)),
    }

    kb_adds_per_output = None
    if outputs and kb_init is not None and kb_final is not None:
        kb_adds_per_output = (kb_final - kb_init) / outputs

    meta = evs[0] if evs and evs[0].get("event") == "START" else {}

    return {
        "meta": {
            "n": meta.get("n"),
            "widths": meta.get("widths"),
            "sao": meta.get("sao"),
            "B_size": meta.get("B_size"),
            "halt_first": meta.get("halt_first"),
        },
        "counts": dict(counts),
        "outputs": outputs,
        "kb_init": kb_init,
        "kb_final": kb_final,
        "kb_adds_per_output": kb_adds_per_output,
        "cover_hit_rate": cover_hit_rate,
        "ratios": ratios,
        "resolve_dims": dict(resolve_dims),
        "episode_stats": episode_stats,
        "depth_summary": depth_summary,
    }

def compare(a, b):
    # Compare only the “screening” metrics; you can extend.
    keys = [
        ("outputs",),
        ("kb_adds_per_output",),
        ("cover_hit_rate",),
        ("ratios", "splits_per_output"),
        ("ratios", "queries_per_output"),
        ("ratios", "resolves_per_output"),
        ("episode_stats", "mean"),
        ("episode_stats", "p90"),
    ]
    out = {}
    for path in keys:
        def get(d):
            for k in path:
                if d is None:
                    return None
                d = d.get(k)
            return d
        va, vb = get(a), get(b)
        out[".".join(path)] = {"A": va, "B": vb}
    return out

if __name__ == "__main__":
    A = summarize(load_events("runA.jsonl"))
    B = summarize(load_events("runB.jsonl"))
    print("A meta:", A["meta"])
    print("B meta:", B["meta"])
    print(json.dumps(compare(A, B), indent=2))
    print("\nA resolve dims:", A["resolve_dims"])
    print("B resolve dims:", B["resolve_dims"])
    print("\nA depth summary (QUERY_COVER/SPLIT/RESOLVE):",
          {k: A["depth_summary"].get(k) for k in ["QUERY_COVER","SPLIT","RESOLVE","UNCOVERED_POINT"]})
    print("B depth summary (QUERY_COVER/SPLIT/RESOLVE):",
          {k: B["depth_summary"].get(k) for k in ["QUERY_COVER","SPLIT","RESOLVE","UNCOVERED_POINT"]})
