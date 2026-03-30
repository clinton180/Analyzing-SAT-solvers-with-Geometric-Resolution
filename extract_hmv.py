import os
import json
import time
import subprocess
from pathlib import Path


SRC_ROOT = Path("HMV").resolve()
DST_ROOT = Path("HMV_cnf").resolve()
LOG_PATH = Path("hmv_conversion_log.jsonl").resolve()

VALID_EXTS = {".aig", ".aag"}

PER_FILE_TIMEOUT_SEC = 300          
TOTAL_RUNTIME_LIMIT_SEC = None      

def safe_remove(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def convert_with_abc(src_file: Path, dst_file: Path, timeout_sec: int | None = 300) -> tuple[bool, str, float]:
    """
    Convert one .aig/.aag file to .cnf using ABC.

    Returns:
        (success, message, elapsed_seconds)
    """
    start = time.time()

    cmd = [
        "abc",
        "-c",
        f'read_aiger "{src_file}"; write_cnf "{dst_file}"'
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec
        )
        elapsed = time.time() - start

        if result.returncode == 0 and dst_file.exists():
            return True, result.stdout.strip(), elapsed

        # failed normal exit -> remove partial output if any
        safe_remove(dst_file)

        msg = (
            f"ABC failed with return code {result.returncode}. "
            f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
        )
        return False, msg, elapsed

    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start

        # timed out -> remove partial output if any
        safe_remove(dst_file)

        msg = (
            f"Timeout after {timeout_sec} seconds. "
            f"stdout={getattr(e, 'stdout', '')} stderr={getattr(e, 'stderr', '')}"
        )
        return False, msg, elapsed

    except Exception as e:
        elapsed = time.time() - start

        # unexpected exception -> remove partial output if any
        safe_remove(dst_file)

        return False, f"Exception during conversion: {repr(e)}", elapsed


def log_jsonl(record: dict) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root does not exist: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {
        "scanned_files": 0,
        "candidate_aig_files": 0,
        "converted_ok": 0,
        "failed": 0,
        "skipped_existing": 0,
        "stopped_due_to_global_timeout": False,
    }

    start_all = time.time()

    for current_dir, _, files in os.walk(SRC_ROOT):
        current_dir = Path(current_dir)

        rel_dir = current_dir.relative_to(SRC_ROOT)
        out_dir = DST_ROOT / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            # ---- global runtime bound check ----
            elapsed_total = time.time() - start_all
            if (
                TOTAL_RUNTIME_LIMIT_SEC is not None
                and elapsed_total >= TOTAL_RUNTIME_LIMIT_SEC
            ):
                summary["stopped_due_to_global_timeout"] = True

                log_jsonl({
                    "status": "global_timeout_stop",
                    "elapsed_total_sec": elapsed_total,
                    "limit_sec": TOTAL_RUNTIME_LIMIT_SEC,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

                final_record = {
                    "status": "summary",
                    "source_root": str(SRC_ROOT),
                    "dest_root": str(DST_ROOT),
                    "total_elapsed_sec": elapsed_total,
                    **summary,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                log_jsonl(final_record)

                print("Stopped due to global runtime limit.")
                print(json.dumps(final_record, indent=2))
                return

            summary["scanned_files"] += 1

            src_file = current_dir / name
            ext = src_file.suffix.lower()

            if ext not in VALID_EXTS:
                continue

            summary["candidate_aig_files"] += 1
            
            dst_file = out_dir / (src_file.stem + ".cnf")

            if dst_file.exists():
                summary["skipped_existing"] += 1
                log_jsonl({
                    "status": "skipped_existing",
                    "src": str(src_file),
                    "dst": str(dst_file),
                    "reason": "destination already exists",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                continue

            ok, message, elapsed = convert_with_abc(
                src_file,
                dst_file,
                timeout_sec=PER_FILE_TIMEOUT_SEC
            )

            if ok:
                summary["converted_ok"] += 1
                log_jsonl({
                    "status": "ok",
                    "src": str(src_file),
                    "dst": str(dst_file),
                    "elapsed_sec": elapsed,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[{summary['converted_ok']}] {src_file}")
            else:
                summary["failed"] += 1
                log_jsonl({
                    "status": "failed",
                    "src": str(src_file),
                    "dst": str(dst_file),
                    "elapsed_sec": elapsed,
                    "error": message,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })

    total_elapsed = time.time() - start_all

    final_record = {
        "status": "summary",
        "source_root": str(SRC_ROOT),
        "dest_root": str(DST_ROOT),
        "total_elapsed_sec": total_elapsed,
        **summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    log_jsonl(final_record)

    print("Done.")
    print(json.dumps(final_record, indent=2))

if __name__ == "__main__":
    main()