"""perf_utils.py — Unified performance recorder shared by all pipeline steps.

Purpose: provide same-hardware, same-protocol measurements for the paper's
performance comparison table: end-to-end wall time + peak memory (peak RSS).

Usage A (context manager, for scripts with a main() function):
    from perf_utils import PerfRecorder
    with PerfRecorder("step5-2_deepfusion"):
        main()

Usage B (one-line start, auto-save via atexit; minimal-intrusion insertion):
    from perf_utils import PerfRecorder
    PerfRecorder.start_now("step6_scib")   # nothing else needed afterwards

Output: <project_root>/enhanced_results/perf_report.jsonl (one JSON record per line, appended)
Fields: task / status / wall_s / peak_rss_mb / start / end / argv / python / host
Summary: python perf_utils.py            -> print a Markdown summary table (sorted by start time)
         python perf_utils.py --csv x.csv -> additionally export CSV
"""
import atexit
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "enhanced_results" / "perf_report.jsonl"
_IS_MAC = platform.system() == "Darwin"


def _peak_rss_mb() -> float:
    """Peak RSS of this process (MB). ru_maxrss is in bytes on macOS, KB on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if _IS_MAC else raw / 1e3


_MACHINE_CACHE = None


def _machine_info() -> str:
    """Static hardware/OS description for the performance table (cached).
    e.g. 'Apple M2 Ultra | 24 cores | 192GB RAM | Darwin 24.x (arm64)'."""
    global _MACHINE_CACHE
    if _MACHINE_CACHE is None:
        def _sysctl(key):
            try:
                return subprocess.run(['sysctl', '-n', key],
                                      capture_output=True, text=True, timeout=5).stdout.strip()
            except Exception:
                return ''
        chip = (_sysctl('machdep.cpu.brand_string') or _sysctl('hw.model')
                or platform.processor() or platform.machine())
        try:
            mem = f"{int(_sysctl('hw.memsize')) / 1e9:.0f}GB RAM"
        except Exception:
            mem = "?GB RAM"
        _MACHINE_CACHE = (f"{chip} | {os.cpu_count()} cores | {mem} | "
                          f"{platform.system()} {platform.release()} ({platform.machine()})")
    return _MACHINE_CACHE


class PerfRecorder:
    """Record wall time and peak memory of one task; write one JSON line on exit."""

    def __init__(self, task: str, out=None, **extra):
        self.task = task
        self.out = Path(out) if out else DEFAULT_OUT
        self.extra = extra
        self.status = "ok"
        self.t0 = None
        self.start = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.start = datetime.now().isoformat(timespec="seconds")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.status = f"error: {exc_type.__name__}"
        self.write()
        return False  # do not swallow exceptions

    def write(self):
        if self.t0 is None:
            return
        rec = {
            "task": self.task,
            "status": self.status,
            "wall_s": round(time.perf_counter() - self.t0, 2),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
            "start": self.start,
            "end": datetime.now().isoformat(timespec="seconds"),
            "argv": " ".join(sys.argv),
            "python": sys.executable,
            "host": platform.node(),
            "machine": _machine_info(),
        }
        rec.update(self.extra)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[perf] {self.task}: {rec['wall_s']}s, "
              f"peak {rec['peak_rss_mb']}MB -> {self.out}")
        self.t0 = None  # prevent double-writing via both atexit and __exit__

    @classmethod
    def start_now(cls, task: str, out=None, **extra) -> "PerfRecorder":
        """Start timing immediately; registers atexit/excepthook so the record
        is written automatically when the script finishes."""
        rec = cls(task, out=out, **extra)
        rec.__enter__()

        def _hook(exc_type, exc, tb):
            rec.status = f"error: {exc_type.__name__}"
            sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = _hook
        atexit.register(rec.write)
        return rec


def summary(out=DEFAULT_OUT, csv_path=None):
    """Read the jsonl and print a Markdown summary table (usable directly in the paper)."""
    out = Path(out)
    if not out.exists():
        print(f"No record file: {out}")
        return
    recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    recs.sort(key=lambda r: r.get("start", ""))
    header = "| task | status | wall_s | wall_min | peak_rss_mb | start |"
    sep = "|---|---|---|---|---|---|"
    print(header)
    print(sep)
    rows = []
    for r in recs:
        row = (f"| {r['task']} | {r['status']} | {r['wall_s']} "
               f"| {round(r['wall_s'] / 60, 1)} | {r['peak_rss_mb']} | {r['start']} |")
        print(row)
        rows.append(r)
    if csv_path:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV exported: {csv_path}")


if __name__ == "__main__":
    csv_path = None
    if "--csv" in sys.argv:
        i = sys.argv.index("--csv")
        csv_path = sys.argv[i + 1]
    summary(csv_path=csv_path)
