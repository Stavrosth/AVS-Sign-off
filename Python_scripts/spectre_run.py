"""
spectre_run.py

Author: Stavros Stathoudakis 
Mentor: Akis Tsoumanis  
Supervising Professor: Christos Sotiriou  
Inspired by Christian Lütkemeyer

This script recursively runs Spectre simulations on .scs files found in a directory structure.
It organizes the results to mirror the input structure, extracting measurement values from .measure files
and compiling them into CSV files for easy analysis down the road.

-- IT IS NECESSARY TO HAVE A SPECTRE LICENSE TO RUN THIS SCRIPT --
"""

import argparse, csv, os, re, subprocess, sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Match floating point numbers, including scientific notation
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# .measure file parsing
MEAS_NAME_LINE = re.compile(r"^\s*Measurement\s+Name\s*:\s*(\S+)", re.IGNORECASE)
MEAS_BLOCK_NAME_NUM = re.compile(r"(?i)\b(?:transient|tran)\D*?(\d+)\b")

def build_meas_value_re(name: str):
    # Match:
    #   delay_out8_out6 = 1.23e-10
    #   delay_out8_ou6 1.23e-10
    #   delay_out8_ou6 temp @ 50 = 1.23e-10
    #   delay_out8_ou6 temp @ 50  1.23e-10
    return re.compile(
        rf"""^\s*
            {re.escape(name)}
            (?:\s+temp\s*@\s*{NUM})?   # optional 'temp @ 50'
            \s*(?:[:=]|\s)\s*
            ({NUM})\s*$                # capture the numeric value
        """,
        re.IGNORECASE | re.VERBOSE
    )

def parse_measure_file(meas_path: Path, meas_name: str) -> List[Dict]:
    """Return a list of dicts: {'block_name','block_num','value_s'} in file order.

    This version is robust to Spectre outputs that include a "Swept Measurements"
    section (i.e., temp-tagged values without a preceding 'Measurement Name').
    """
    rows: List[Dict] = []
    if not meas_path.exists():
        return rows
    lines = meas_path.read_text(errors="ignore").splitlines()
    val_re = build_meas_value_re(meas_name)

    current_block: Optional[Dict] = None
    next_block_num = 1  # used for synthetic blocks or when no index is present

    def new_block(name: Optional[str] = None, num: Optional[int] = None) -> Dict:
        nonlocal next_block_num
        bname = name or f"synthetic{next_block_num}"
        bnum = None
        if isinstance(num, int):
            bnum = num
        else:
            mnum = MEAS_BLOCK_NAME_NUM.search(bname)
            if mnum:
                try:
                    bnum = int(mnum.group(1))
                except Exception:
                    bnum = None
        if bnum is None:
            bnum = next_block_num
        next_block_num = bnum + 1
        return {"block_name": bname, "block_num": bnum}

    for ln in lines:
        m_name = MEAS_NAME_LINE.search(ln)
        if m_name:
            # Flush the previous block if it got a value
            if current_block and ("value_s" in current_block):
                rows.append(current_block)
            # Start a new named block; infer/assign its block_num
            bname = m_name.group(1)
            current_block = new_block(bname)
            continue

        # Look for the measurement value line (may be in "Swept Measurements")
        m_val = val_re.search(ln)
        if m_val:
            # If there's no current block or it already has a value,
            # start a synthetic "next" block so we don't overwrite.
            if (current_block is None) or ("value_s" in current_block):
                current_block = new_block()
            try:
                current_block["value_s"] = float(m_val.group(1))
            except Exception:
                pass
            # We can safely append now because a block holds exactly one value
            rows.append(current_block)
            current_block = None

    # Final flush if needed
    if current_block and ("value_s" in current_block):
        rows.append(current_block)

    # Sort by block number to ensure proper 1->base, 2->alter#1, ...
    rows.sort(key=lambda r: (r.get("block_num") if isinstance(r.get("block_num"), int) else 1_000_000))
    return rows

# ------------------------
# .scs contexts parsing
# ------------------------
ALTER_LINE = re.compile(r"^\s*\.ALTER\b", re.IGNORECASE)
TEMP_LINE  = re.compile(r"^\s*\.TEMP\s+({})\s*$".format(NUM), re.IGNORECASE)
VVDVAL_LINE = re.compile(r"(?i)^\s*\.param\s+vvdval\s*=\s*({})\s*$".format(NUM), re.IGNORECASE)

def parse_contexts_from_scs(scs_path: Path) -> Tuple[Dict, List[Dict]]:
    """Parse base (pre-.ALTER) context and a list of alter contexts in order.
    Returns (base_ctx, alter_ctxs). Each ctx has keys: temp, vvdval, alter_index (0 for base, 1..N for alters)
    """
    base = {"temp": None, "vvdval": None, "alter_index": 0}
    alter_ctxs: List[Dict] = []

    if not scs_path.exists():
        return base, alter_ctxs

    lines = scs_path.read_text(errors="ignore").splitlines()

    # Pass 1: collect base values up to first .ALTER
    for ln in lines:
        if ALTER_LINE.search(ln):
            break
        m_t = TEMP_LINE.search(ln)
        if m_t:
            try:
                base["temp"] = float(m_t.group(1))
            except Exception:
                pass
        m_v = VVDVAL_LINE.search(ln)
        if m_v:
            try:
                base["vvdval"] = float(m_v.group(1))
            except Exception:
                pass

    # Pass 2: build one context per .ALTER (inherit previous context, starting from base)
    current: Optional[Dict] = None
    in_alter = False
    for ln in lines:
        if ALTER_LINE.search(ln):
            if in_alter and current is not None:
                alter_ctxs.append({"temp": current.get("temp"), "vvdval": current.get("vvdval")})
            parent = current.copy() if current else base.copy()
            current = parent
            in_alter = True
            continue

        if not in_alter:
            continue

        m_t = TEMP_LINE.search(ln)
        if m_t:
            try:
                current["temp"] = float(m_t.group(1))
            except Exception:
                pass
            continue

        m_v = VVDVAL_LINE.search(ln)
        if m_v:
            try:
                current["vvdval"] = float(m_v.group(1))
            except Exception:
                pass
            continue

    if in_alter and current is not None:
        alter_ctxs.append({"temp": current.get("temp"), "vvdval": current.get("vvdval")})

    for i, ctx in enumerate(alter_ctxs, start=1):
        ctx["alter_index"] = i
    return base, alter_ctxs

# ------------------------
# Spectre runner + glue
# ------------------------
def bash(cmd: str, cwd: Optional[Path] = None):
    return subprocess.run(["bash","-lc", cmd], cwd=str(cwd) if cwd else None,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, env=os.environ.copy())

def run_spectre(scs: Path, extra_args: Optional[List[str]]) -> Tuple[int, Path]:
    cmd = "spectre " + " ".join([f'"{scs.name}"'] + (extra_args or []))
    print(f"\n[RUN] cwd=\"{scs.parent}\" cmd: {cmd}")
    proc = bash(cmd, cwd=scs.parent)
    log = scs.with_suffix(".spectre.log")
    try:
        log.write_text(proc.stdout)
    except Exception:
        pass
    return proc.returncode, log

PROC_RE = re.compile(r"_(tt|ss|ff)\.scs?$", re.IGNORECASE)

def process_from_name(p: Path) -> str:
    m = PROC_RE.search(p.name)
    return (m.group(1).lower() if m else "")

def find_measure_file(scs: Path) -> Optional[Path]:
    cand = scs.with_suffix(".measure")
    if cand.exists():
        return cand
    else:
        print(f"Warning: .measure file not found for {scs}", file=sys.stderr)
        return None

def discover_runs(current_dir: Path, glob: str, results_root: Path, top_level_dir: Path, out_csv_name: str) -> List[Dict]:
    """
    Recursively find directories containing .scs files and define simulation runs.
    """
    runs = []
    
    # 1. Check for .scs files in the current directory
    scs_files_here = sorted(current_dir.glob(glob))
    if scs_files_here:
        # This is a "leaf" directory with simulations. Define a run.
        relative_path = current_dir.relative_to(top_level_dir)
        output_dir = results_root / relative_path
        
        # Replace the first part of the path name (e.g., 'corners' -> 'results')
        # This makes paths like 'corners/pin1' become 'results/pin1'
        parts = list(output_dir.parts)
        if parts:
             parts[0] = parts[0].replace("corners", "results", 1)
             output_dir = Path(*parts)
        
        runs.append({
            "scs_files": scs_files_here,
            "output_csv": output_dir / out_csv_name
        })
    else:
        # 2. If no files, recurse into subdirectories
        for subdir in sorted(d for d in current_dir.iterdir() if d.is_dir()):
            runs.extend(discover_runs(subdir, glob, results_root, top_level_dir, out_csv_name))
            
    return runs

def main():
    ap = argparse.ArgumentParser(
        description="Runs Spectre recursively on .scs files found in a directory structure, organizing results to mirror the input structure.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("path", help="A single .scs file or a directory to search recursively for .scs files (e.g., 'corners').")
    ap.add_argument("--glob", default="*.scs", help="Glob pattern for simulation files (default: *.scs)")
    ap.add_argument("--meas", default="delay_out8_out6", help="Measurement name inside .measure file (default: delay_out8_ou6)")
    ap.add_argument("--out-csv", default="measurements.csv", help="The filename for the output CSV(s) (default: measurements.csv)")
    ap.add_argument("--spectre-args", nargs=argparse.REMAINDER, help="Args after -- are passed to Spectre")
    args = ap.parse_args()

    ipath = Path(args.path).resolve()
    if not ipath.exists():
        print(f"Error: path not found: {ipath}", file=sys.stderr); sys.exit(2)

    # --- New logic to find run definitions ---
    run_definitions = []
    if ipath.is_file():
        # Handle single file case: place results in a 'results' subdir relative to the file
        results_dir = ipath.parent / "results"
        run_definitions.append({
            "scs_files": [ipath],
            "output_csv": results_dir / args.out_csv
        })
    elif ipath.is_dir():
        # Recursively search for directories containing .scs files
        print(f"Scanning for simulation sets in '{ipath}'...")
        # The top-level results dir will be a sibling to the input dir
        # e.g., if input is './corners', results will be in './results'
        base_results_dir = ipath.parent / ipath.name.replace("corners", "results", 1)
        run_definitions = discover_runs(ipath, args.glob, base_results_dir, ipath, args.out_csv)

    if not run_definitions:
        print(f"No .scs files found matching '{args.glob}' in {ipath} or its subdirectories.", file=sys.stderr)
        sys.exit(3)

    # --- Main processing loop over each defined run ---
    for run_def in run_definitions:
        all_rows: List[Dict] = []
        scs_files = run_def["scs_files"]
        out_csv_path = run_def["output_csv"]

        print(f"\n--- Starting run for output: {out_csv_path} ---")

        for scs in scs_files:
            rc, log = run_spectre(scs, args.spectre_args or [])
            print(f"[{'OK' if rc==0 else 'FAIL'}] {scs.name} -> log={log.name}")

            # Parse contexts (this defines ALL temps that MUST be written)
            base_ctx, alter_ctxs = parse_contexts_from_scs(scs)
            contexts_full = [base_ctx] + alter_ctxs  # always write one row per context

            # Try to read .measure values (optional)
            meas_rows = []
            meas_path = find_measure_file(scs)
            if meas_path:
                meas_rows = parse_measure_file(meas_path, args.meas)

            # Create a value list aligned by index: 0->base, 1->alter#1, ...
            values_ps: List[Optional[float]] = [None] * len(contexts_full)
            # If block numbers exist, map 1->index0, 2->index1, ...
            if any(r.get("block_num") is not None for r in meas_rows):
                for r in meas_rows:
                    b = r.get("block_num")
                    if isinstance(b, int) and b >= 1:
                        idx = b - 1
                        if idx < len(values_ps):
                            val = r.get("value_s")
                            values_ps[idx] = (val * 1e12) if isinstance(val, (float, int)) else None
            else:
                # Positional fallback: first value -> base, next -> alter#1, etc.
                for i, r in enumerate(meas_rows):
                    if i < len(values_ps):
                        val = r.get("value_s")
                        values_ps[i] = (val * 1e12) if isinstance(val, (float, int)) else None

            section = process_from_name(scs)  # tt/ss/ff from filename

            # ALWAYS emit one row per context (even if delay is missing)
            for i, ctx in enumerate(contexts_full):
                row = {
                    "file": scs.name,
                    "dc": ctx.get("vvdval"),
                    "section": section,
                    "temp": ctx.get("temp"),
                    "delay_ps": values_ps[i] if i < len(values_ps) else None,
                }
                all_rows.append(row)

        if not all_rows:
            print(f"No contexts found for this run. Not writing {out_csv_path}.", file=sys.stderr)
            continue # Skip to the next run definition

        # Create the parent directory if it doesn't exist
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Write ONLY the requested headers, in that exact order
        fieldnames = ["file", "dc", "section", "temp", "delay_ps"]

        with open(out_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            # Normalize None -> empty field to keep CSV tidy
            for r in all_rows:
                rr = {k: ("" if r[k] is None else r[k]) for k in fieldnames}
                w.writerow(rr)

        print(f"CSV saved: {out_csv_path} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()