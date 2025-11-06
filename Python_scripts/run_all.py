"""
run_all.py

Author: Stavros Stathoudakis 
Mentor: Akis Tsoumanis  
Supervising Professor: Christos Sotiriou  
Inspired by Christian Lütkemeyer

A master script to orchestrate the entire simulation and analysis workflow.
This script provides a single entry point to run the three stages of the process:
1.  Corner Generation (`create_corners.py`): Creates the necessary .scs files for all corners.
2.  Spectre Simulation (`spectre_run.py`): Runs Spectre on all generated corner files.
3.  Polynomial Fitting (`poly_fit.py`): Analyzes the simulation results and generates predictive models.

It takes a directory of base .scs files as input and handles the creation
of 'corners' and 'results' directories, passing the correct paths between scripts.
The new --cell_name argument is now correctly passed to poly_fit.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# --- Get the absolute path to the directory containing this script ---
# This makes the script runnable from any directory.
SCRIPT_DIR = Path(__file__).resolve().parent

# --- Configuration: Script names (now with absolute paths) ---
CREATE_CORNERS_SCRIPT = str(SCRIPT_DIR / "create_corners.py")
SPECTRE_RUN_SCRIPT = str(SCRIPT_DIR / "spectre_run.py")
POLY_FIT_SCRIPT = str(SCRIPT_DIR / "poly_fit.py")

def run_command(command: list, stage_name: str):
    """
    Executes a command in a subprocess, prints its output in real-time,
    and checks for errors.
    """
    print(f"\n{'='*20} STAGE: {stage_name.upper()} {'='*20}")
    print(f"Executing command: {' '.join(command)}")
    print("-" * (49 + len(stage_name)))

    try:
        # Using subprocess.Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Read and print output line by line
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"\n--- ERROR ---")
            print(f"Stage '{stage_name}' failed with exit code {return_code}.")
            print(f"Command was: {' '.join(command)}")
            sys.exit(1) # Exit the master script on failure

        print(f"--- Stage '{stage_name}' completed successfully. ---")

    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Error: The script for stage '{stage_name}' was not found.")
        print(f"Please ensure '{command[1]}' exists and is executable.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"An unexpected error occurred during stage '{stage_name}': {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full [corners -> spectre -> analysis] workflow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing the base .scs netlist files.")

    # --- Arguments for create_corners.py ---
    corners_group = parser.add_argument_group('Corner Generation Options (for create_corners.py)')
    corners_group.add_argument("--pins", type=int, help="Number of input pins for multi-pin analysis.")
    corners_group.add_argument("--pin-names", type=str, help="Comma-separated list of logical pin names (e.g., \"A,B\")")
    corners_group.add_argument(
        "--cell-map",
        type=str,
        help="Mapping from .scs cell names to .lib cell names. Format: \"SCS_NAME_1:LIB_NAME_1,SCS_NAME_2:LIB_NAME_2\""
    )
    corners_group.add_argument(
        "--base-lib-file",
        type=str,
        default="../sg13g2_stdcell_typ_1p20V_25C.lib",
        help="Path to the *exact* .lib file to use for capacitance parsing."
    )
    corners_group.add_argument("--vstep", type=str, default="0.1", help="Voltage step for sweep mode.")

    # --- Arguments for spectre_run.py ---
    spectre_group = parser.add_argument_group('Spectre Simulation Options (for spectre_run.py)')

    # --- Arguments for poly_fit.py (MODIFIED SECTION) ---
    fit_group = parser.add_argument_group('Fitting Options (for poly_fit.py)')
    fit_group.add_argument('--poly_degree', type=int, default=3, help='Polynomial degree for the residual model.')
    fit_group.add_argument('--cell_name', type=str, default='Unknown_Cell', help='The name of the cell being tested for plotting.')

    args = parser.parse_args()

    # --- Define Directory Structure ---
    base_dir = args.input_dir.resolve()
    corners_dir = base_dir / "corners"
    results_dir = base_dir / "results"

    if not base_dir.is_dir():
        print(f"Error: Input path '{base_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
        
    scs_files = list(base_dir.glob("*.scs"))
    if not scs_files:
        print(f"Error: No .scs files found in '{base_dir}'.", file=sys.stderr)
        sys.exit(1)

    # =========================================================================
    # STAGE 1: CREATE CORNERS
    # =========================================================================
    # We run this for each .scs file found in the input directory
    for scs_file in scs_files:
        # We now explicitly call the script using the same Python interpreter
        # that is running this master script. This is more robust.
        cmd_corners = [
            sys.executable,
            CREATE_CORNERS_SCRIPT,
            str(scs_file),
            str(corners_dir)
        ]
        if args.pins:
            cmd_corners.extend(["--pins", str(args.pins)])
        if args.vstep:
            cmd_corners.extend(["--vstep", args.vstep])
            
        if args.pin_names:
            cmd_corners.extend(["--pin-names", args.pin_names])
        if args.cell_map:
            cmd_corners.extend(["--cell-map", args.cell_map])
        if args.base_lib_file:
            cmd_corners.extend(["--base-lib-file", args.base_lib_file])

        run_command(cmd_corners, f"Create Corners for {scs_file.name}")


    # =========================================================================
    # STAGE 2: RUN SPECTRE
    # =========================================================================
    cmd_spectre = [
        sys.executable,
        SPECTRE_RUN_SCRIPT,
        str(corners_dir)
    ]
    run_command(cmd_spectre, "Run Spectre Simulations")


    # =========================================================================
    # STAGE 3: POLYNOMIAL FITTING & ANALYSIS (MODIFIED SECTION)
    # =========================================================================
    cmd_fit = [
        sys.executable,
        POLY_FIT_SCRIPT,
        str(results_dir),
        "--poly_degree", str(args.poly_degree),
        "--cell_name", str(args.cell_name)
    ]
    run_command(cmd_fit, "Fit Model and Analyze Results")

    print(f"\n{'='*50}")
    print("      Workflow finished successfully!")
    print(f"      Final analysis summary is in: {results_dir / 'overall_results.txt'}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()