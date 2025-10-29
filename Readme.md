## Stavros Stathoudakis | CASLab 2025 Summer Internship
Work Inspired by Christian Lütkemeyer
Mentor: Akis Tsoumanis


## Project Goal ##

.
.
.
.

## Python scripts ##
run_all.py: Runs all the scripts automatically and generates the results in the results folder.
    necessary arguments:
        input_dir :            Directory containing the base .scs netlist files.

    optional arguments:

        Corner Generation Options (for create_corners.py):
            --pins PINS :          Number of input pins for multi-pin analysis.
            --pin-names PIN_NAMES : Comma-separated list of logical pin names (e.g., "A,B")
            --cell-map CELL_MAP :  Mapping from .scs cell names to .lib cell names. Format: "SCS_NAME_1:LIB_NAME_1,SCS_NAME_2:LIB_NAME_2"
            --base-lib-file BASE_LIB_FILE : Path to the .lib file to use for capacitance parsing.
            --vstep VSTEP :        Voltage step for sweep mode.

        Fitting Options (for poly_fit.py):
            --poly_degree POLY_DEGREE : Polynomial degree for the residual model.



create_corners.py: Generate TT/SS/FF .scs with for multiple pins, and for different load configurations.

    necessary arguments:
    input_netlist :        Path to base .scs netlist
    output_dir :           Directory to write generated .scs files

    optional arguments:
    
        Operating Modes:
            --vstep VSTEP :        Voltage step (default 0.01) [SWEEP MODE]
            --fixed-v FIXED_V :    If set, generate TT/SS/FF each at this single voltage (e.g., 0.96) [FIXED-V MODE]

        Multi-pin Functionality:
            --pins PINS :          Number of input pins on the cell to test. Assumes the input signal is named 'in10'.
            --pin-names PIN_NAMES : Comma-separated list of logical pin names (e.g., "A,B") corresponding to pin indices 1, 2, ...
            --base-lib-file BASE_LIB_FILE : Path to the .lib file to use for capacitance parsing. This single file will be used for all corners (tt, ss, ff).
            --cell-map CELL_MAP :  Mapping from .scs cell names to .lib cell names. Format: "SCS_NAME:LIB_NAME"

        General Options:
            --temps TEMPS :        Comma-separated temperatures (default: 0,50,125)



spectre_run.py: Runs Spectre recursively on .scs files found in a directory structure, organizing results to mirror the input structure.

    necessary arguments:
        path :               A single .scs file or a directory to search recursively for .scs files.

    optional arguments:
        --glob GLOB :        Glob pattern for simulation files (default: *.scs)
        --meas MEAS :        Measurement name inside .measure file (default: delay_out8_ou6)
        --out-csv OUT_CSV :  The filename for the output CSV(s) (default: measurements.csv)
        --spectre-args ... :   Args after -- are passed to Spectre



poly_fit.py: Fit a PVT-aware delay model to all '*.csv' files found in a directory.

    necessary arguments:
        path :                 Directory to search recursively for '*.csv' files.

    optional arguments:
        --poly_degree POLY_DEGREE : Polynomial degree for the residual model.


calc_delay.py: Calculates and plots cell delay based on a the timing model calculated in interoplation (poly_fit.py).

    -p PROCESS, --process PROCESS : Process corner(s), comma-separated. (e.g., 'ss' or 'ss,tt,ff')
    -v VOLTAGE, --voltage VOLTAGE : Voltage in Volts, as a single value or a range. (e.g., '0.8' or '0.7-0.9')
    -t TEMPERATURE, --temperature TEMPERATURE : Temperature in Celsius, as a single value or a range. (e.g., '25' or '0-125')
    -i INPUT, --input INPUT : Path to the input parameters file (default: results.txt).
    -o OUTPUT, --output OUTPUT : Path to save the output plot PNG (default: delay_prediction_plot.png).
    --csv CSV :            Optional: Path to save the output data as a CSV file.
    --integral INTEGRAL :   Number of integral points to generate for input ranges (default: 20).



modify_scs.py: Modify Spectre .scs file for different logic gates.
    parser = argparse.ArgumentParser(description="")
    --in INPUT_SCS : Path to input .scs file
    --out OUTPUT_SCS : Path to output .scs file
    --cell CELL_NAME : New cell name (e.g., NAND2X1)
    --inputs :  Stable input names separated by spaces (e.g., 'vss' or 'vss vdd')



lib_parser.py: helper module for create_corners.py script. Helps with parsint the .lib file to find the capacitance for each cell and pin In particulare it searches the cell that corresponds to the cell used in the .scs file.