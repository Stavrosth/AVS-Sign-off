# AVS Sing-off | CASLab 2025 Summer Internship
**Author: Stavros Stathoudakis**  
Mentor: Akis Tsoumanis  
Supervising Professor: Christos Sotiriou  
Inspired by Christian Lütkemeyer


## Project Goal 
AVS's goals is to reduce power consumption. This project aims at enabling that ability. In particular, this project runs a multicorner analysis for a cell and stores the measurements. Then it proceeds to fit the measurements using Least Squares.  
With this method we can interpolate the delay of a cell for virtually all PVT corners. This allows us observe power savings between corners as well as identify weak points in the library that drive the voltage upwards.  


## Python scripts 
### run_all.py
Runs the entire simulation and analysis workflow. Provides a single entry point to run the three stages of the process:
1.  Corner Generation (`create_corners.py`): Creates the necessary .scs files for all corners.
2.  Spectre Simulation (`spectre_run.py`): Runs Spectre on all generated corner files.
3.  Polynomial Fitting (`poly_fit.py`): Analyzes the simulation results and generates predictive models.

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


### create_corners.py
- Generates TT/SS/FF .scs for different stable input pins, and for different load configurations.  
- In order to change the files that contain the corner cases navigate to line 60-63 and modify the INCLUDE parameters.

    necessary arguments:
    input_netlist :        Path to base .scs netlist
    output_dir :           Directory to write generated .scs files

    optional arguments:
    
        Operating Modes:
            --vstep VSTEP : Voltage step (default 0.01) [SWEEP MODE]
            --fixed-v FIXED_V : If set, generate TT/SS/FF each at this single voltage (e.g., 0.96) [FIXED-V MODE]

        Multi-pin Functionality:
            --pins PINS : Number of input pins on the cell to test. Assumes the input signal is named 'in10'.
            --pin-names PIN_NAMES : Comma-separated list of logical pin names (e.g., "A,B") corresponding to pin indices 1, 2, ...
            --base-lib-file BASE_LIB_FILE : Path to the .lib file to use for capacitance parsing. This single file will be used for all corners (tt, ss, ff).
            --cell-map CELL_MAP : Mapping from .scs cell names to .lib cell names. Format: "SCS_NAME:LIB_NAME"

        General Options:
            --temps TEMPS : Comma-separated temperatures (default: 0,50,125)


### spectre_run.py
- Runs Spectre recursively on .scs files found in a directory structure, organizing results to mirror the input structure.
- Measurements are stores in csv files.

        necessary arguments:
            path : A single .scs file or a directory to search recursively for .scs files.

        optional arguments:
            --glob GLOB :        Glob pattern for simulation files (default: *.scs)
            --meas MEAS :        Measurement name inside .measure file (default: delay_out8_ou6)
            --out-csv OUT_CSV :  The filename for the output CSV(s) (default: measurements.csv)
            --spectre-args ... :   Args after -- are passed to Spectre


### poly_fit.py
- Creates a predictive model for circuit delay.
- This script can be pointed at a directory (e.g., 'results/') and will
recursively find and process all 'measurements.csv' files within it.
For each CSV, it generates a full set of plots and analysis files in that
CSV's local directory.
- Aggregates the key voltage shift and anchor delay metrics from
all runs into a single 'overall_results.txt' file at the top level.

The fitting process involves:
1. Fitting a base delay model using non-linear least squares.
2. Fitting a polynomial ridge regression model to the residuals.

        necessary arguments:
            path :                 Directory to search recursively for '*.csv' files.

        optional arguments:
            --poly_degree POLY_DEGREE : Polynomial degree for the residual model.


### calc_delay.py
Calculates and plots cell delay based on a the timing model calculated in interoplation (poly_fit.py).

    Arguments:
        -p PROCESS, --process PROCESS : Process corner(s), comma-separated. (e.g., 'ss' or 'ss,tt,ff')
        -v VOLTAGE, --voltage VOLTAGE : Voltage in Volts, as a single value or a range. (e.g., '0.8' or '0.7-0.9')
        -t TEMPERATURE, --temperature TEMPERATURE : Temperature in Celsius, as a single value or a range. (e.g., '25' or '0-125')
        -i INPUT, --input INPUT : Path to the input parameters file (default: results.txt).
        -o OUTPUT, --output OUTPUT : Path to save the output plot PNG (default: delay_prediction_plot.png).
        --csv CSV :            Optional: Path to save the output data as a CSV file.
        --integral INTEGRAL :   Number of integral points to generate for input ranges (default: 20).


### modify_scs.py
- Creates new .scs files for different cells.
- It has as input an existing .scs file and modifies it to create a new .scs file with a different cell name and stable inputs.

        Arguments: 
            --in INPUT_SCS : Path to input .scs file
            --out OUTPUT_SCS : Path to output .scs file
            --cell CELL_NAME : New cell name (e.g., NAND2X1)
            --inputs :  Stable input names separated by spaces (e.g., 'vss' or 'vss vdd')


### lib_parser.py
Module for create_corners.py script. Helps with parsint the .lib file to find the capacitance for each cell and pin In particulare it searches the cell that corresponds to the cell used in the .scs file.  
Can not be called to run on its own