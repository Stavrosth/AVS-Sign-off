"""
calc_delay.py
This script calculates and plots the delay of a cell based on a timing model
defined by parameters read from a 'results.txt' file. The delay can be computed
for different process corners, supply voltages, and temperatures.

The results.txt file is the output of the poly_fit.py script, which fits the timing
model parameters.
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Map process corner names to numerical values used in the equations.
PROCESS_MAP = {
    'ss': 1.0,  # Slow-Slow
    'tt': 0.0,   # Typical-Typical
    'ff': -1.0    # Fast-Fast
}

# --- Functions ---
def parse_parameters(filepath):
    """
    Parses the 9 model parameters (D0, kDP, kDT, etc.) from the provided text file.
    
    Args:
        filepath (str): The path to the 'results.txt' file.
        
    Returns:
        dict: A dictionary containing the parameter names and their float values.
    """
    
    params = {}
    # This regular expression looks for lines with 'key = value' and captures them.
    param_regex = re.compile(r"(\w+)\s*=\s*([-\d.]+)")
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            matches = param_regex.findall(content)
            for key, value in matches:
                params[key] = float(value)
        
        # Verify that all 9 required parameters were successfully extracted.
        required_keys = ['D0', 'kDP', 'kDT', 'Vt', 'kVtP', 'kVtT', 'alpha', 'kAlphaP', 'kAlphaT']
        if not all(key in params for key in required_keys):
            raise ValueError("Not all 9 required parameters were found in the file.")
        return params
    except FileNotFoundError:
        print(f"Error: The parameter file '{filepath}' was not found.", file=sys.stderr)
        sys.exit(1)
    except (ValueError, IndexError) as e:
        print(f"Error parsing parameter file: {e}", file=sys.stderr)
        sys.exit(1)

class DelayModel:
    """
    Implements the delay calculation based on the provided model equations.
    The model is initialized with the parameters parsed from the results file.
    """
    def __init__(self, params):
        self.p = params

    def predict(self, P, VDD, T):
        """
        Calculates the delay D(P, VDD, T) for a given Process, Voltage, and Temperature.
        
        Args:
            P (float): The numerical value for the process corner.
            VDD (np.ndarray): An array of voltage points.
            T (float): The temperature in Celsius.
            
        Returns:
            np.ndarray: An array of calculated delay values in picoseconds.
        """
        # D_eff = D0 * (1 + kDP*P + kDT*T)
        D_eff = self.p['D0'] * (1 + self.p['kDP'] * P + self.p['kDT'] * T)
        
        # Vt_eff = Vt * (1 + kVtP*P + kVtT*T)
        Vt_eff = self.p['Vt'] * (1 + self.p['kVtP'] * P + self.p['kVtT'] * T)
        
        # alpha_eff = alpha * (1 + kAlphaP*P + kAlphaT*T)
        alpha_eff = self.p['alpha'] * (1 + self.p['kAlphaP'] * P + self.p['kAlphaT'] * T)
        
        # Ensure that the base of the power is positive to avoid invalid calculations.
        denominator = VDD - Vt_eff
        if np.any(denominator <= 0):
            # For any invalid VDD values, we return NaN (Not a Number).
            delay = np.full_like(VDD, np.nan, dtype=float)
            valid_indices = denominator > 0
            delay[valid_indices] = D_eff / (denominator[valid_indices] ** alpha_eff)
            return delay
            
        # D(P,VDD,T) = D_eff / (VDD - Vt_eff)^alpha_eff
        return D_eff / (denominator ** alpha_eff)

def plot_results(df, output_path):
    """
    Plots the calculated delay vs. voltage and saves it to the specified path.
    The plot format is adjusted to be more compact when many lines are present.
    
    Args:
        df (pd.DataFrame): DataFrame with calculation results.
        output_path (str): The file path where the plot image will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 7)) # More compact figure size
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    
    # Group data by process and temperature to plot separate lines.
    groups = df.groupby(['process', 'temperature'])
    num_lines = len(groups)

    for i, ((proc, temp), group) in enumerate(groups):
        ax.plot(group['voltage'], group['delay'],
                marker=markers[i % len(markers)], 
                linestyle='-',
                ms=4, # Smaller markers
                lw=1.2, # Thinner lines
                label=f'{proc.upper()} @ {temp}°C')

    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Delay (ps)')
    ax.set_title('Predicted Cell Delay vs. Voltage')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adjust legend for compactness if there are many lines
    if num_lines > 8:
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.legend()
        
    fig.tight_layout()
    
    try:
        # Create the directory for the output file if it doesn't exist.
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        print(f"\nPlot has been saved to '{output_path}'", file=sys.stderr)
    except Exception as e:
        print(f"\nError: Could not save the plot to '{output_path}'.\n{e}", file=sys.stderr)
        
    plt.close(fig)

def parse_input_range(input_str, num_integral=20):
    """
    Parses a string that can be a single number or a range (e.g., '0.7-0.9').
    """
    if '-' in input_str:
        try:
            start, end = map(float, input_str.split('-'))
            return np.linspace(start, end, num_integral)
        except ValueError:
            print(f"Error: Invalid range format for '{input_str}'. Use 'start-end'.", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            return np.array([float(input_str)])
        except ValueError:
            print(f"Error: Invalid numeric value '{input_str}'.", file=sys.stderr)
            sys.exit(1)

def parse_process_corners(input_str):
    """
    Parses a comma-separated string of process corners.
    """
    processes = [p.strip().lower() for p in input_str.split(',')]
    for p in processes:
        if p not in PROCESS_MAP:
            print(f"Error: Invalid process corner '{p}'. Available options: {list(PROCESS_MAP.keys())}", file=sys.stderr)
            sys.exit(1)
    return processes


# --- Main Execution ---

if __name__ == "__main__":
    # Set up command-line argument parsing.
    parser = argparse.ArgumentParser(
        description="Calculate and plot cell delay based on a timing model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Required arguments
    parser.add_argument('-p', '--process', type=str, required=True, help="Process corner(s), comma-separated.\nE.g., 'ss' or 'ss,tt,ff'")
    parser.add_argument('-v', '--voltage', type=str, required=True, help="Voltage in Volts, as a single value or a range.\nE.g., '0.8' or '0.7-0.9'")
    parser.add_argument('-t', '--temperature', type=str, required=True, help="Temperature in Celsius, as a single value or a range.\nE.g., '25' or '0-125'")
    
    # Optional arguments for file paths and integral count
    parser.add_argument('-i', '--input', default='results.txt', help="Path to the input parameters file (default: results.txt).")
    parser.add_argument('-o', '--output', default='delay_prediction_plot.png', help="Path to save the output plot PNG (default: delay_prediction_plot.png).")
    parser.add_argument('--csv', type=str, help="Optional: Path to save the output data as a CSV file.")
    parser.add_argument('--integral', type=int, default=20, help="Number of integral points to generate for input ranges (default: 20).")
    
    args = parser.parse_args()

    # 1. Load model parameters from the specified file.
    parameters = parse_parameters(args.input)
    model = DelayModel(parameters)

    # 2. Parse the user-provided inputs, using the new integral argument.
    processes = parse_process_corners(args.process)
    voltages = parse_input_range(args.voltage, num_integral=args.integral)
    temperatures = parse_input_range(args.temperature, num_integral=args.integral)

    # 3. Perform calculations.
    results_list = []
    for proc_str in processes:
        p_val = PROCESS_MAP[proc_str]
        for temp in temperatures:
            delays = model.predict(p_val, voltages, temp)
            for volt, delay in zip(voltages, delays):
                 if not np.isnan(delay):
                    results_list.append({
                        'process': proc_str,
                        'voltage': volt,
                        'temperature': temp,
                        'delay': delay
                    })
    
    if not results_list:
        print("\nCould not calculate any valid delay points for the given input ranges.", file=sys.stderr)
        print("Please check if the voltage range is valid for the calculated Vt_eff.", file=sys.stderr)
        sys.exit(0)

    results_df = pd.DataFrame(results_list)

    # 4. Output the results.
    is_single_point = (len(processes) == 1 and len(voltages) == 1 and len(temperatures) == 1)
    
    print("\n--- Calculated Delays ---")
    if is_single_point:
        row = results_df.iloc[0]
        print(f"Voltage: {row['voltage']:.3f}V, Temp: {row['temperature']:.1f}°C, Process: {row['process'].upper()} -> Delay: {row['delay']:.3f} ps")
    else:
        # Sort the DataFrame to group by process, then voltage, then temperature.
        results_df = results_df.sort_values(by=['process', 'voltage', 'temperature'])
        
        # Print the formatted table to the console
        print(results_df.to_string(index=False, float_format="%.3f"))

        # If a CSV output path is provided, save the file.
        if args.csv:
            try:
                # Create the directory for the output file if it doesn't exist.
                csv_output_dir = os.path.dirname(args.csv)
                if csv_output_dir:
                    os.makedirs(csv_output_dir, exist_ok=True)
                results_df.to_csv(args.csv, index=False, float_format="%.3f")
                print(f"\nData table has been saved to '{args.csv}'", file=sys.stderr)
            except Exception as e:
                print(f"\nError: Could not save the CSV file to '{args.csv}'.\n{e}", file=sys.stderr)

        # And generate the plot at the specified location.
        plot_results(results_df, args.output)