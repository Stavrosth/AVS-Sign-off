"""
poly_fit.py

A script to create a predictive model for circuit delay.

This script can be pointed at a directory (e.g., 'results/') and will
recursively find and process all 'measurements.csv' files within it.
For each CSV, it generates a full set of plots and analysis files in that
CSV's local directory.

Finally, it aggregates the key voltage shift and anchor delay metrics from
all runs into a single 'overall_results.txt' file at the top level.

The fitting process involves:
1. Fitting a base delay model using non-linear least squares.
2. Fitting a polynomial ridge regression model to the residuals.

Usage:
    python poly_fit.py <path> [--poly_degree POLY_DEGREE]
"""

from __future__ import division, print_function
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from typing import List, Dict

# ----------------- Configuration & Constants -----------------
CSV_INPUT_NAME = 'measurements.csv'
MATCH_CSV_NAME = 'voltage_match.csv'
PREDICTIONS_CSV_NAME = 'predictions.csv'
OVERLAY_PLOT_NAME = 'overlay_fit.png'
PRED_SCATTER_PLOT_NAME = 'measured_vs_predicted.png'
RESULTS_TXT_NAME = 'results.txt'
OVERALL_RESULTS_TXT_NAME = 'overall_results.txt'
PROCESS_MAP = {"tt": 0.0, "ff": -1.0, "ss": 1.0}
VMIN_GLOBAL = 0.4 # Minimum voltage for searching matches

# ----------------- Base Delay Model Definition -----------------
class BaseModelParameters:
    __slots__ = ['D0', 'kDP', 'kDT', 'Vt', 'kVtP', 'kVtT', 'alpha', 'kAlphaP', 'kAlphaT']
    def __init__(self, D0, kDP, kDT, Vt, kVtP, kVtT, alpha, kAlphaP, kAlphaT):
        self.D0, self.kDP, self.kDT = D0, kDP, kDT
        self.Vt, self.kVtP, self.kVtT = Vt, kVtP, kVtT
        self.alpha, self.kAlphaP, self.kAlphaT = alpha, kAlphaP, kAlphaT
        def as_array(self): return np.array([getattr(self, attr) for attr in self.__slots__], dtype=float)
    @classmethod
    def from_array(cls, arr): return cls(*arr)
    def __str__(self):
        return (f'D0    = {self.D0:8.6g}, kDP = {self.kDP:8.6g}, kDT = {self.kDT:8.6g}\n'
                f'Vt    = {self.Vt:8.6g}, kVtP= {self.kVtP:8.6g}, kVtT= {self.kVtT:8.6g}\n'
                f'alpha = {self.alpha:8.6g}, kAlphaP={self.kAlphaP:8.6g}, kAlphaT={self.kAlphaT:8.6g}')

# ----------------- Base Delay Prediction Function -----------------
def base_delay(params, P, V, T):
    D_eff = params.D0 * (1.0 + params.kDP * P + params.kDT * T)
    Vt_eff = params.Vt + params.kVtP * P + params.kVtT * T
    alpha_eff = params.alpha + params.kAlphaP * P + params.kAlphaT * T
    epsilon = 1e-6
    dv = V - Vt_eff
    is_bad = dv <= epsilon
    safe_dv = np.where(is_bad, epsilon, dv)
    safe_alpha_eff = np.clip(alpha_eff, a_min=-50, a_max=50)
    denominator = np.power(safe_dv, safe_alpha_eff)
    safe_denominator = np.clip(denominator, 1e-9, None)
    pred = D_eff / safe_denominator
    delay_at_boundary = np.clip(D_eff / np.power(epsilon, safe_alpha_eff), a_min=None, a_max=1e12)
    penalty = delay_at_boundary - 1e4 * (dv - epsilon)
    return np.where(is_bad, penalty, pred)

# ----------------- Model Fitting Pipeline -----------------
def fit_base_model(P, V, T, y):
    x0 = np.array([float(np.median(y)), 0.0, 0.0, float(max(0.4, np.min(V) - 0.05)), 0.0, 0.0, 1.3, 0.0, 0.0])
    def residuals(x): return base_delay(BaseModelParameters.from_array(x), P, V, T) - y
    bounds = (np.array([1e-9, -10, -10, 0.2, -5, -5, 0.1, -5, -5]),
              np.array([1e9, 10, 10, 1.2, 5, 5, 5.0, 5, 5]))
    res = least_squares(residuals, x0, bounds=bounds, loss='soft_l1', f_scale=1.0, max_nfev=10000)
    return BaseModelParameters.from_array(res.x)

def build_residual_features(P, V, T, base_params):
    Vt_eff = base_params.Vt + base_params.kVtP * P + base_params.kVtT * T
    u = np.log(np.maximum(V - Vt_eff, 1e-6))
    return np.stack([P, T, u], axis=1)

def fit_residual_model(P, V, T, y, base_params, degree=3):
    residuals = y - base_delay(base_params, P, V, T)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) * 1.4826 + 1e-12
    mask = np.abs((residuals - med) / mad) <= 3.5
    if mask.sum() < 0.7 * len(residuals): mask = np.ones_like(residuals, dtype=bool)
    X, r = build_residual_features(P, V, T, base_params)[mask], residuals[mask]
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                          RidgeCV(alphas=np.logspace(-6, 3, 20), cv=5, fit_intercept=True))
    model.fit(X, r)
    return model

# ----------------- Prediction and Analysis -----------------
class FullModel:
    def __init__(self, base_params, residual_model):
        self.base_params = base_params
        self.residual_model = residual_model
    def predict(self, P, V, T):
        base_pred = base_delay(self.base_params, P, V, T)
        X_resid = build_residual_features(P, V, T, self.base_params)
        return base_pred + self.residual_model.predict(X_resid)

def compute_voltage_matches_and_shift(df, model):
    anchor_df = df[(df['section'] == 'ss') & (df['temp'] == 0)]

    if anchor_df.empty:
        print("Warning: Anchor data ('ss' corner at temp=0) not found. Cannot calculate shift.")
        return None, None, None, np.nan
    
    v_anchor = anchor_df['dc'].mean()
    delay_target = model.predict(np.array([PROCESS_MAP['ss']]), np.array([v_anchor]), np.array([0.0]))[0]
    print(f"Anchor point: 'ss', 0C at avg V={v_anchor:.4f}V has target delay={delay_target:.4f} ps")
    match_rows, voltage_shift = [], np.nan
    
    for sec, temp in sorted(set(zip(df['section'], df['temp']))):
        p_val = PROCESS_MAP.get(sec, 0.0)
        def objective_func(v): return model.predict(np.array([p_val]), np.array([v]), np.array([float(temp)]))[0] - delay_target
        corner_df = df[(df['section'] == sec) & (df['temp'] == temp)]
        
        if corner_df.empty: continue
        search_min = max(VMIN_GLOBAL, corner_df['dc'].min() - 0.1)
        search_max = corner_df['dc'].max() + 0.1
        
        try:
            # Check if a root exists in the interval
            if np.sign(objective_func(search_min)) == np.sign(objective_func(search_max)):
                raise ValueError("Objective function has the same sign at both ends of the interval.")
            
            matched_voltage = brentq(objective_func, a=search_min, b=search_max)
            match_rows.append({'section': sec, 'temp': temp, 'target_delay_ps': delay_target, 'matched_voltage_v': matched_voltage})

            if sec == 'ff' and temp == 0: voltage_shift = v_anchor - matched_voltage
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Could not find matching voltage for {sec} @ {temp}C. Reason: {e}")
            match_rows.append({'section': sec, 'temp': temp, 'target_delay_ps': delay_target, 'matched_voltage_v': np.nan})

    return pd.DataFrame(match_rows), v_anchor, delay_target, voltage_shift

# ----------------------- Plotting & Main Execution -----------------------
def plot_overlay_fit(df, model, output_path, anchor_v=None, anchor_delay=None):
    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df['temp'].unique())))
    temp_color_map = {t: c for t, c in zip(sorted(df['temp'].unique()), colors)}

    for sec in sorted(df['section'].unique()):

        sec_df = df[df['section'] == sec]
        if sec_df.empty: continue
        vmin, vmax = sec_df['dc'].min(), sec_df['dc'].max()
        v_grid = np.linspace(vmin, vmax, 200)

        for t in sorted(sec_df['temp'].unique()):
            p_arr, t_arr = np.full_like(v_grid, PROCESS_MAP.get(sec, 0.0)), np.full_like(v_grid, float(t))
            pred = model.predict(p_arr, v_grid, t_arr)
            ax.plot(v_grid, pred, lw=1.2, alpha=0.9, color=temp_color_map.get(t), label=f'{sec.upper()} @ {t}°C Model')
            ax.scatter(sec_df[sec_df['temp'] == t]['dc'], sec_df[sec_df['temp'] == t]['delay_ps'], s=20, alpha=0.75, color=temp_color_map.get(t), label=f'{sec.upper()} @ {t}°C Measured', edgecolors='none')

    if anchor_v is not None and anchor_delay is not None:
        ax.axhline(y=anchor_delay, color='red', ls='--', lw=1.5, label=f'Anchor Delay ({anchor_delay:.2f} ps)')
        ax.plot(anchor_v, anchor_delay, 'r*', ms=12, zorder=10, label=f'Anchor Point ({anchor_v:.2f}V)')

    ax.set_xlabel('Voltage (V)'); ax.set_ylabel('Delay (ps)'); ax.set_title(f'Model Fit vs. Measured Data\n(Source: {output_path.parent.name})')
    ax.grid(True, which='both', ls='--', lw=0.5); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(output_path, dpi=160); plt.close(fig)

def plot_measured_vs_predicted(y_train_true, y_train_pred, y_test_true, y_test_pred, output_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_train_true, y_train_pred, s=15, alpha=0.6, edgecolors='none', c='blue', label='Train Data')
    ax.scatter(y_test_true, y_test_pred, s=25, alpha=0.8, edgecolors='none', c='orange', label='Test Data')

    all_y = np.concatenate([y_train_true, y_train_pred, y_test_true, y_test_pred])
    lims = [np.min(all_y), np.max(all_y)]

    ax.plot(lims, lims, 'r--', lw=1.0, label='y=x (Ideal)')
    ax.set_xlabel('Measured Delay (ps)'); ax.set_ylabel('Predicted Delay (ps)')
    ax.set_title(f'Measured vs. Predicted Delay\n(Source: {output_path.parent.name})'); ax.grid(True, ls='--', lw=0.5)
    ax.axis('equal'); ax.legend(); fig.tight_layout()
    fig.savefig(output_path, dpi=160); plt.close(fig)

def process_csv_file(csv_path: Path, poly_degree: int) -> Dict:
    """
    Loads a single CSV, runs the full analysis, saves all local outputs,
    and returns the key metrics for aggregation.
    """
    print(f"\n{'='*20}\nProcessing: {csv_path}\n{'='*20}")
    output_dir = csv_path.parent
    
    try:
        df = pd.read_csv(csv_path, dtype={'temp': float, 'delay_ps': float, 'dc': float}).dropna(subset=['delay_ps'])
        if df.empty:
            print("Warning: CSV is empty or contains no valid delay data. Skipping.")
            return {}
    except FileNotFoundError:
        print(f"Error: Input file not found at '{csv_path}'"); return {}
    
    print(f"Loaded {len(df)} valid data points.")
    P_arr = np.array([PROCESS_MAP.get(s, 0.0) for s in df['section']])
    V_arr, T_arr, y_arr = df['dc'].values, df['temp'].values, df['delay_ps'].values
    
    P_train, P_test, V_train, V_test, T_train, T_test, y_train, y_test = train_test_split(
        P_arr, V_arr, T_arr, y_arr, test_size=0.20, random_state=42
    )
    print(f"Split data into {len(y_train)} training and {len(y_test)} test points.")

    print("Fitting models on training data...");
    base_params = fit_base_model(P_train, V_train, T_train, y_train)
    residual_model = fit_residual_model(P_train, V_train, T_train, y_train, base_params, degree=poly_degree)
    full_model = FullModel(base_params, residual_model)

    print("Computing voltage matches and shift...");
    match_df, anchor_v, anchor_d, v_shift = compute_voltage_matches_and_shift(df, full_model)
    
    print("Generating outputs and calculating performance...");
    plot_overlay_fit(df, full_model, output_dir / OVERLAY_PLOT_NAME, anchor_v, anchor_d)
    
    y_pred_train = full_model.predict(P_train, V_train, T_train)
    y_pred_test = full_model.predict(P_test, V_test, T_test)

    plot_measured_vs_predicted(y_train, y_pred_train, y_test, y_pred_test, output_dir / PRED_SCATTER_PLOT_NAME)
    
    y_pred_full = full_model.predict(P_arr, V_arr, T_arr)
    if match_df is not None: match_df.to_csv(output_dir / MATCH_CSV_NAME, index=False, float_format='%.6g')
    df.assign(predicted_delay_ps=y_pred_full, error_ps=y_arr - y_pred_full).to_csv(output_dir / PREDICTIONS_CSV_NAME, index=False, float_format='%.6g')
    
    rrmse_train = (np.sqrt(mean_squared_error(y_train, y_pred_train)) / np.mean(y_train))
    r2_train = r2_score(y_train, y_pred_train)
    rrmse_test = (np.sqrt(mean_squared_error(y_test, y_pred_test)) / np.mean(y_test))
    r2_test = r2_score(y_test, y_pred_test)
    
    with open(output_dir / RESULTS_TXT_NAME, 'w') as f:
        f.write(f"--- Fitted Base Model Parameters ---\n{base_params}\n\n"
                f"--- Model Performance (On Training Data) ---\n"
                f"R-squared (R2 Score): {r2_train:.6f}\n"
                f"RRMSE:                  {rrmse_train:.6f}\n\n"
                f"--- Model Performance (On UNSEEN Test Data) ---\n"
                f"R-squared (R2 Score): {r2_test:.6f}\n"
                f"RRMSE:                  {rrmse_test:.6f}\n\n"
                f"--- Voltage Shift Analysis ---\n")
        f.write(f"Anchor Delay (ss, 0C): {anchor_d:.6f} ps\n" if anchor_d is not None else "Anchor delay not computed.\n")
        f.write(f"Voltage Shift (V_anchor[ss,0C] - V_match[ff,0C]): {v_shift:.6f} V\n" if not np.isnan(v_shift) else
                "Could not compute voltage shift.\n")
    
    print(f"Local outputs saved to directory '{output_dir}'.")
    return {'source_csv': csv_path, 'voltage_shift': v_shift, 'anchor_delay': anchor_d}


def main():
    parser = argparse.ArgumentParser(description="Fit a PVT-aware delay model to all 'measurements.csv' files found in a directory.")
    parser.add_argument('path', type=Path, help=f"Directory to search recursively for '{CSV_INPUT_NAME}' files.")
    parser.add_argument('--poly_degree', type=int, default=5, help='Polynomial degree for the residual model.')
    args = parser.parse_args()

    base_dir = args.path.resolve()
    if not base_dir.is_dir():
        print(f"Error: Input path '{base_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Find all 'measurements.csv' files recursively
    csv_files_to_process = sorted(list(base_dir.rglob(CSV_INPUT_NAME)))

    if not csv_files_to_process:
        print(f"Error: No '{CSV_INPUT_NAME}' files found within '{base_dir}'.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(csv_files_to_process)} CSV file(s) to process.")
    
    overall_results: List[Dict] = []
    for csv_path in csv_files_to_process:
        result = process_csv_file(csv_path, args.poly_degree)
        if result:
            overall_results.append(result)

    # Write the final aggregated results file
    overall_results_path = base_dir / OVERALL_RESULTS_TXT_NAME
    print(f"\n--- Aggregating all results into '{overall_results_path}' ---")
    with open(overall_results_path, 'w') as f:
        f.write("--- Overall Voltage Shift and Anchor Delay Summary ---\n\n")
        for result in overall_results:
            relative_path = result['source_csv'].relative_to(base_dir)
            f.write(f"Source: {relative_path}\n")
            
            anchor_delay = result.get('anchor_delay')
            if anchor_delay is not None:
                f.write(f"  - Anchor Delay:  {anchor_delay:.6f} ps\n")
            else:
                f.write("  - Anchor Delay:  Not computed\n")
            
            v_shift = result.get('voltage_shift')
            if v_shift is not None and not np.isnan(v_shift):
                f.write(f"  - Voltage Shift: {v_shift:.6f} V\n\n")
            else:
                f.write("  - Voltage Shift: Not computed\n\n")
    
    print("Done. All processing complete.")


if __name__ == '__main__':
    main()
