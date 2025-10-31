"""
create_corners.py
Generates .scs netlists (TT/SS/FF) from a base Spectre netlist.

Some comments are necessary for the correct functioning of the script.
    - The chain must be inbetween the comments:
        // Chain
        ...
        // Chain ends

    - The input .scs file must end with the comment: 
        // finish

    - The instantiations of the cells must start with the string:
        ncell

Modes:
1) Sweep mode (default, legacy behavior)
   - Per-process voltage range (from PROC_INFO) with step --vstep

2) Fixed-voltage mode
   - Use --fixed-v <voltage> to generate exactly THREE files (tt/ss/ff).
   - By default, no .ALTER blocks are appended.
   - Opt-in to temp alters via --with-alters to generate 0,50,125 °C alters.
   - You can change the temp set with --temps "0,50,125".

3) Multi-pin mode
   - Use --pins <N> for cells with N inputs. The input signal is assumed to be 'in10'.
   - This will generate N sets of corner files, one for each input pin being active.
   - For each pin `i` from 1 to N, it creates a directory (e.g., corners/corner_i)
     and generates netlists where the active signal is connected to the i-th pin
     for ALL cells in the chain, ignoring any line starting with 'vin'.
   - Use --pin-names "A,B,..." to map pin indices (1, 2, ...) to
     logical pin names from the .lib file for capacitance parsing.
   - Use --base-lib-file <path> to specify the .lib file to parse.
   - Use --cell-map "SCS_CELL:LIB_CELL,..." to map .scs cell names
     to their corresponding .lib cell names.

4) Load-splitting mode 
   - If the input netlist contains comment blocks:
     //Gate load start ... //Gate load ends
     //Wire load start ... //wire load ends
   - The script will automatically generate two sets of outputs in subdirectories
     named 'gate_load' and 'wire_load'.
   - The 'gate_load' version will contain ONLY the gate load section.
   - The 'wire_load' version will contain ONLY the wire load section.
   - **Capacitance replacement from .lib files only runs for 'wire_load'.**    
"""

import argparse
import re
import lib_parser # Local module for .lib parsing included in the lib_parser.py file

from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, List, Tuple, Dict

#### ----- This needs to change based on the files that contain the corner cases ----- ###
# ----- Foundry include tokens -----
INCLUDE_TT  = "ihp013ng_include_all"
INCLUDE_SS  = "ihp013ng_include_all_slow"
INCLUDE_FF  = "ihp013ng_include_all_fast"
INCLUDE_PATTERN = re.compile(r"(ihp013ng_include_all(?:_slow|_fast)?)", flags=re.IGNORECASE)

# ----- Per-process meta -----
PROC_INFO = {
    "tt": {
        "include": INCLUDE_TT, "vmin": Decimal("0.96"), "vmax": Decimal("1.44"), "suffix": "tt",
    },
    "ff": {
        "include": INCLUDE_FF, "vmin": Decimal("0.96"), "vmax": Decimal("1.44"), "suffix": "ff",
    },
    "ss": {
        "include": INCLUDE_SS, "vmin": Decimal("0.96"), "vmax": Decimal("1.44"), "suffix": "ss",
    },
}

# ----- Regex helpers -----
VVDVAL_PATTERN = re.compile(r"(?i)(^|\s)(\.param\s+vvdval\s*=\s*)([0-9]*\.?[0-9]+)", flags=re.IGNORECASE | re.MULTILINE)
TEMP_PATTERN   = re.compile(r"(?i)^\s*\.temp\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$", flags=re.MULTILINE)

# Regex for load splitting
GATE_LOAD_PATTERN = re.compile(r"//\s*Gate\s+load\s+start.*//\s*Gate\s+load\s+ends", re.IGNORECASE | re.DOTALL)
WIRE_LOAD_PATTERN = re.compile(r"//\s*Wire\s+load\s+start.*//\s*wire\s+load\s+ends", re.IGNORECASE | re.DOTALL)


"""
Specific modifications for the 'gate_load' netlist.
    - Removes trailing '0' from primary input/output pins (e.g., out60 -> out6, in10 -> in1).
    - Updates .meas statements to match the new pin names (e.g., V(out60) -> V(out6)).
"""
def modify_gate_load_netlist(text: str) -> str:
    print("--- Applying special modifications for gate_load netlist. ---")
    modified_text = re.sub(r'\b((?:in|out)\d*)0\b', r'\1', text)
    return modified_text


"""
Analyzes netlist for gate/wire load comments and returns versions of the text.
Returns: [ (load_subdir, processed_text), ... ]
e.g., [ ('gate_load', text_A), ('wire_load', text_B) ]
or    [ ('', original_text) ]
"""
def process_load_sections(netlist_text: str) -> List[Tuple[str, str]]:
    has_gate_load = GATE_LOAD_PATTERN.search(netlist_text)
    has_wire_load = WIRE_LOAD_PATTERN.search(netlist_text)

    if has_gate_load and has_wire_load:
        print("--- Found Gate and Wire load sections. Splitting outputs. ---")
        # Create a version with ONLY the gate load (by removing the wire load section)
        gate_load_text_original = WIRE_LOAD_PATTERN.sub("", netlist_text)
        
        gate_load_text = modify_gate_load_netlist(gate_load_text_original)
        
        # Create a version with ONLY the wire load (by removing the gate load section)
        wire_load_text = GATE_LOAD_PATTERN.sub("", netlist_text)
        
        return [
            ("gate_load", gate_load_text),
            ("wire_load", wire_load_text),
        ]
    else:
        print("--- Gate/Wire load sections not found. Proceeding with original netlist. ---")
        return [("", netlist_text)]


"""Return only the base (pre-.ALTER) portion of the deck."""
def extract_pre_alter(text: str) -> str:
    parts = re.split(r'^\s*//finish', text, flags=re.MULTILINE)
    return parts[0]

def build_temp_alters_for_voltage(v: Decimal, temps: List[Decimal]) -> str:
    v_str = f"{v:.3f}"
    out_lines = []
    for t in temps:
        t_str = f"{Decimal(str(t)):.1f}"
        out_lines.append("* Auto .ALTER for fixed-V run\n")
        out_lines.append(".ALTER\n")
        out_lines.append(f".TEMP {t_str}\n")
        out_lines.append(f".PARAM VVDVAL={v_str}\n\n")
    return "".join(out_lines)

def build_alter_block_for_voltage(v: Decimal) -> str:
    return build_temp_alters_for_voltage(v, [Decimal("0"), Decimal("50"), Decimal("125")])

def set_initial_vvdval(text: str, v_start: Decimal) -> str:
    """Set/replace the initial VVDVAL in the base deck (before .ALTER blocks)."""
    v_str = f"{v_start:.3f}"
    if VVDVAL_PATTERN.search(text):
        def repl(m):
            prefix = m.group(1)
            head = m.group(2)
            return f"{prefix}{head}{v_str}"
        return VVDVAL_PATTERN.sub(repl, text, count=1)
    return text.rstrip() + f"\n\n* Added initial VVDVAL by create_corners.py\n.PARAM VVDVAL={v_str}\n"

def prepare_body_for_process(base_text: str, include_name: str, v_start: Decimal) -> str:
    """
    - Strip any existing .ALTER blocks from the input deck
    - Normalize include token to TT then switch to process-specific include
    - Ensure initial VVDVAL is set to v_start (e.g., 0.96)
    """
    text = extract_pre_alter(base_text)
    text = INCLUDE_PATTERN.sub(INCLUDE_TT, text)
    text = text.replace(INCLUDE_TT, include_name)
    text = set_initial_vvdval(text, v_start)
    return text

def parse_temps_arg(temps_str: str) -> List[Decimal]:
    parts = [p.strip() for p in temps_str.split(",") if p.strip() != ""]
    if not parts:
        return [Decimal("0"), Decimal("50"), Decimal("125")]
    return [Decimal(p) for p in parts]

def swap_pins_for_chain(netlist_text: str, num_pins: int, signal_name: str, target_pin_one_based: int) -> Tuple[str, Optional[str]]:
    """
    Modifies the netlist to connect the active signal to a specific input pin
    for ALL cells in a chain, ignoring any line that starts with 'vin'.
    
    Returns:
        (modified_netlist_text, master_cell_name)
    """
    lines = netlist_text.splitlines()

    # --- Step 1 & 2: Find original configuration and master cell name ---
    original_active_pin_idx = -1
    master_cell_name: Optional[str] = None

    for line in lines:
        if signal_name in line and not line.strip().startswith(('*', '//')):
            tokens = line.split()
            if len(tokens) > 1 + num_pins and not tokens[0].lower() == 'vin':
                try:
                    input_pins = tokens[1 : 1 + num_pins]
                    original_active_pin_idx = input_pins.index(signal_name)
                    master_cell_name = tokens[-1]
                    print(f"--- Found master .scs cell name: {master_cell_name} ---")
                    break
                except (ValueError, IndexError):
                    continue

    if original_active_pin_idx == -1:
        print(f"--- WARNING: Could not find signal '{signal_name}' in a valid cell line. Pin swapping skipped.")
        return netlist_text, None
        
    if master_cell_name is None:
        print(f"--- WARNING: Found signal '{signal_name}' but could not determine master cell name. Pin swapping skipped.")
        return netlist_text, None

    # --- Step 3: Iterate and modify all lines in the chain ---
    target_idx_zero_based = target_pin_one_based - 1

    if target_idx_zero_based < 0 or target_idx_zero_based >= num_pins:
        raise ValueError(f"Target pin {target_pin_one_based} is out of bounds for a cell with {num_pins} pins.")

    if original_active_pin_idx == target_idx_zero_based:
        return netlist_text, master_cell_name # No swap needed

    modified_lines = []
    for line in lines:
        tokens = line.split()
        is_chain_cell = len(tokens) > 1 + num_pins and tokens[-1] == master_cell_name
        is_comment = line.strip().startswith(('*', '//'))
        is_vin_line = tokens and tokens[0].lower() == 'vin'

        if is_chain_cell and not is_comment and not is_vin_line:
            try:
                input_pins = tokens[1 : 1 + num_pins]
                pin1 = input_pins[original_active_pin_idx]
                pin2 = input_pins[target_idx_zero_based]
                input_pins[original_active_pin_idx] = pin2
                input_pins[target_idx_zero_based] = pin1

                new_tokens = tokens[0:1] + input_pins + tokens[1 + num_pins:]
                modified_lines.append(" ".join(new_tokens))
            except IndexError:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    return "\n".join(modified_lines), master_cell_name


def substitute_capacitance(
    text: str,
    proc_key: str,
    load_subdir: str,
    lib_cell_name: Optional[str], # Renamed: this is now the .lib cell name
    pin_name: Optional[str],
    base_lib_file_str: Optional[str]
) -> str:
    """
    Replaces the 'Capacitance=...pf' parameter in the netlist by parsing
    the value from the corresponding .lib file.
    
    This logic only runs for:
    - 'wire_load' mode
    - When lib_cell_name, pin_name, and base_lib_file_str are all provided.
    """
    # Only run this logic for wire_load, as in the original script
    if load_subdir != 'wire_load':
        return text

    if not lib_cell_name or not pin_name or not base_lib_file_str:
        print(f"--- ({proc_key}) Skipping capacitance replacement for wire_load: Cell/pin/lib-file info missing.")
        return text

    print(f"--- ({proc_key}) Attempting to find capacitance for {lib_cell_name} / {pin_name}...")
    
    try:
        # 1. Get correct lib file path
        lib_path = lib_parser.get_lib_path(base_lib_file_str, proc_key)
        if not lib_path:
            # Error message is now printed by get_lib_path
            print(f"--- WARNING: ({proc_key}) Skipping cap replacement.")
            return text
        
        # 2. Read lib file (cached)
        lib_content = lib_parser.read_lib_file(lib_path)
        if not lib_content:
            print(f"--- WARNING: ({proc_key}) Could not read .lib file at {lib_path}. Skipping cap replacement.")
            return text

        # 3. Parse value
        cap_value_str = lib_parser.parse_capacitance(lib_content, lib_cell_name, pin_name)
        if not cap_value_str:
            print(f"--- WARNING: ({proc_key}) Could not parse capacitance for {lib_cell_name} / {pin_name} from {lib_path}. Skipping cap replacement.")
            return text

        # 4. Perform substitution
        text_with_sub, count = re.subn(
            r'(Capacitance=)[^\s]+pf',
            f'\\g<1>{cap_value_str}pf',
            text,
            count=1,
            flags=re.IGNORECASE
        )
        
        if count > 0:
            print(f"--- SUCCESS: ({proc_key}) For wire_load, set Capacitance to {cap_value_str}pf (from .lib) ---")
        else:
            print(f"--- WARNING: ({proc_key}) Could not find 'Capacitance=<value>pf' to replace for wire_load ---")
        
        return text_with_sub

    except Exception as e:
        print(f"--- ERROR during capacitance substitution for {proc_key}: {e}. Returning original text.")
        return text


def generate_corners(
    base_text: str,
    output_dir: Path,
    base_name: str,
    args: argparse.Namespace,
    load_subdir: str = "",
    pin_num: int = 0,
    scs_master_cell_name: Optional[str] = None, # Renamed for clarity
    logical_pin_name: Optional[str] = None,
    cell_map: Optional[Dict[str, str]] = None
):
    """
    Core logic to generate corner files (TT/SS/FF) from a given base netlist text.
    """
    fixed_v: Optional[Decimal] = Decimal(args.fixed_v) if args.fixed_v is not None else None
    temps = parse_temps_arg(args.temps)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Look up the .lib cell name from the .scs cell name ---
    lib_cell_name: Optional[str] = None
    if scs_master_cell_name and cell_map:
        lib_cell_name = cell_map.get(scs_master_cell_name)
        if not lib_cell_name:
            print(f"--- WARNING: No .lib mapping found for .scs cell '{scs_master_cell_name}' in --cell-map.")
        else:
            print(f"--- Mapped .scs cell '{scs_master_cell_name}' to .lib cell '{lib_cell_name}' ---")
    elif scs_master_cell_name and not cell_map:
         print(f"--- WARNING: Found .scs cell '{scs_master_cell_name}' but --cell-map is not provided. Skipping cap replacement.")

    if fixed_v is not None:
        # Fixed-voltage mode
        header_template = (
            "* ---- Auto-generated corners ({proc_upper}) [FIXED-V MODE] ----\n"
            "* FIXED_V = {fixed_v}\n"
            "* Note: No .ALTER blocks were added by default.\n\n"
        )
        for proc_key, info in PROC_INFO.items():
            # --- Call capacitance substitution ---
            corner_text = substitute_capacitance(
                base_text,
                proc_key,
                load_subdir,
                lib_cell_name,
                logical_pin_name,
                args.base_lib_file 
            )
            
            include_name = info["include"]
            suffix = info["suffix"]
            body = prepare_body_for_process(corner_text, include_name, fixed_v)
            header = header_template.format(proc_upper=proc_key.upper(), fixed_v=f"{fixed_v:.3f}")
            final_text = body.rstrip() + "\n\n" + header

            out_path = output_dir / f"{base_name}_{suffix}.scs"
            out_path.write_text(final_text)
            print(f"Wrote: {out_path}")
        return

    # Sweep mode (legacy)
    vstep = Decimal(args.vstep)
    header_template = (
        "* ---- Auto-generated corners ({proc_upper}) ----\n"
        "* VSTEP = {vstep}\n\n"
    )
    for proc_key, info in PROC_INFO.items():
        # --- NEW: Call capacitance substitution ---
        corner_text = substitute_capacitance(
            base_text,
            proc_key,
            load_subdir,
            lib_cell_name,
            logical_pin_name,
            args.base_lib_file
        )
            
        include_name = info["include"]
        vmin = info["vmin"]
        vmax = info["vmax"]
        suffix = info["suffix"]

        body = prepare_body_for_process(corner_text, include_name, vmin)
        start_alt = (vmin + vstep).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        header = header_template.format(proc_upper=proc_key.upper(), vstep=vstep)

        voltages = []
        current = start_alt
        eps = Decimal("0.0000001")
        while current <= vmax + eps:
            voltages.append(current.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
            current = (current + vstep).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        alters = "".join(build_alter_block_for_voltage(v) for v in voltages)
        final_text = body.rstrip() + "\n\n" + header + alters

        out_path = output_dir / f"{base_name}_{suffix}.scs"
        out_path.write_text(final_text)
        print(f"Wrote: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate TT/SS/FF .scs with optional alters, multi-pin, and load-splitting support.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_netlist", type=Path, help="Path to base .scs netlist")
    parser.add_argument("output_dir", type=Path, help="Directory to write generated .scs files")
    
    # --- Mode control ---
    mode_group = parser.add_argument_group('Operating Modes')
    mode_group.add_argument("--vstep", type=str, default="0.1", help="Voltage step (default 0.1) [SWEEP MODE]")
    mode_group.add_argument("--fixed-v", type=str, default=None, help="If set, generate TT/SS/FF each at this single voltage (e.g., 0.96) [FIXED-V MODE]")
    
    # --- Multi-pin functionality ---
    pin_group = parser.add_argument_group('Multi-pin Functionality')
    pin_group.add_argument("--pins", type=int, help="Number of input pins on the cell to test. Assumes the input signal is named 'in10'.")
    pin_group.add_argument("--pin-names", type=str, help="Comma-separated list of logical pin names (e.g., \"A,B\") corresponding to pin indices 1, 2, ...")
    pin_group.add_argument(
        "--base-lib-file", 
        type=str,
        default="../sg13g2_stdcell_typ_1p20V_25C.lib",
        help="Path to the *exact* .lib file to use for capacitance parsing. This single file will be used for all corners (tt, ss, ff)." 
    )

    #-- Cell mapping argument ---
    pin_group.add_argument(
        "--cell-map",
        type=str,
        help="Mapping from .scs cell names to .lib cell names. Format: \"SCS_NAME_1:LIB_NAME_1,SCS_NAME_2:LIB_NAME_2\"",
        required=False # Set to True if you always want to require it
    )
    
    # --- General options ---
    option_group = parser.add_argument_group('General Options')
    option_group.add_argument("--temps", type=str, default="0,50,125", help="Comma-separated temperatures (default: 0,50,125)")
    
    args = parser.parse_args()

    input_path: Path = args.input_netlist
    if not input_path.exists():
        raise FileNotFoundError(f"Input netlist not found: {input_path}")

    base_text = input_path.read_text()
    base_name = input_path.stem
    
    # --- Argument validation ---
    if args.pins and not args.pin_names:
        print("--- WARNING: --pins was specified, but --pin-names was not. ---")
        print("---         Capacitance replacement from .lib files will be SKIPPED. ---")
    
    pin_names_list: List[str] = []
    if args.pin_names:
        pin_names_list = [p.strip() for p in args.pin_names.split(',') if p.strip()]
        if args.pins and len(pin_names_list) != args.pins:
            raise ValueError(f"Mismatch: --pins was set to {args.pins}, but --pin-names provided {len(pin_names_list)} names. They must match.")

    # --- Parse cell map ---
    cell_map_dict: Optional[Dict[str, str]] = None
    if args.cell_map:
        try:
            cell_map_dict = dict(
                item.split(':', 1) for item in args.cell_map.split(',') if ':' in item.strip()
            )
            print(f"--- Loaded cell map: {cell_map_dict} ---")
        except ValueError:
            raise ValueError("Invalid format for --cell-map. Use: \"SCS_NAME_1:LIB_NAME_1,SCS_NAME_2:LIB_NAME_2\"")
    elif args.pins:
         print("--- WARNING: Running in multi-pin mode but --cell-map was not provided. Capacitance replacement will be skipped. ---")

    # --- Main logic ---
    load_versions = process_load_sections(base_text)
    
    if args.pins:
        # Multi-pin mode
        print(f"--- Running in Multi-pin mode for {args.pins} pins ---")
        for pin_num in range(1, args.pins + 1):
            pin_output_dir_base = args.output_dir / f"corner_pin{pin_num}"
            
            # Determine the logical pin name for this iteration
            logical_pin_name: Optional[str] = None
            if pin_names_list:
                try:
                    logical_pin_name = pin_names_list[pin_num - 1]
                except IndexError:
                    # This check is now redundant due to validation above, but good for safety
                    raise IndexError(f"Internal error: pin_num {pin_num} is out of bounds for pin_names_list.")

            # Iterate through the load versions (either just the original or gate/wire splits)
            for load_subdir, processed_text in load_versions:
                final_output_dir = pin_output_dir_base / load_subdir
                load_type_str = f" ({load_subdir})" if load_subdir else ""
                
                # Determine the primary input signal based on the load configuration
                primary_signal = 'in1' if load_subdir == 'gate_load' else 'in10'
                
                print(f"\n... Generating corners for pin {pin_num}{load_type_str} in '{final_output_dir}' (Signal: {primary_signal}, Logical Pin: {logical_pin_name}) ...")
                
                modified_for_pin_text, scs_master_cell_name = swap_pins_for_chain(processed_text, args.pins, primary_signal, pin_num)
                
                generate_corners(
                    modified_for_pin_text,
                    final_output_dir,
                    base_name,
                    args,
                    load_subdir,
                    pin_num,
                    scs_master_cell_name, 
                    logical_pin_name,
                    cell_map_dict 
                )
    else:
        # Single-pin mode
        for load_subdir, processed_text in load_versions:
            final_output_dir = args.output_dir / load_subdir
            load_type_str = f" ({load_subdir})" if load_subdir else ""

            print(f"\n--- Running in Single-file mode{load_type_str} in '{final_output_dir}' ---")
            generate_corners(
                processed_text,
                final_output_dir,
                base_name,
                args,
                load_subdir,
                0, # pin_num
                None, # scs_master_cell_name
                None, # logical_pin_name
                cell_map_dict # Pass the map (will be None/empty, so no-op)
            )

if __name__ == "__main__":
    main()

