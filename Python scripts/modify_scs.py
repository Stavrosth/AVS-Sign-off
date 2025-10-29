import argparse
import re

def modify_scs(input_file, output_file, new_cell, stable_inputs):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        stripped = line.strip()
        # Only modify lines that start with "cell"
        if stripped.startswith("ncell"):
            tokens = stripped.split()
            # Replace last token (cell name)
            if len(tokens) > 1:
                tokens[-1] = new_cell

                # Replace stable input(s)
                for i, t in enumerate(tokens):
                    if t.lower() == "vss":  # original stable input
                        new_stables = stable_inputs.split()
                        tokens = tokens[:i] + new_stables + tokens[i+1:]
                        break

                line = " ".join(tokens) + "\n"

        modified_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(modified_lines)

    print(f"✅ Created '{output_file}' with cell '{new_cell}' and stable inputs '{stable_inputs}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify .scs file: change only 'cell' lines (cell name and stable inputs).")
    parser.add_argument("--input", required=True, help="Input .scs file")
    parser.add_argument("--output", required=True, help="Output .scs file")
    parser.add_argument("--cell", required=True, help="New cell name (e.g., NAND2X1)")
    parser.add_argument("--stable_inputs", required=True, help="Stable input names separated by spaces (e.g., 'vss' or 'vss vdd')")
    args = parser.parse_args()

    modify_scs(args.input, args.output, args.cell, args.stable_inputs)
