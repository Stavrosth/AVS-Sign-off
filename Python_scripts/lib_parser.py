"""
lib_parser.py

Author: Stavros Stathoudakis 
Mentor: Akis Tsoumanis  
Supervising Professor: Christos Sotiriou  
Inspired by Christian Lütkemeyer

Helper module for parsing capacitance values from .lib files.
"""

import re
import functools
import os
from typing import Optional, Dict, Tuple

# --- Caching ---
# Cache the content of .lib files to avoid re-reading the same large file
@functools.lru_cache(maxsize=5)
def read_lib_file(lib_path_str: str) -> Optional[str]:
    """
    Reads a .lib file from the given path.
    Returns the content as a string or None if an error occurs.
    """
    if not os.path.exists(lib_path_str):
        print(f"--- ERROR: .lib file not found at: {lib_path_str}")
        return None
    try:
        print(f"--- Reading .lib file: {lib_path_str}")
        with open(lib_path_str, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"--- ERROR: Could not read .lib file at {lib_path_str}: {e}")
        return None

# --- Path Generation ---
# This function is simplified to just return the provided path.
# The script will use the same file for all corners.

def get_lib_path(base_lib_path_str: str, proc_key: str) -> Optional[str]:
    """
    Returns the exact .lib file path provided.
    The 'proc_key' argument is unused but kept for API compatibility.
    """
    if not os.path.exists(base_lib_path_str):
        print(f"--- ERROR: ({proc_key}) .lib file not found at: {base_lib_path_str}")
        return None
    
    # Return the exact path given.
    return base_lib_path_str


# --- Parsing Logic ---
CAP_REGEX = re.compile(r"capacitance\s*:\s*([0-9\.]+)\s*;", re.IGNORECASE)

def find_block(content: str, block_type: str, block_name: str, start_index: int = 0) -> Optional[Tuple[str, int]]:
    """
    Finds a specific named block (e.g., "cell (name)") in the text
    and returns its content and its end index.
    
    This uses a brace-balancing algorithm instead of regex to handle
    nested structures correctly.
    """

    # Added a space between block_type and the opening parenthesis
    search_str = f"{block_type} ({block_name})"
    
    try:
        # Find the start of the block declaration, e.g., "pin (A)"
        block_start_idx = content.index(search_str, start_index)
        
        # Find the opening brace '{' after the declaration
        brace_open_idx = content.index('{', block_start_idx)
        
        brace_level = 1
        search_idx = brace_open_idx + 1
        content_len = len(content)
        
        # Scan character by character to find the matching closing brace
        while search_idx < content_len:
            char = content[search_idx]
            if char == '{':
                brace_level += 1
            elif char == '}':
                brace_level -= 1
            
            if brace_level == 0:
                # Found the matching brace
                block_content = content[brace_open_idx + 1 : search_idx]
                return block_content, search_idx + 1
                
            search_idx += 1
            
        print(f"--- PARSE ERROR: Could not find closing brace for {search_str} starting at index {block_start_idx}.")
        return None
        
    except ValueError:
        # The search_str (e.g., "pin (A)") was not found
        return None

def parse_capacitance(lib_content: str, cell_name: str, pin_name: str) -> Optional[str]:
    """
    Parses the .lib file content to find the capacitance for a specific cell and pin.
    """
    try:
        # 1. Find the cell block
        cell_block_result = find_block(lib_content, "cell", cell_name)
        if not cell_block_result:
            print(f"--- PARSE ERROR: Cell '{cell_name}' not found in .lib file.")
            return None
        
        cell_content, _ = cell_block_result
        
        # 2. Find the pin block within the cell block
        pin_block_result = find_block(cell_content, "pin", pin_name)
        if not pin_block_result:
            print(f"--- PARSE ERROR: Pin '{pin_name}' not found in cell '{cell_name}'.")
            return None
            
        pin_content, _ = pin_block_result
        
        # 3. Find the capacitance value within the pin block
        cap_match = CAP_REGEX.search(pin_content)
        if not cap_match:
            print(f"--- PARSE ERROR: 'capacitance' not found for pin '{pin_name}' in cell '{cell_name}'.")
            return None
            
        return cap_match.group(1)
        
    except Exception as e:
        print(f"--- ERROR during .lib parsing for {cell_name}/{pin_name}: {e}")
        return None

