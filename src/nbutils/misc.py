import os
import json
from typing import Dict, Any

def load_simulation_tree(root_path: str) -> Dict[str, Any]:
    """
    Recursively crawls a directory to build a dictionary mapping the file tree.
    
    This function traverses the provided directory path. Folder names become 
    keys in the returned dictionary, and JSON files are loaded as nested 
    dictionaries. Non-JSON files are ignored.
    
    Args:
        root_path: The absolute or relative path to the simulation data 
            directory (e.g., 'experiment10[...]').

    Returns:
        A nested dictionary where keys are directory or file names and 
        values are either further nested dictionaries (for subfolders) 
        or the parsed content of JSON files.

    Raises:
        FileNotFoundError: If the provided root_path does not exist.
        PermissionError: If the script lacks read permissions for the directory.
        
    Example:
        >>> data = load_simulation_tree("./data/tmax_5")
        >>> print(data['ZNEs']['ric4'].keys())
    """
    tree_dict = {}
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Path not found: {root_path}")

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        
        if os.path.isdir(item_path):
            # Recursively build the tree for sub-directories (tmax_X, ZNEs, etc.)
            tree_dict[item] = load_simulation_tree(item_path)
            
        elif item.endswith('.json'):
            with open(item_path, 'r', encoding='utf-8') as f:
                try:
                    # Use the filename without extension as the key
                    file_key = os.path.splitext(item)[0]
                    tree_dict[file_key] = json.load(f)
                except json.JSONDecodeError:
                    # Gracefully handle corrupted simulation logs
                    tree_dict[file_key] = {"error": "Invalid JSON format"}
                    
    return tree_dict