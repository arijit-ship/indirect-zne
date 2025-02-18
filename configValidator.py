from typing import Dict


def validate_yml_config(config: Dict) -> bool:
    # Run section
    if "run" not in config or not isinstance(config["run"], str):
        raise ValueError("Missing or invalid 'run' key. It should be a string.")

    # Nqubits section
    if "nqubits" not in config or not isinstance(config["nqubits"], int):
        raise ValueError("Missing or invalid 'nqubits'. It should be an integer.")

    if "state" not in config or not isinstance(config["state"], str):
        raise ValueError("Missing or invalid 'state'. It should be a string.")

    # Target Hamiltonian section
    if "observable" not in config or not isinstance(config["observable"], dict):
        raise ValueError("Missing or invalid 'observable' section.")

    if "def" not in config["observable"] or not isinstance(config["observable"]["def"], str):
        raise ValueError("Missing or invalid 'def' in 'observable'. It should be a string.")

    if "coefficients" not in config["observable"] or not isinstance(config["observable"]["coefficients"], dict):
        raise ValueError("Missing or invalid 'coefficients' in 'observable'. It should be a dictionary.")

    coefficients = config["observable"]["coefficients"]
    if "cn" not in coefficients or not isinstance(coefficients["cn"], list):
        raise ValueError("Missing or invalid 'cn' in 'coefficients'. It should be a list.")

    if "bn" not in coefficients or not isinstance(coefficients["bn"], list):
        raise ValueError("Missing or invalid 'bn' in 'coefficients'. It should be a list.")

    if "r" not in coefficients or not isinstance(coefficients["r"], (int, float)):
        raise ValueError("Missing or invalid 'r' in 'coefficients'. It should be a number.")

    # Output section
    if "output" not in config or not isinstance(config["output"], dict):
        raise ValueError("Missing or invalid 'output' section.")

    if "file_name_prefix" not in config["output"] or not isinstance(config["output"]["file_name_prefix"], str):
        raise ValueError("Missing or invalid 'file_name_prefix' in 'output'. It should be a string.")

    if "draw" not in config["output"] or not isinstance(config["output"]["draw"], dict):
        raise ValueError("Missing or invalid 'draw' in 'output'. It should be a dictionary.")

    if "status" not in config["output"]["draw"] or not isinstance(config["output"]["draw"]["status"], bool):
        raise ValueError("Missing or invalid 'status' in 'draw'. It should be a boolean.")

    if "fig_dpi" not in config["output"]["draw"] or not isinstance(config["output"]["draw"]["fig_dpi"], int):
        raise ValueError("Missing or invalid 'fig_dpi' in 'draw'. It should be an integer.")

    if "type" not in config["output"]["draw"] or not isinstance(config["output"]["draw"]["type"], str):
        raise ValueError("Missing or invalid 'type' in 'draw'. It should be a string.")

    # VQE section
    if "vqe" not in config or not isinstance(config["vqe"], dict):
        raise ValueError("Missing or invalid 'vqe' section.")

    if "iteration" not in config["vqe"] or not isinstance(config["vqe"]["iteration"], int):
        raise ValueError("Missing or invalid 'iteration' in 'vqe'. It should be an integer.")

    if "optimization" not in config["vqe"] or not isinstance(config["vqe"]["optimization"], dict):
        raise ValueError("Missing or invalid 'optimization' in 'vqe'. It should be a dictionary.")

    if "status" not in config["vqe"]["optimization"] or not isinstance(config["vqe"]["optimization"]["status"], bool):
        raise ValueError("Missing or invalid 'status' in 'optimization'. It should be a boolean.")

    if "algorithm" not in config["vqe"]["optimization"] or not isinstance(
        config["vqe"]["optimization"]["algorithm"], str
    ):
        raise ValueError("Missing or invalid 'algorithm' in 'optimization'. It should be a string.")

    if "constraint" not in config["vqe"]["optimization"] or not isinstance(
        config["vqe"]["optimization"]["constraint"], bool
    ):
        raise ValueError("Missing or invalid 'constraint' in 'optimization'. It should be a boolean.")

    if "ansatz" not in config["vqe"] or not isinstance(config["vqe"]["ansatz"], dict):
        raise ValueError("Missing or invalid 'ansatz' in 'vqe'. It should be a dictionary.")

    if "type" not in config["vqe"]["ansatz"] or not isinstance(config["vqe"]["ansatz"]["type"], str):
        raise ValueError("Missing or invalid 'type' in 'ansatz'. It should be a string.")

    if "layer" not in config["vqe"]["ansatz"] or not isinstance(config["vqe"]["ansatz"]["layer"], int):
        raise ValueError("Missing or invalid 'layer' in 'ansatz'. It should be an integer.")

    if "gateset" not in config["vqe"]["ansatz"] or not isinstance(config["vqe"]["ansatz"]["gateset"], int):
        raise ValueError("Missing or invalid 'gateset' in 'ansatz'. It should be an integer.")

    if "ugate" not in config["vqe"]["ansatz"] or not isinstance(config["vqe"]["ansatz"]["ugate"], dict):
        raise ValueError("Missing or invalid 'ugate' in 'ansatz'. It should be a dictionary.")

    ugate = config["vqe"]["ansatz"]["ugate"]
    if "coefficients" not in ugate or not isinstance(ugate["coefficients"], dict):
        raise ValueError("Missing or invalid 'coefficients' in 'ugate'. It should be a dictionary.")

    if "time" not in ugate or not isinstance(ugate["time"], dict):
        raise ValueError("Missing or invalid 'time' in 'ugate'. It should be a dictionary.")

    # Check if init_param is either a list or a string
    if "init_param" not in config["vqe"]["ansatz"]:
        raise ValueError("Missing 'init_param' in 'ansatz'.")
    init_param = config["vqe"]["ansatz"]["init_param"]
    if not (isinstance(init_param, list) or isinstance(init_param, str)):
        raise ValueError("Invalid 'init_param' in 'ansatz'. It should be either a list or a string.")

    # Redundant circuit section
    if "identity_factors" in config and config["identity_factors"] is not None:
        if not isinstance(config["identity_factors"], list):
            raise ValueError("Invalid 'identity_factors'. It should be a list or None.")

    # ZNE section
    if "zne" not in config or not isinstance(config["zne"], dict):
        raise ValueError("Missing or invalid 'zne' section.")

    if "method" not in config["zne"] or not isinstance(config["zne"]["method"], str):
        raise ValueError("Missing or invalid 'method' in 'zne'. It should be a string.")

    if "degree" not in config["zne"] or not isinstance(config["zne"]["degree"], int):
        raise ValueError("Missing or invalid 'degree' in 'zne'. It should be an integer.")

    if "sampling" not in config["zne"] or not isinstance(config["zne"]["sampling"], str):
        raise ValueError("Missing or invalid 'sampling' in 'zne'. It should be a string.")

    # Check if 'data_points' is None or a list
    if "data_points" in config["zne"]:
        data_points = config["zne"]["data_points"]
        if data_points is not None and not isinstance(data_points, list):
            raise ValueError("Invalid 'data_points' in 'zne'. It should be a list or None.")

    # If everything passes
    return True
