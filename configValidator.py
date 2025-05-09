from typing import Dict


def validate_yml_config(config: Dict) -> bool:
    # Run section
    if "run" not in config or not isinstance(config["run"], str):
        raise ValueError("Missing or invalid 'run' key. It should be a string.")

    run_type = config["run"]
    if "nqubits" not in config or not isinstance(config["nqubits"], int):
        raise ValueError("Missing or invalid 'nqubits'. It should be an integer.")

    if "state" not in config or not isinstance(config["state"], str):
        raise ValueError("Missing or invalid 'state'. It should be a string.")

    # Output section
    if "output" not in config or not isinstance(config["output"], dict):
        raise ValueError("Missing or invalid 'output' section.")

    output = config["output"]
    if "file_name_prefix" not in output or not isinstance(output["file_name_prefix"], str):
        raise ValueError("Missing or invalid 'file_name_prefix' in 'output'. It should be a string.")

    if "draw" not in output or not isinstance(output["draw"], dict):
        raise ValueError("Missing or invalid 'draw' in 'output'.")
    draw = output["draw"]
    if "status" not in draw or not isinstance(draw["status"], bool):
        raise ValueError("Missing or invalid 'status' in 'draw'.")
    if "fig_dpi" not in draw or not isinstance(draw["fig_dpi"], int):
        raise ValueError("Missing or invalid 'fig_dpi' in 'draw'.")
    if "type" not in draw or not isinstance(draw["type"], str):
        raise ValueError("Missing or invalid 'type' in 'draw'.")

    # Observable section
    if "observable" not in config or not isinstance(config["observable"], dict):
        raise ValueError("Missing or invalid 'observable' section.")

    observable = config["observable"]
    if "def" not in observable or not isinstance(observable["def"], str):
        raise ValueError("Missing or invalid 'def' in 'observable'.")

    coeff = observable.get("coefficients", {})
    if not isinstance(coeff, dict):
        raise ValueError("'coefficients' in 'observable' must be a dictionary.")
    if "cn" not in coeff or not isinstance(coeff["cn"], list):
        raise ValueError("Missing or invalid 'cn' in 'coefficients'.")
    if "bn" not in coeff or not isinstance(coeff["bn"], list):
        raise ValueError("Missing or invalid 'bn' in 'coefficients'.")
    if "r" not in coeff or not isinstance(coeff["r"], (int, float)):
        raise ValueError("Missing or invalid 'r' in 'coefficients'.")

    # Ansatz section
    if "ansatz" not in config or not isinstance(config["ansatz"], dict):
        raise ValueError("Missing or invalid 'ansatz' section.")

    ansatz = config["ansatz"]
    if "type" not in ansatz["ugate"] or not isinstance(ansatz["ugate"]["type"], str):
        raise ValueError("Missing or invalid 'type' in 'ansatz'.")
    if "layer" not in ansatz or not isinstance(ansatz["layer"], int):
        raise ValueError("Missing or invalid 'layer' in 'ansatz'.")
    if "gateset" not in ansatz or not isinstance(ansatz["gateset"], int):
        raise ValueError("Missing or invalid 'gateset' in 'ansatz'.")

    if "ugate" not in ansatz or not isinstance(ansatz["ugate"], dict):
        raise ValueError("Missing or invalid 'ugate' in 'ansatz'.")

    ugate = ansatz["ugate"]
    if "coefficients" not in ugate or not isinstance(ugate["coefficients"], dict):
        raise ValueError("Missing or invalid 'coefficients' in 'ugate'.")
    if "time" not in ugate or not isinstance(ugate["time"], dict):
        raise ValueError("Missing or invalid 'time' in 'ugate'.")
    time = ugate["time"]
    if "min" not in time or not isinstance(time["min"], (int, float)):
        raise ValueError("Missing or invalid 'min' in 'ugate.time'.")
    if "max" not in time or not isinstance(time["max"], (int, float)):
        raise ValueError("Missing or invalid 'max' in 'ugate.time'.")

    # Noise section
    if "noise_profile" in config:
        noise = config["noise_profile"]
        if not isinstance(noise, dict):
            raise ValueError("Invalid 'noise_profile'. Should be a dictionary.")
        if "status" not in noise or not isinstance(noise["status"], bool):
            raise ValueError("Missing or invalid 'status' in 'noise_profile'.")
        if "type" not in noise or not isinstance(noise["type"], str):
            raise ValueError("Missing or invalid 'type' in 'noise_profile'.")
        if "noise_prob" not in noise or not isinstance(noise["noise_prob"], list):
            raise ValueError("Missing or invalid 'value' in 'noise_profile'.")
        init_param_noise = noise.get("noise_on_init_param", {})
        if "status" in init_param_noise and not isinstance(init_param_noise["status"], bool):
            raise ValueError("Invalid 'status' in 'noise_on_init_param'.")
        if "value" in init_param_noise and not isinstance(init_param_noise["value"], (int, float)):
            raise ValueError("Invalid 'value' in 'noise_on_init_param'.")

    # VQE section
    if "vqe" not in config or not isinstance(config["vqe"], dict):
        raise ValueError("Missing or invalid 'vqe' section.")

    vqe = config["vqe"]
    if "iteration" not in vqe or not isinstance(vqe["iteration"], int):
        raise ValueError("Missing or invalid 'iteration' in 'vqe'.")

    if "optimization" not in vqe or not isinstance(vqe["optimization"], dict):
        raise ValueError("Missing or invalid 'optimization' in 'vqe'.")

    opt = vqe["optimization"]
    if "status" not in opt or not isinstance(opt["status"], bool):
        raise ValueError("Missing or invalid 'status' in 'optimization'.")
    if "algorithm" not in opt or not isinstance(opt["algorithm"], str):
        raise ValueError("Missing or invalid 'algorithm' in 'optimization'.")
    if "constraint" not in opt or not isinstance(opt["constraint"], bool):
        raise ValueError("Missing or invalid 'constraint' in 'optimization'.")

    # Initial parameters
    if "init_param" not in config or not isinstance(config["init_param"], dict):
        raise ValueError("Missing or invalid 'init_param'.")
    param_val = config["init_param"].get("value")
    if not (isinstance(param_val, list) or isinstance(param_val, str)):
        raise ValueError("'init_param.value' should be a list or a string.")

    # Redundant section
    if "redundant" in config:
        redundant = config["redundant"]
        if "identity_factors" in redundant and not isinstance(redundant["identity_factors"], list):
            raise ValueError("'identity_factors' must be a list.")

    # ZNE section (validate only if "run" == "zne")
    if run_type == "zne":
        if "zne" not in config or not isinstance(config["zne"], dict):
            raise ValueError("Missing or invalid 'zne' section.")
        zne = config["zne"]
        if "method" not in zne or not isinstance(zne["method"], str):
            raise ValueError("Missing or invalid 'method' in 'zne'.")
        if "degree" not in zne or not isinstance(zne["degree"], int):
            raise ValueError("Missing or invalid 'degree' in 'zne'.")
        if "sampling" not in zne or not isinstance(zne["sampling"], str):
            raise ValueError("Missing or invalid 'sampling' in 'zne'.")
        if "data_points" in zne and not isinstance(zne["data_points"], list):
            raise ValueError("'data_points' in 'zne' should be a list.")

    return True
