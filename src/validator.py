"""
Validator module for validating the configuration of the application.
"""


def validate_noiseprofile(noise_profile: dict) -> bool:
    """
    Validates the noise profile configuration.

    Args:
        noise_profile (dict): Dictionary containing noise profile.

    Returns:
        bool: True if the noise profile is valid, False otherwise.
    """
    # Required top-level keys
    required_keys = ["status", "type", "noise_prob", "noise_on_init_param"]

    for key in required_keys:
        if key not in noise_profile:
            print(f"Missing required key: {key}")
            return False

    if not isinstance(noise_profile["status"], bool):
        print("'status' must be a boolean.")
        return False

    if noise_profile["type"] not in ["Depolarizing", "BitFlip", "Dephasing", "IndependentXZ", "AmplitudeDamping"]:
        print(f"Invalid 'type': {noise_profile['type']}")
        return False

    noise_prob = noise_profile["noise_prob"]
    if not isinstance(noise_prob, list) or len(noise_prob) != 4:
        print("'noise_prob' must be a list of length 4.")
        return False
    if not all(isinstance(p, (int, float)) for p in noise_prob):
        print("'noise_prob' must contain numeric values.")
        return False

    # Validate noise_on_init_param
    init_param = noise_profile["noise_on_init_param"]
    if not isinstance(init_param, dict):
        print("'noise_on_init_param' must be a dictionary.")
        return False

    if "status" not in init_param or not isinstance(init_param["status"], bool):
        print("Missing or invalid 'status' in 'noise_on_init_param'.")
        return False

    if "value" not in init_param or not isinstance(init_param["value"], (int, float)):
        print("Missing or invalid 'value' in 'noise_on_init_param'.")
        return False

    return True
