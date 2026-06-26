import argparse
import json
import os
import sys
import yaml


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Convert a JSON configuration file into a customized YAML file."
    )

    # Positional argument for the input file path
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the input JSON file (e.g., input_file.json).",
    )

    # Optional argument for the output file path/name
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="config.yaml",
        help="Custom name/path for the output YAML file (default: config.yaml).",
    )

    args = parser.parse_args()

    # 1. Validate that the input file exists
    if not os.path.exists(args.json_path):
        print(f"❌ Error: The file '{args.json_path}' does not exist.")
        sys.exit(1)

    try:
        # 2. Open and parse the JSON file
        with open(args.json_path, "r", encoding="utf-8") as json_file:
            raw_data = json.load(json_file)

        # 3. Extract the configuration block (handles files wrapped in a top-level 'config' key)
        config_data = (
            raw_data["config"] if "config" in raw_data else raw_data
        )

        # 4. Write out the clean, structured YAML file using your customized path
        with open(args.output, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                config_data,
                yaml_file,
                default_flow_style=False,  # Ensures a clean block style layout
                sort_keys=False,  # Maintains the same order of keys as the JSON
            )

        print(
            f"🎉 Successfully converted '{args.json_path}' ➡️ '{args.output}'"
        )

    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse '{args.json_path}' as valid JSON.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()