# vqe-mcore.py
import os
import sys
import time
import yaml
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_single_vqe(args):
    i, config_file = args
    result = subprocess.run(
        [sys.executable, "main.py", config_file],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"VQE #{i+1} FAILED:\n{result.stderr}")
        return {"index": i, "success": False}
    print(f"VQE #{i+1} done.")
    return {"index": i, "success": True}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 vqe-mcore.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    num_runs = config["vqe"]["iteration"]
    num_workers = os.cpu_count()

    print(f"Running {num_runs} VQE runs across {num_workers} workers...")
    start_time = time.time()

    args_list = [(i, config_file) for i in range(num_runs)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(run_single_vqe, args): args[0] for args in args_list}
        for future in as_completed(futures):
            future.result()

    total_time = time.time() - start_time
    print(f"All {num_runs} runs done in {total_time:.2f} sec")