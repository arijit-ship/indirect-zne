# vqe-mcore.py
import os
import sys
import time
import yaml
import subprocess


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 vqe-mcore.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    num_runs = config["vqe"]["iteration"]

    print(f"Launching {num_runs} VQE processes...")
    start_time = time.time()

    # Launch all processes simultaneously
    processes = []
    for i in range(num_runs):
        p = subprocess.Popen(
            [sys.executable, "main.py", config_file],
        )
        processes.append((i, p))
        print(f"VQE #{i+1} started (PID: {p.pid})")

    # Wait for all to finish
    for i, p in processes:
        p.wait()
        if p.returncode != 0:
            print(f"VQE #{i+1} FAILED (PID: {p.pid})")
        else:
            print(f"VQE #{i+1} done (PID: {p.pid})")

    total_time = time.time() - start_time
    print(f"All {num_runs} runs done in {total_time:.2f} sec")