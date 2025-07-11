import subprocess
import time

def benchmark_run(description, command):
    print(f"--- {description} ---")
    start = time.perf_counter()
    subprocess.run(command, shell=True, check=True)
    elapsed = time.perf_counter() - start
    print(f"{description} completed in {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    # Adjust python executable if needed
    script = "netmonster_mapper.py"

    print("Benchmarking SignalTopo pipeline with netmonster_sample.csv")
    print("Ensure netmonster_sample.csv is in the working directory.\n")

    benchmark_run("SignalTopo Full Run", f"python {script}")
