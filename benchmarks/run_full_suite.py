
import subprocess
import os
import sys

scripts = [
    "bench_linear_algebra.py",
    "bench_statistics.py",
    "bench_monte_carlo.py",
    "bench_quant_models.py",
    "bench_sparse.py",
    "bench_ai_verification.py",
    "bench_gpu.py"
]

def run_all():
    print("="*60)
    print("VECTORQUANT PRODUCTION BENCHMARK SUITE")
    print("="*60)
    
    for script in scripts:
        print(f"\n>>> Executing {script}...")
        path = os.path.join(os.path.dirname(__file__), script)
        # We need to set PYTHONPATH to include the project root
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        try:
            subprocess.run([sys.executable, path], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("\n" + "="*60)
    print("FULL SUITE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    run_all()
