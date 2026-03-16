"""
VectorQuant Benchmark Suite Runner
===================================

Unified runner for all benchmarks with comprehensive reporting and visualization.
Run: python benchmarks/bench_runner.py
"""

import subprocess
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict


class BenchmarkRunner:
    """Orchestrates and runs all benchmarks."""
    
    def __init__(self):
        self.results = {}
        self.benchmark_dir = Path(__file__).parent
        self.timestamp = datetime.now().isoformat()
    
    def run_benchmark(self, script_name: str) -> bool:
        """Run a single benchmark script."""
        print(f"\n{'='*80}")
        print(f"Running: {script_name}")
        print(f"{'='*80}")
        
        script_path = self.benchmark_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                timeout=600  # 10 minute timeout
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"⏱️  Benchmark timeout: {script_name}")
            return False
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return False
    
    def collect_results(self) -> Dict:
        """Collect results from generated JSON files."""
        results = {
            "timestamp": self.timestamp,
            "benchmarks": {}
        }
        
        json_files = [
            ("bench_comprehensive_results.json", "comprehensive"),
            ("bench_performance_metrics.json", "metrics"),
            ("bench_speedup_analysis.json", "speedup_analysis"),
        ]
        
        for json_file, key in json_files:
            file_path = self.benchmark_dir / json_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        results["benchmarks"][key] = json.load(f)
                    print(f"✓ Loaded: {json_file}")
                except Exception as e:
                    print(f"❌ Could not load {json_file}: {e}")
        
        return results
    
    def generate_summary(self, results: Dict):
        """Generate summary report."""
        print("\n" + "="*80)
        print("BENCHMARK EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"Benchmarks Executed: {len(results['benchmarks'])}")
        
        # Check availability
        if 'comprehensive' in results['benchmarks']:
            avail = results['benchmarks']['comprehensive'].get('availability', {})
            print(f"\nComponent Availability:")
            print(f"  • C Backend: {'✓' if avail.get('c_backend') else '✗'}")
            print(f"  • NumPy: {'✓' if avail.get('numpy') else '✗'}")
            print(f"  • SciPy: {'✓' if avail.get('scipy') else '✗'}")
            print(f"  • QuantLib: {'✓' if avail.get('quantlib') else '✗'}")
        
        # Performance summary
        if 'metrics' in results['benchmarks']:
            print(f"\nPerformance Categories:")
            metrics = results['benchmarks']['metrics'].get('measurements', {})
            for category in metrics.keys():
                print(f"  • {category}")
        
        # Save comprehensive report
        report_path = self.benchmark_dir / "benchmark_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Full report saved: {report_path}")
    
    def run_all(self):
        """Run all benchmarks."""
        print("\n" + "="*80)
        print("VectorQuant Benchmark Suite")
        print("="*80)
        print(f"Start Time: {self.timestamp}")
        
        benchmarks = [
            "bench_comprehensive_comparison.py",
            "bench_performance_metrics.py",
            "bench_speedup_analysis.py",
        ]
        
        success_count = 0
        for benchmark in benchmarks:
            if self.run_benchmark(benchmark):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"Benchmark Execution Results: {success_count}/{len(benchmarks)} successful")
        print(f"{'='*80}")
        
        # Collect and report results
        results = self.collect_results()
        self.generate_summary(results)
        
        return success_count == len(benchmarks)


def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    success = runner.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
