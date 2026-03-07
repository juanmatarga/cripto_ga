#!/usr/bin/env python3
"""Run evolve → validate → OTS for multiple seeds."""

import subprocess
import sys
import time

SEEDS = [42, 123, 777]
CONFIG = 'config_v2.yaml'

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}\n")
    t0 = time.time()

    # Evolve
    print(f"[seed={seed}] Evolving...")
    result = subprocess.run(
        [sys.executable, 'main_v2.py', 'evolve', '--config', CONFIG, '--seed', str(seed)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[seed={seed}] Evolution FAILED:\n{result.stderr[-500:]}")
        continue

    # Find the results dir from output
    lines = result.stdout.strip().split('\n')
    results_dir = None
    for line in lines:
        if 'Results saved to' in line:
            results_dir = line.split('Results saved to')[-1].strip()
            break

    if not results_dir:
        print(f"[seed={seed}] Could not find results dir in output")
        # Try to find the most recent directory
        import glob
        dirs = sorted(glob.glob(f'results/experiment_seed{seed}_*'))
        if dirs:
            results_dir = dirs[-1]
        else:
            print(f"[seed={seed}] No results directory found, skipping")
            continue

    print(f"[seed={seed}] Results: {results_dir}")

    # Validate
    print(f"[seed={seed}] Validating...")
    result = subprocess.run(
        [sys.executable, 'main_v2.py', 'validate', '--config', CONFIG,
         '--seed', str(seed), '--results', results_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[seed={seed}] Validation FAILED:\n{result.stderr[-500:]}")
        continue

    # Print validation summary
    for line in result.stdout.strip().split('\n'):
        if 'PASS' in line or 'FAIL' in line or 'passed all' in line:
            print(f"  {line.strip()}")

    # OTS
    print(f"[seed={seed}] OTS evaluation...")
    result = subprocess.run(
        [sys.executable, 'main_v2.py', 'ots', '--config', CONFIG,
         '--seed', str(seed), '--results', results_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[seed={seed}] OTS FAILED:\n{result.stderr[-500:]}")
        continue

    # Print OTS summary
    for line in result.stdout.strip().split('\n'):
        if 'OTS:' in line or 'Strategies evaluated' in line or 'no_alpha' in line:
            print(f"  {line.strip()}")

    elapsed = time.time() - t0
    print(f"[seed={seed}] Done in {elapsed:.0f}s")

print(f"\n{'='*60}")
print("ALL EXPERIMENTS COMPLETE")
print(f"{'='*60}")
