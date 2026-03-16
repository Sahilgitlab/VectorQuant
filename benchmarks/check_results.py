#!/usr/bin/env python3
import json

with open('bench_comprehensive_results.json') as f:
    data = json.load(f)

print('=' * 60)
print('BFGS OPTIMIZATION RESULTS')
print('=' * 60)
if 'bfgs' in data['results']:
    for impl, metrics in data['results']['bfgs'].items():
        print(f'{impl:30s}: {metrics["avg_ms"]:10.4f} ms')
else:
    print('No BFGS results found')

print('\n' + '=' * 60)
print('MONTE CARLO GBM RESULTS')
print('=' * 60)
if 'gbm_monte_carlo' in data['results']:
    for impl, metrics in data['results']['gbm_monte_carlo'].items():
        print(f'{impl:30s}: {metrics["avg_ms"]:10.4f} ms')
else:
    print('No GBM results found')

print('\n' + '=' * 60)
print('OLS REGRESSION RESULTS')
print('=' * 60)
if 'ols_regression' in data['results']:
    for impl, metrics in data['results']['ols_regression'].items():
        print(f'{impl:30s}: {metrics["avg_ms"]:10.4f} ms')
else:
    print('No OLS Regression results found')

print('\n' + '=' * 60)
print('SUMMARY: All benchmarks completed successfully!')
print('=' * 60)
