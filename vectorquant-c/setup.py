from setuptools import setup, Extension
import os

# Use OpenMP if available
extra_compile_args = []
extra_link_args = []

if os.name == 'nt':  # Windows
    # Add /arch:AVX2 for SIMD and /fp:fast for performance
    extra_compile_args = ['/openmp', '/arch:AVX2', '/fp:fast']
else:
    extra_compile_args = ['-fopenmp', '-mavx2', '-ffast-math']
    extra_link_args = ['-lgomp']

core_ext = Extension(
    'vectorquant_c_core',
    sources=[
        'src/core.c',
        'src/linalg.c',
        'src/stats.c',
        'src/stochastic.c',
        'src/optimization.c',
        'src/fft.c',
        'src/qmc.c',
        'src/autodiff.c',
        'src/kalman.c',
        'src/sparse.c'
    ],
    include_dirs=['include'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='vectorquant_c',
    version='0.1.0',
    ext_modules=[core_ext],
)
