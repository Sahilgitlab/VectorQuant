"""
Multiprocessing Monte Carlo Engine
"""
import multiprocessing

def _generic_worker(args):
    """
    Helper for multiprocessing pool.
    args: (func, kwargs)
    """
    func, kwargs = args
    return func(**kwargs)

def parallel_simulate_paths(func, n_paths, n_jobs=None, **kwargs):
    """
    Generic parallel dispatcher for stochastic processes.
    Distributes paths across multiple CPUs to bypass Python global lock on math ops.
    """
    if n_jobs is None:
        try:
            n_jobs = multiprocessing.cpu_count()
        except NotImplementedError:
            n_jobs = 4
            
    # Divide paths roughly equally
    base_chunk = n_paths // n_jobs
    remainder = n_paths % n_jobs
    
    chunks = []
    for i in range(n_jobs):
        chunk_size = base_chunk + (1 if i < remainder else 0)
        if chunk_size > 0:
            worker_kwargs = dict(kwargs)
            worker_kwargs['n_paths'] = chunk_size
            chunks.append((func, worker_kwargs))
            
    # Run parallel
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(_generic_worker, chunks)
        
    if not results:
        return []
        
    # Flatten results
    # Handle tuple returns (like Heston returning s_paths, v_paths)
    if isinstance(results[0], tuple):
        num_returns = len(results[0])
        combined = [[] for _ in range(num_returns)]
        for res in results:
            for i in range(num_returns):
                combined[i].extend(res[i])
        return tuple(combined)
    else:
        all_paths = []
        for res in results:
            all_paths.extend(res)
        return all_paths
