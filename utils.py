from time import perf_counter
from contextlib import contextmanager


@contextmanager
def timeit(name):
    print(f'[{name}] Starting...')
    start = perf_counter()
    yield
    end = perf_counter()
    print(f'[{name}] Finished in {end - start:.2f}s.')
