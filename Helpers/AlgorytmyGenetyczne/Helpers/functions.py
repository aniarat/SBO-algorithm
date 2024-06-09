from benchmark_functions import BenchmarkFunction, Rastrigin, Schwefel


def rastrigin(n: int) -> BenchmarkFunction:
    return Rastrigin(n_dimensions=n)


def schwefel(n: int) -> BenchmarkFunction:
    return Schwefel(n_dimensions=n)
