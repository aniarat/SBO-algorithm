def binatodeci(binary):
    return sum(val * (2 ** idx) for idx, val in enumerate(reversed(binary)))


def binary_to_decimal(binary_chain, start=-10, end=10):
    chain_len = len(binary_chain)
    return start + binatodeci(binary_chain) * (end - start) / (2 ** chain_len - 1)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def get_actual_values(solution, decode_start, decode_end, num_of_dimensions):
    return [binary_to_decimal(
             spec
             , decode_start
             , decode_end) for spec in list(split(solution, num_of_dimensions))]