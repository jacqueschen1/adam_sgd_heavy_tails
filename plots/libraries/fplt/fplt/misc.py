def percentage_to_index(p, L):
    """For a list of length L, returns the index closest to percentage p"""
    return int(p * (L - 1))


def dummy(*args, **kwargs):
    return None
