


def all_equal(iterable) -> bool:
    """ Returns true if all values in iterable are equal """
    return len(set(iterable)) <= 1

def mean(iterable) -> float:
    """ Returns the mean of the given iterable """
    if not hasattr(iterable, "__len__"):
        iterable = list(iterable)
    return sum(iterable) / len(iterable) if len(iterable) > 0 else 0