import numpy as np

from typing import List


def decompose(total: int, max_num_terms: int) -> List[int]:
    """ Represent an integer as a sum of approximately equal terms.
    
    Parameters
    ----------
    total : int >= 0
        An integer to decompose into a sum of positive terms.
    max_num_terms : int > 0
        The number of terms to use, if possible.

    Returns
    -------
    terms : list of integers > 0
        A list of nearly equal integer terms (differing at most by 1)
        that add up to the given total. The largest number of terms
        consistent with the given maximum is used.

    Examples
    --------
    >>> decompose(10, 4)
    [2, 2, 3, 3]
    >>> decompose(3, 3)
    [1, 1, 1, 1]
    >>> decompose(0, 5)
    []
    """
    if not np.issubdtype(type(total), np.integer):
        raise ValueError("`total` must be int, got %r" % (total,))
    if total < 0:
        raise ValueError("`total` must be nonnegative, got %r" % (total,))
    if not np.issubdtype(type(max_num_terms), np.integer):
        raise ValueError("Cannot decompose an integer into a sum of %r terms"
                         % (max_num_terms,))
    if max_num_terms < 1:
        raise ValueError("Cannot decompose an integer into a sum of %r terms"
                         % (max_num_terms,))
    if total <= max_num_terms:
        return [1 for _ in range(total)]
    small_size = total // max_num_terms
    big_size = small_size + 1
    num_big = total - small_size * max_num_terms
    num_small = max_num_terms - num_big
    return num_small*[small_size] + num_big*[big_size]


def random_splits(num_episodes: int, max_num_splits: int) -> List[np.ndarray]:
    """ Return different choices of validation episode indices.
    
    Parameters
    ----------
    num_episodes : int >= 0
        A number of episodes to split into approximately equally-sizes
        sets of episodes (largest permitted difference in size is 1).
    max_num_splits int >= 1
        The largest permitted number of sets to use. When `num_episodes`
        is large, this will also be the actual number of sets.
    
    Returns
    -------
    segments : list of integer arrays
        A list of approximately equal-length integer arrays that contain
        all of the integers in `range(num_episodes)` exactly once.

    Examples
    --------
    >>> random_splits(10, 3)
    [array([0, 5, 2]), array([1, 7, 6]), array([8, 3, 9, 4])]
    >>> random_splits(10, 3)
    [array([2, 5, 4]), array([8, 1, 7]), array([0, 9, 6, 3])]
    >>> random_splits(10, 3)
    [array([3, 5, 7]), array([9, 1, 4]), array([6, 0, 8, 2])]
    """
    if num_episodes < 2:
        return []
    chunk_sizes = decompose(num_episodes, max_num_splits)
    endpoints = np.cumsum(chunk_sizes).tolist()
    assert endpoints.pop(-1) == num_episodes
    randomized = np.random.permutation(num_episodes)
    return np.split(randomized, endpoints)


def test_that_decompose_returns_numbers_that_add_up_to_total() -> None:

    for total in np.random.randint(0, 20, size=10):
        for max_terms in np.random.randint(1, 10, size=10):
            terms = decompose(total, max_terms)
            assert sum(terms) == total, (total, max_terms)


def test_that_random_splits_contain_all_indices_exactly_once():

    for total in np.random.randint(0, 20, size=10):
        for max_terms in np.random.randint(1, 10, size=10):
            segments = random_splits(total, max_terms)
            assert sorted(np.concatenate(segments)) == list(range(total))

