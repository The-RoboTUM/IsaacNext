"""Utility functions for tendons."""


def list_from_dict(d: dict, n: int) -> list:
    """Convert a dict of lists to a list of lists."""
    assert (
        min(d.keys()) == 0 and max(d.keys()) == n - 1 and len(set(d.keys())) == n
    ), "Dict keys must be consecutive integers starting from 0."
    return [d[k] for k in sorted(d.keys())]
