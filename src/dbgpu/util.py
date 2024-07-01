from __future__ import annotations

from re import sub, IGNORECASE
from typing import Any, Iterable, List, Tuple, Union
from itertools import islice
from datetime import datetime, date

__all__ = [
    "safe_name",
    "chunk_iterable",
    "reduce_units",
    "json_serialize",
]

def safe_name(input_str: str) -> str:
    """
    Convert a name to a safe, lowercase, trimmed string.

    >>> safe_name("Hello, World!")
    'hello-world'
    >>> safe_name("  Hello, World!  ")
    'hello-world'
    >>> safe_name("[40GB] PCI-E 3.0 x16")
    '40gb-pci-e-3-0-x16'
    >>> safe_name("[40 GB] PCI-E 3.0 x16")
    '40gb-pci-e-3-0-x16'
    """
    # Remove leading and trailing whitespace
    input_str = input_str.strip()

    # Standardize "GB" and "GiB" to remove spaces between numbers and these units
    input_str = sub(r'(\d+)[\s\-\._,]+(GB|GiB)', r'\1\2', input_str, flags=IGNORECASE)

    # Remove special characters except spaces and alphanumeric characters
    input_str = sub(r'[^a-zA-Z0-9\s]', ' ', input_str)

    # Replace multiple spaces with a single space
    input_str = sub(r'\s+', ' ', input_str)

    # Remove leading and trailing whitespace
    input_str = input_str.strip()

    # Replace spaces with dashes
    input_str = sub(r'\s', '-', input_str)

    # Convert to lowercase
    input_str = input_str.lower()

    return input_str

def chunk_iterable(
    iterable: Iterable[Any],
    chunk_size: int,
    pad_to_size: bool = False,
    pad_with: Any = None,
) -> Iterable[List[Any]]:
    """
    Split an iterable into chunks of a given size.

    >>> list(chunk_iterable(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(chunk_iterable(range(10), 3, pad_to_size=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
    >>> list(chunk_iterable(range(10), 3, pad_to_size=True, pad_with='x'))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 'x', 'x']]
    """
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            break
        if pad_to_size and len(chunk) < chunk_size:
            chunk += [pad_with] * (chunk_size - len(chunk))
        yield chunk


def reduce_units(
    value: Union[int, float],
    units: List[str],
    base: int = 1000,
    threshold: int = 1000,
) -> Tuple[float, str]:
    """
    Reduce a value to the smallest unit possible.

    >>> reduce_units(4e9, ["bytes/s", "kb/s", "mb/s", "gb/s"])
    (4.0, 'gb/s')
    """
    for unit in units:
        if value < threshold:
            break
        value /= base
    return value, unit


def json_serialize(obj: Any) -> Any:
    """
    Serialize an object for json.dumps, enabling date formatting.
    """
    if isinstance(obj, datetime) or isinstance(obj, date):
        return obj.isoformat()
    return obj
