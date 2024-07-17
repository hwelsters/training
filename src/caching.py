import os

import numpy as np

def make_path_if_not_exists(path: str) -> None:
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

def cache_numpy_array(array: np.ndarray, path: str) -> None:
    make_path_if_not_exists(path)

    with open(path, "wb") as file:
        np.save(file, array)

def load_cached_numpy_array(path: str) -> np.ndarray:
    with open(path, "rb") as file:
        return np.load(file)

def is_numpy_array_cached(path: str) -> bool:
    return os.path.exists(path)