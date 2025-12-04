"""Vector utility functions."""
from typing import Iterable, List
import math


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a_list = list(vec_a)
    b_list = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("Vectors must have equal length")
    dot = sum(a * b for a, b in zip(a_list, b_list))
    norm_a = math.sqrt(sum(a * a for a in a_list))
    norm_b = math.sqrt(sum(b * b for b in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize(vec: Iterable[float]) -> List[float]:
    a_list = list(vec)
    norm = math.sqrt(sum(a * a for a in a_list))
    if norm == 0:
        return a_list
    return [a / norm for a in a_list]
