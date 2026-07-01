"""
sdm_library — four modular, reusable SDM implementations from the thesis.

Usage:
    from sdm_library import StandardSDM, RankOrderSDM, WheelSDM, RDLIFSDM
    from sdm_library import make_binary_nofm, make_significance_vectors

All four share the same interface:
    sdm.write(address_vec, data_vec)
    sdm.read(address_vec)  -> raw output vector
    sdm.similarity(a, b)   -> normalised dot product
    sdm.capacity_test(address_codes, data_codes) -> dict

Variants:
    (a) StandardSDM   — binary N-of-M, OR weights, thesis Chapter 3
    (b) RankOrderSDM  — significance vectors, MAX weights, thesis Chapter 4
    (c) WheelSDM      — (b) but decode via wheel-model spiking neurons, Ch 7
    (d) RDLIFSDM      — (b) but via RDLIF spiking neurons, Ch 6 / 2005 papers
"""

from .base import (
    SDMBase,
    make_binary_nofm,
    make_significance_vectors,
    normalised_dot_product,
    topk_indices,
)
from .standard_sdm import StandardSDM
from .rankorder_sdm import RankOrderSDM
from .wheel_sdm import WheelSDM
from .rdlif_sdm import RDLIFSDM

__all__ = [
    "SDMBase", "StandardSDM", "RankOrderSDM", "WheelSDM", "RDLIFSDM",
    "make_binary_nofm", "make_significance_vectors",
    "normalised_dot_product", "topk_indices",
]
