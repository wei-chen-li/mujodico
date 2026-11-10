"""IP-PAS: FATROP's interior-point algorithm in JAX with a PAS KKT backend.

``IpPasSolver`` (``ippas.py``) is the fully jitted, FATROP-faithful solver
used by ``IppasLiftedMultipleShooting``.  ``FatropReferenceJAX``
(``solver.py``) is the earlier host-driven reference port, kept for
comparison.
"""
from .ippas import IpPasOptions, IpPasSolver, STATUS_NAMES
from .solver import FatropReferenceJAX, FatropRefOptions, FatropRefResult
from .linear import LinearSolution, solve_serial, solve_pas

__all__ = [
    "IpPasSolver",
    "IpPasOptions",
    "STATUS_NAMES",
    "FatropReferenceJAX",
    "FatropRefOptions",
    "FatropRefResult",
    "LinearSolution",
    "solve_serial",
    "solve_pas",
]
