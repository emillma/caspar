# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from dataclasses import dataclass


@dataclass
class SolverParams:

    # Dampening
    diag_init: float = 1.0
    diag_min: float = 1e-5
    up_scale: float = 3.0
    up_scale_exp: float = 2.0
    down_scale: float = 1 / 3
    adapt_scale: bool = True

    # Step acceptance
    min_rel_decrease: float = 1e-3

    # Exit conditions
    max_iter: int = 1000
    end_score: float = 1e-5
    end_diag: float = 1e3

    # Preconditioned Conjugate Gradient
    pcg_rel_tol: float = 0.5
    pcg_max_iter: int = 10

    use_double: bool = False

    def __post_init__(self):
        assert self.use_double is False
