# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from dataclasses import dataclass
import time
from typing import Type
from pprint import pprint
import torch
from symforce.ops.interfaces import LieGroup


from .factor import Factor
from .res_collection import ResidualCollection
from .state_handler import StateHandler
from ..cuda.cufunc import CuFuncLib
from .. import FactorLib, torch_tools as tt
from caspar.cuda.kernel import Kernel
from .params import SolverParams


class Solver:

    def __init__(
        self,
        flib: FactorLib,
        typ2storage: dict[Type[LieGroup], torch.Tensor],
        factors: list[tuple[Factor, dict[str, torch.Tensor]]],
        params: SolverParams = SolverParams(),
        constraints: list[tuple[Factor, dict[str, torch.Tensor]]] = [],
    ):
        # TODO: handle constraints
        # TODO: use streams and graph capture
        t0 = time.perf_counter()
        print(f"\nSOLVER_SETUP")

        self.flib = flib
        self.params = params
        fac2arg2data = dict(factors)

        self.states = StateHandler(flib, typ2storage, params)
        self.facs = ResidualCollection(flib, self.states, fac2arg2data, params)
        pprint(params)
        print()
        print(f"{'Parameters:':15} {sum(t.numel() for t in typ2storage.values()):10,}")
        print(f"{'Residuals:':15} {self.facs.res_data.numel():10,}")
        print(f"{'Minimizer:':15} Levenberg-Marquardt")
        print(f"{'Linear solvers:':15} CGNR")
        print(f"{'Preconditioner:':15} BLOCK_JACOBI (not available in ceres)")
        print(f"{'Preprocessor:':15} {time.perf_counter() - t0:.6f} seconds\n")

    def solve(self):
        print(f"\nSOLVER_SOLVE")
        Kernel.reset_timing_data()
        t0 = time.perf_counter()
        t_prev = t0

        best_score = self.facs.do_score()
        self.facs.diag[0] = self.params.diag_init
        diag_min = tt.tensor_f32([self.params.diag_min])
        up_scale = self.params.up_scale
        step_quality = 0
        down_scale = tt.tensor_f32([self.params.down_scale])
        self.facs.do_full()
        vnames = [
            "best score",
            "new score",
            "difference",
            "pcg rel err",
            "step quality",
            "dampening",
        ]
        tnames = ["t step", "t total"]
        print(
            f"\n  iter"
            + "".join(map("{:>15s}".format, vnames))
            + f"{'pcg iter':>10s}"
            + "".join(map("{:>10s}".format, tnames))
        )
        print(f"{-1:6}{best_score:15.6e}")
        for i in range(self.params.max_iter):

            step, diff_guess = self.facs.do_pcg(i, step_quality)
            self.states.do_retract(step)
            self.states.swap_active_other()

            new_score = self.facs.do_score()
            diff = new_score - best_score
            step_quality = diff / diff_guess

            if accept := new_score < best_score * (1 - self.params.min_rel_decrease):
                self.facs.do_full()
                best_score = new_score
                if self.params.adapt_scale:
                    scale = down_scale.maximum(1.0 - (2.0 * step_quality - 1.0) ** 3)
                else:
                    scale = down_scale
                self.facs.diag[0] = diag_min.maximum(self.facs.diag * scale)
                up_scale = self.params.up_scale
            else:
                self.facs.diag[0] *= up_scale
                up_scale *= self.params.up_scale_exp
                self.states.swap_active_other()
            # mse = (self.facs.res_data.reshape(2, -1) ** 2).sum(0).mean().sqrt()
            t_now = time.perf_counter()
            vstuff = [
                best_score,
                new_score,
                diff,
                self.facs.pcg_rel,
                step_quality,
                self.facs.diag[0],
            ]
            tstuff = [
                t_now - t_prev,
                t_now - t0,
            ]

            print(
                f"{'*' if accept else ' '}{i: 5}"
                + "".join(map("{: 15.6e}".format, vstuff))
                + f"{self.facs.inner_iter:10d}"
                + "".join(map("{: 10.6f}".format, tstuff))
            )
            t_prev = t_now

            if (
                self.facs.diag[0] > self.params.end_diag
                or new_score < self.params.end_score
            ):
                break
        print(f"\nFinished in: {time.perf_counter() - t0:.6f} seconds\n")
        return self.states.typ2storage0

    def print_stats(self):
        print(f"\nKERNEL STATS")

        torch.cuda.synchronize()
        name_len = max(len(name) for name in Kernel._kernels)
        import numpy as np

        things = ["n calls", "total (s)", "mean (s)", "min (s)", "max (s)"]
        print(f"\n{'kernel':^{name_len}s}" + "".join(f"{t:>12s}" for t in things))
        gpu_total = 0
        strings = {}
        for name, k in Kernel._kernels.items():
            if not k.event_pairs:
                continue
            times = [a.elapsed_time(b) * 0.001 for a, b in k.event_pairs]
            tot = np.sum(times)
            mean = np.mean(times)
            amin = np.amin(times)
            amax = np.amax(times)
            gpu_total += tot
            strings[tot] = (
                f"{name:<{name_len}s}"
                f"{len(times):12d}{tot: 12.3e}{mean: 12.3e}{amin: 12.3e}{amax: 12.3e}"
            )
        for s in sorted(strings, reverse=True):
            print(strings[s])
        print(f"\nTotal symbolic kernel time (s):{gpu_total: .3e}\n")
