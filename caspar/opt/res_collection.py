# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from itertools import pairwise

import numpy as np
import torch
from typing import TYPE_CHECKING
from .factor import Factor
from .. import torch_tools as tt

from symforce.ops.interfaces import LieGroup
from symforce.ops import LieGroupOps as Ops

from caspar import FactorLib

from .params import SolverParams

if TYPE_CHECKING:
    from .state_handler import StateHandler


def offs_slice(offs, idx):
    return slice(offs[idx], offs[idx + 1])


def diag_indices(typ: LieGroup):
    tdim = typ.tangent_dim()
    r = tt.arange_i32(tdim)
    return r * (tdim + 1)


class ResidualCollection:

    def __init__(
        self,
        flib: "FactorLib",
        states: "StateHandler",
        fac2arg2data: dict[Factor, dict[str, torch.Tensor]],
        params: "SolverParams",
    ):
        self.flib = flib
        self.states = states
        self.params = params
        dtype = torch.float64 if params.use_double else torch.float32

        def allocate(*args):
            return tt.empty_f64(*args) if params.use_double else tt.empty_f32(*args)

        fac2map = {
            fac: torch.argsort(arg2data[fac.arg_lg])
            for fac, arg2data in fac2arg2data.items()
        }

        self.fac2arg2idx = {
            f: {a: d[fac2map[f]] for a, d in a2d.items() if a not in f.consts}
            for f, a2d in fac2arg2data.items()
        }
        self.fac2arg2const = {
            f: {a: d[:, fac2map[f]].to(dtype) for a, d in a2d.items() if a in f.consts}
            for f, a2d in fac2arg2data.items()
        }
        self.fac2arg2jac_elem: dict[Factor, dict[str, torch.Tensor]] = {}
        self.fac2arg2jac_ord: dict[Factor, dict[str, torch.Tensor]] = {}
        self.fac2arg2res_idx: dict[Factor, dict[str, torch.Tensor]] = {}
        for f, a2i in self.fac2arg2idx.items():
            for a, idx in a2i.items():
                sorted, argsort = idx.sort()
                self.fac2arg2res_idx.setdefault(f, {})[a] = argsort.int()
                ord = tt.empty_i32(argsort.numel())
                ord[argsort] = tt.arange_i32(argsort.numel())
                if a != f.arg_lg:
                    self.fac2arg2jac_ord.setdefault(f, {})[a] = ord
                unq, inv = sorted.unique_consecutive(return_inverse=True)
                self.fac2arg2jac_elem.setdefault(f, {})[a] = unq[inv].int()

        # N CALLS
        self.fac2n_calls = {
            f: next(iter(t.values())).numel() for f, t in self.fac2arg2idx.items()
        }
        # SCORE
        self.score = allocate(1)
        self.denum = allocate(1)
        self.ratio = allocate(1)
        self.diag = allocate(1)

        # Jac
        self.fac2arg2jac = {
            f: {
                a: allocate(Ops.tangent_dim(t) * f.res_dim, n)
                for a, t in f.node_types.items()
            }
            for f, n in self.fac2n_calls.items()
        }
        self.fac2arg2jac_sorted = {
            f: {
                a: torch.empty_like(j)
                for a, j in self.fac2arg2jac[f].items()
                if a != f.arg_lg
            }
            for f, n in self.fac2n_calls.items()
        }
        # Preconditioner
        self.typ2precond = {
            t: allocate(Ops.tangent_dim(t) * (Ops.tangent_dim(t) + 1) // 2, n)
            for t, n in self.states.typ2n_vals.items()
        }

        # Residual
        r_sizes = np.cumsum([0, *(f.res_dim * n for f, n in self.fac2n_calls.items())])
        r_slices = {f: slice(*p) for f, p in zip(self.fac2n_calls, pairwise(r_sizes))}
        self.res_data = allocate(r_sizes[-1])
        self.fac2res = {
            f: self.res_data[s].reshape(f.res_dim, -1) for f, s in r_slices.items()
        }
        self.jnjtr_data = allocate(r_sizes[-1])
        self.fac2jnjtr = {
            f: self.jnjtr_data[s].reshape(f.res_dim, -1) for f, s in r_slices.items()
        }
        # self.fac2res = {f: allocate(f.res_dim, n) for f, n in self.fac2n_calls.items()}
        # self.fac2jnjtr = {
        #     f: allocate(f.res_dim, n) for f, n in self.fac2n_calls.items()
        # }

        def njtr_like():
            data = allocate(states.tan_size)
            typ2view = {
                t: data[s].reshape(Ops.tangent_dim(t), -1)
                for t, s in states.typ2tan_slice.items()
            }
            return data, typ2view

        self.njtr_data, self.typ2njtr = njtr_like()
        self.z_data, self.typ2z = njtr_like()
        self.w_data, self.typ2w = njtr_like()
        self.x_data, self.typ2x = njtr_like()
        self.err_data, self.typ2err = njtr_like()
        self.p_data, self.typ2p = njtr_like()
        self.rprev_data, self.typ2rprev = njtr_like()

        self.beta_denum = allocate(1)

        self.pcg_rel = 1.0

    def arg2data(self, fac: Factor):

        nodes = {k: self.states.typ2storage0[v] for k, v in fac.node_types.items()}
        return {**nodes, **self.fac2arg2const[fac]}

    def do_full(self):
        for ten in [*self.typ2njtr.values(), *self.typ2precond.values()]:
            ten.zero_()

        for fac, prob_size in self.fac2n_calls.items():
            fac.do_full(
                args=self.arg2data(fac).values(),
                outs=[
                    self.fac2res[fac],
                    *self.fac2arg2jac[fac].values(),
                    *self.fac2arg2jac_sorted[fac].values(),
                    self.typ2precond[fac.node_types[fac.arg_lg]],
                    self.typ2njtr[fac.node_types[fac.arg_lg]],
                ],
                vec_idx=[
                    *self.fac2arg2idx[fac].values(),
                    self.fac2arg2jac_elem[fac][fac.arg_lg],
                    *self.fac2arg2jac_ord[fac].values(),
                ],
                prob_size=prob_size,
            )

            for arg in (arg for arg in fac.nodes if arg != fac.arg_lg):
                fac.do_precond_njtr(
                    arg,
                    args=[
                        self.fac2res[fac],
                        self.fac2arg2jac_sorted[fac][arg],
                    ],
                    outs=[
                        self.typ2precond[fac.node_types[arg]],
                        self.typ2njtr[fac.node_types[arg]],
                    ],
                    vec_idx=[
                        self.fac2arg2jac_elem[fac][arg],
                        self.fac2arg2res_idx[fac][arg],
                    ],
                    prob_size=prob_size,
                )

    def normalize(self, typ2data_in, typ2data_out):
        for typ, out in typ2data_out.items():
            self.flib.normalize_type(
                typ,
                args=[self.typ2precond[typ], typ2data_in[typ], self.diag],
                outs=[out],
                prob_size=self.states.typ2n_vals[typ],
            )
        return typ2data_out

    def do_JTJx(self, typ2data_in, typ2data_out):
        for fac, prob_size in self.fac2n_calls.items():
            fac.do_JnJTr(
                args=[
                    *[typ2data_in[t] for t in fac.node_types.values()],
                    *self.fac2arg2jac[fac].values(),
                ],
                outs=[
                    self.fac2jnjtr[fac],
                    typ2data_out[fac.node_types[fac.arg_lg]],
                ],
                vec_idx=[
                    *self.fac2arg2idx[fac].values(),
                    self.fac2arg2jac_elem[fac][fac.arg_lg],
                ],
                prob_size=prob_size,
            )

            for arg in (a for a in fac.nodes if a != fac.arg_lg):
                fac.do_JTJnJTr(
                    arg,
                    args=[
                        self.fac2jnjtr[fac],
                        self.fac2arg2jac_sorted[fac][arg],
                    ],
                    outs=[
                        typ2data_out[fac.node_types[arg]],
                    ],
                    vec_idx=[
                        self.fac2arg2jac_elem[fac][arg],
                        self.fac2arg2res_idx[fac][arg],
                    ],
                    prob_size=prob_size,
                )
        return typ2data_out

    def do_pcg(self, iteration: int, prev_step_quality: float):
        init_norm = self.njtr_data.norm()

        self.normalize(self.typ2njtr, self.typ2z)

        self.w_data[:] = self.diag * self.njtr_data
        self.do_JTJx(self.typ2z, self.typ2w)

        numerator = self.njtr_data @ self.z_data
        if iteration == 0 or True:
            self.p_data[:] = self.z_data
        else:
            # trust = (2.0 * step_quality - 1.0) ** 3
            beta = 0.1 * numerator / self.beta_denum
            self.p_data[:] = self.z_data + beta * self.p_data

        denum = self.p_data @ self.w_data
        alpha = numerator / denum
        assert alpha.isfinite().all()

        self.x_data[:] = alpha * self.p_data
        self.err_data[:] = self.njtr_data - alpha * self.w_data
        self.beta_denum[:] = numerator

        for self.inner_iter in range(self.params.pcg_max_iter):
            self.pcg_rel = self.err_data.norm() / init_norm
            if self.pcg_rel < self.params.pcg_rel_tol:
                break

            self.normalize(self.typ2err, self.typ2z)
            numerator = self.err_data @ self.z_data
            beta = numerator / self.beta_denum
            self.p_data[:] = self.z_data + beta * self.p_data

            self.w_data[:] = self.diag * self.err_data
            self.do_JTJx(self.typ2p, self.typ2w)
            denum = self.z_data @ self.w_data
            alpha = numerator / denum
            if not alpha.isfinite().all():
                break
            self.x_data += alpha * self.p_data
            self.err_data -= alpha * self.w_data
            self.beta_denum[:] = numerator

        pred_decrease = -(self.x_data @ (self.err_data + self.njtr_data)) * 0.5
        return self.typ2x, pred_decrease

    def do_score(self):
        self.score.zero_()

        for fac, prob_size in self.fac2n_calls.items():
            fac.do_score(
                args=self.arg2data(fac).values(),
                outs=[self.score],
                vec_idx=[*self.fac2arg2idx[fac].values()],
                prob_size=prob_size,
            )

        return self.get_score()

    def get_score(self):
        return self.score.item() * 0.5
