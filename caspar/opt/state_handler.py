# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from itertools import pairwise
from typing import Type
import numpy as np
import torch
from symforce.ops.interfaces import LieGroup
from symforce.ops import LieGroupOps as Ops

from .. import torch_tools as tt

from caspar import FactorLib


from .params import SolverParams

ltype = Type[LieGroup]


class StateHandler:
    def __init__(
        self,
        flib: FactorLib,
        typ2storage: dict[ltype, torch.Tensor],
        params: "SolverParams",
    ):
        self.flib = flib
        for v in typ2storage.values():
            assert v.is_cuda and v.is_contiguous()

        dtype = torch.float64 if params.use_double else torch.float32

        types_sorted = flib.sort_types(list(typ2storage.keys()))
        type_pairs = flib.get_pair_sort(types_sorted)
        # INIT TYPES
        self.typ2storage0 = {k: typ2storage[k].to(dtype) for k in types_sorted}
        self.typ2storage1 = {
            t: torch.empty_like(v) for t, v in self.typ2storage0.items()
        }

        self.typ2n_vals = {t: v.size(1) for t, v in self.typ2storage0.items()}
        self.typ2tan_dim = {t: Ops.tangent_dim(t) for t in self.typ2storage0}

        # INIT TAN OFFSETS
        offs = [Ops.tangent_dim(t) * n for t, n in self.typ2n_vals.items()]
        tan_offs = torch.from_numpy(np.cumsum([0, *offs]))
        tan_slices = [slice(a, b) for a, b in pairwise(tan_offs)]

        self.typ2tan_slice = {t: stic for t, stic in zip(self.typ2n_vals, tan_slices)}

        self.tan_size = int(tan_offs[-1].item())
        self.jtj_sizes = {
            p: (self.typ2n_vals[p[0]], self.typ2n_vals[p[1]]) for p in type_pairs
        }
        self.jtr_sizes = self.typ2n_vals

    def do_retract(self, typ2step_tn: dict[ltype, torch.Tensor]):
        for typ, storage in self.typ2storage0.items():
            self.flib.retract_type(
                typ,
                args=[
                    storage,
                    typ2step_tn[typ],
                ],
                outs=[self.typ2storage1[typ]],
                prob_size=storage.shape[1],
            )

    def swap_active_other(self):
        self.typ2storage0, self.typ2storage1 = (
            self.typ2storage1,
            self.typ2storage0,
        )
