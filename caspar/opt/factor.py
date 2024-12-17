# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import inspect
from pathlib import Path

import numpy as np
from symforce import symbolic as sf
from symforce.ops import LieGroupOps as Ops
from symforce.ops.interfaces import LieGroup
import torch

from ..cuda.lib_loader import CuLib
from ..cuda.lib_maker import add_function, call_generated


def make_tril_list(mat):
    indices = ((a, b) for a in range(mat.shape[0]) for b in range(a + 1))
    return [mat[a, b] for a, b in indices]


class FactorLib:

    def __init__(self, workdir: Path, order: list[LieGroup]):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)

        self.order = order = {t: i for i, t in enumerate(order)}

        self.factors: set["Factor"] = set()
        self.types: set[LieGroup] = set()

    def register(self, res_fn, jac_fn=None, consts=None):
        factor = Factor(self, res_fn, jac_fn=jac_fn, consts=consts)
        self.factors.add(factor)
        return factor

    def register_typ(self, typ: LieGroup):
        self.types.add(typ)

    def with_consts(self, *consts: str):
        def inner(res_fn):
            return self.register(res_fn, consts=consts)

        return inner

    def load(self):
        for typ in self.types:
            # Retractor
            sym = Ops.symbolic(typ, typ.__name__)
            delta = sf.Matrix(Ops.tangent_dim(typ), 1).symbolic("delta")
            add_function(
                self.workdir,
                f"{typ.__name__}_retract",
                {typ.__name__: sym, f"{typ.__name__}_d": delta},
                {"out": Ops.retract(sym, delta)},
                develop=True,
                # double="all",
            )
            # Normalizer
            mat = sf.Matrix(Ops.tangent_dim(typ), Ops.tangent_dim(typ)).symbolic("mat")
            tril = [(a, b) for a in range(Ops.tangent_dim(typ)) for b in range(a + 1)]
            precond_stored = [mat[a, b] for a, b in tril]
            for a, b in tril:
                mat[b, a] = mat[a, b]
            njtr_sym = sf.Matrix(Ops.tangent_dim(typ), 1).symbolic("vec")
            diag = sf.symbols("diag")
            safe: sf.Matrix = (
                mat.multiply_elementwise(mat.eye() * diag + 1)
                + mat.eye() * sf.epsilon()
            )
            normalized = sf.Matrix(safe.mat.solve(njtr_sym.mat, method="LU"))
            add_function(
                self.workdir,
                f"{typ.__name__}_precondition",
                {"precond": precond_stored, "njtr": njtr_sym, "diag": diag},
                {"normalized": normalized},
                unique=["diag"],
                # double="all",
            )

        self.culib = CuLib(self.workdir)

    def retract_type(self, type, args, outs, prob_size):
        kernel = self.culib.get_kernel(f"{type.__name__}_retract")
        return call_generated(kernel, args, outs, prob_size)

    def normalize_type(self, type, args, outs, prob_size):
        kernel = self.culib.get_kernel(f"{type.__name__}_precondition")
        return call_generated(kernel, args, outs, prob_size)

    def sort_types(self, types: list[type]) -> list[LieGroup]:
        return sorted(types, key=self.order.__getitem__)

    def argsort_types(self, types: list[type]):
        type_order = [self.order[t] for t in types]
        return sorted(range(len(type_order)), key=type_order.__getitem__)

    def sort_pairs(self, pairs: list[tuple[type, type]]):
        return tuple(sorted(pairs, key=lambda x: (self.order[x[0]], self.order[x[1]])))

    def get_pair_sort(self, types: list[type]):
        ord_pairs = self.get_pair_argsort(types)
        return [(types[a], types[b]) for a, b in ord_pairs]

    def get_pair_argsort(self, types: list[type]):
        args = self.argsort_types(types)
        ord_pairs = [(a, b) for i, a in enumerate(args) for b in args[: i + 1]]
        return ord_pairs


class Factor:

    def __init__(self, lib: FactorLib, res_fn, jac_fn=None, consts=None):
        self.lib = lib
        self.res_fn = res_fn
        self.jac_fn = jac_fn

        assert jac_fn is None, "Jacobian not implemented"
        self.consts = consts or []

        self.sig = inspect.signature(self.res_fn)
        assert all([c in self.sig.parameters for c in self.consts])

        annotations = {k: v.annotation for k, v in self.sig.parameters.items()}
        self.args: dict[str, LieGroup] = {}
        self.args = {k: Ops.symbolic(v, k) for k, v in annotations.items()}
        self.arg_types = {k: v for k, v in annotations.items()}
        self.arg_dims = {k: Ops.storage_dim(v) for k, v in self.args.items()}

        self.nodes = {k: self.args[k] for k in annotations if k not in self.consts}
        self.node_types = {k: v for k, v in annotations.items() if k not in self.consts}
        self.node_sdims = {k: Ops.storage_dim(v) for k, v in self.nodes.items()}
        self.node_tdims = {k: Ops.tangent_dim(v) for k, v in self.nodes.items()}
        self.consts = {k: self.args[k] for k in annotations if k in self.consts}
        res_vals = Ops.to_storage(self.res_fn(**self.args))
        res = sf.Matrix([r for r in res_vals])
        self.res_dim = Ops.storage_dim(res)

        self.arg_lg = min(self.nodes, key=lambda x: -self.node_tdims[x])
        # JACOBIAN

        jacs = {k: res.jacobian(node) for k, node in self.nodes.items()}

        node_types = [v for k, v in annotations.items() if k not in self.consts]
        node_names = list(self.nodes.keys())
        pair_args = self.lib.get_pair_argsort(node_types)
        self.jtj_type_pairs = [(node_types[a], node_types[b]) for a, b in pair_args]
        self.jtj_arg_pairs = [(node_names[a], node_names[b]) for a, b in pair_args]

        types_args = self.lib.argsort_types(node_types)
        self.jtr_types = [node_types[i] for i in types_args]
        self.jtr_args = [node_names[i] for i in types_args]
        scalars = [k for k, t in self.arg_types.items() if t in (float, int)]

        precond_lg = jacs[self.arg_lg].T * jacs[self.arg_lg]
        njtr_lg = -jacs[self.arg_lg].T * res

        add_function(
            self.lib.workdir,
            f"{self.res_fn.__name__}_linearize_and_more",
            {**self.args},
            {
                "res": res,
                **{f"{a}_jac": v for a, v in jacs.items()},
                **{f"{a}_jac_sorted": v for a, v in jacs.items() if a != self.arg_lg},
                "precond_lg": make_tril_list(precond_lg),
                "njtr_lg": njtr_lg,
            },
            vec_idx=[*self.nodes],
            atomic_add=["precond_lg", "njtr_lg"],
            vec_idx_map={"njtr_lg": "precond_lg"},
            vec_idx_unq=[*(f"{a}_jac_sorted" for a in jacs if a != self.arg_lg)],
            scalar=scalars,
            # double="all",
        )
        # Preconditioner
        res_sym = Ops.symbolic(res, "res")
        jac_sym = {k: Ops.symbolic(v, f"{k}_jac") for k, v in jacs.items()}
        zeros = [(k, i) for (k, j) in jacs.items() for (i, e) in enumerate(j) if e == 0]

        njtr = {a: -jac_sym[a].T * res_sym for a in self.nodes}
        for arg, typ in self.node_types.items():
            precond = jac_sym[arg].T * jac_sym[arg]
            tril = [(a, b) for a in range(precond.shape[0]) for b in range(a + 1)]
            precond_stored = [precond[a, b] for a, b in tril]
            add_function(
                self.lib.workdir,
                f"{self.res_fn.__name__}_precond_njtr_{arg}",
                {
                    "res": res_sym,
                    f"{arg}_jac": jac_sym[arg],
                },
                {
                    "precond": precond_stored,
                    "njtr": njtr[arg],
                },
                atomic_add=["precond", "njtr"],
                vec_idx_map={"njtr": "precond"},
                vec_idx_unq=["res"],
                # double="all",
            )

        # JnJTr
        njtr_sym = {f"{a}_njtr": v.symbolic(f"{a}_njtr") for a, v in njtr.items()}
        jnjtr = sum(
            [jac * njtr_s for jac, njtr_s in zip(jac_sym.values(), njtr_sym.values())],
            start=sf.Matrix(self.res_dim, 1),
        )
        jtjnjtr_lg = sf.Matrix(jac_sym[self.arg_lg].T * jnjtr)
        add_function(
            self.lib.workdir,
            f"{self.res_fn.__name__}_JnJTr",
            {
                **njtr_sym,
                **{f"{k}_jac": v for k, v in jac_sym.items()},
            },
            {
                "jnjtr": jnjtr,
                "jtjnjtr_lg": jtjnjtr_lg,
            },
            vec_idx=[*njtr_sym],
            atomic_add=["jtjnjtr_lg"],
            # double="all",
        )
        # JTJnJTr
        for arg, typ in self.node_types.items():
            jnjtr_sym = sf.Matrix(self.res_dim, 1).symbolic("jnjtr")
            jtjnjtr = sf.Matrix(jac_sym[arg].T * jnjtr_sym)
            add_function(
                self.lib.workdir,
                f"{self.res_fn.__name__}_JTJnJTr_{arg}",
                {
                    "jnjtr": jnjtr_sym,
                    f"{arg}_jac": jac_sym[arg],
                },
                {"jtjnjtr": jtjnjtr},
                atomic_add=["jtjnjtr"],
                vec_idx_unq=["jnjtr"],
                # double="all",
            )
        # Score
        add_function(
            self.lib.workdir,
            f"{self.res_fn.__name__}_score",
            {**self.args},
            {"rtr": res.T * res},
            vec_idx=[*self.nodes.keys()],
            unique=["rtr"],
            atomic_add=["rtr"],
            scalar=scalars,
            # double="all",
        )
        for typ in node_types:
            self.lib.register_typ(typ)

    def __call__(self, *args, **kwargs):
        bound = self.sig.bind(*args, **kwargs)

        for ten in bound.arguments.values():
            assert ten.is_cuda and ten.stride()[-1] == 1
        for k in self.nodes:
            assert bound.arguments[k].dtype == torch.int32
        # for k in self.consts:
        #     assert bound.arguments[k].dtype == torch.float32
        return self, bound.arguments

    def do_full(self, args, outs, prob_size, vec_idx=[]):
        kernel = self.lib.culib.get_kernel(f"{self.res_fn.__name__}_linearize_and_more")
        return call_generated(kernel, args, outs, prob_size, vec_idx)

    def do_precond_njtr(self, arg, args, outs, prob_size, vec_idx=[]):
        kernel = self.lib.culib.get_kernel(f"{self.res_fn.__name__}_precond_njtr_{arg}")
        return call_generated(kernel, args, outs, prob_size, vec_idx)

    def do_JnJTr(self, args, outs, prob_size, vec_idx=[]):
        kernel = self.lib.culib.get_kernel(f"{self.res_fn.__name__}_JnJTr")
        return call_generated(kernel, args, outs, prob_size, vec_idx)

    def do_JTJnJTr(self, arg, args, outs, prob_size, vec_idx=[]):
        kernel = self.lib.culib.get_kernel(f"{self.res_fn.__name__}_JTJnJTr_{arg}")
        return call_generated(kernel, args, outs, prob_size, vec_idx)

    def do_score(self, args, outs, prob_size, vec_idx=[]):
        kernel = self.lib.culib.get_kernel(f"{self.res_fn.__name__}_score")
        return call_generated(kernel, args, outs, prob_size, vec_idx)

    def __repr__(self):
        return f"Factor_{self.res_fn.__name__}"
