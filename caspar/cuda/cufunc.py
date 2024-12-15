# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import inspect
from pathlib import Path
from types import UnionType
from symforce.ops import LieGroupOps as Ops
import torch

from .lib_maker import add_function, call_generated
from .. import torch_tools as tt
from .lib_loader import CuLib


def as_symbolic(name: str, p: inspect.Parameter):
    if not isinstance(p.annotation, UnionType):
        return Ops.symbolic(p.annotation, name)
    else:
        return Ops.symbolic(p.annotation.__args__[0], name)


class CuFuncLib:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(exist_ok=True, parents=True)
        self.funcs: set["CuFunc"] = set()

    def register(self, func, **kwargs):
        cufunc = CuFunc(self, func, **kwargs)
        self.funcs.add(cufunc)
        return cufunc

    def indexed(self, func):
        return self.register(func, vec_idx=list(inspect.signature(func).parameters))

    def add(self, **kwargs):
        def inner(func):
            return self.register(func, **kwargs)

        return inner

    def load(self):
        self.culib = CuLib(
            self.workdir,
            cufiles=[self.workdir / f"{f.fname}.cu" for f in self.funcs],
        )


class CuFunc:

    def __init__(
        self,
        lib: CuFuncLib,
        func,
        name=None,
        **kwargs,
    ):
        self.lib = lib
        self.func = func

        self.fname = name or func.__name__

        self.sig = inspect.signature(self.func)
        self.fcode = f"/*\n{inspect.getsource(self.func)}\n*/\n"
        self.arg_types = {k: v.annotation for k, v in self.sig.parameters.items()}
        self.args = {k: Ops.symbolic(t, k) for k, t in self.arg_types.items()}
        self.arg_dims = {k: Ops.storage_dim(v) for k, v in self.args.items()}
        outs = self.func(*self.args.values())
        outs = outs if isinstance(outs, tuple) else (outs,)
        self.outs = {f"out{i}": v for i, v in enumerate(outs)}
        self.out_dims = {k: Ops.storage_dim(v) for k, v in self.outs.items()}
        self.double = kwargs.get("double", [])

        add_function(
            self.lib.workdir,
            self.fname,
            self.args,
            self.outs,
            scalar=[k for k, t in self.arg_types.items() if t in (float, int)],
            **kwargs,
        )

    def __call__(
        self,
        *args,
        prob_size,
        outs=None,
        vec_idx=[],
        **kwargs,
    ):

        if outs is None:
            gen = {False: tt.zeros_f32, True: tt.zeros_f64}
            outs = [
                gen[self.double == "all" or k in (self.double or [])](d, prob_size)
                for k, d in self.out_dims.items()
            ]
        binding = self.sig.bind(*args, **kwargs)

        prob_size = int(prob_size)

        kernel = self.lib.culib.get_kernel(self.fname)
        for arg in vec_idx:
            assert arg.dtype == torch.int32
        call_generated(
            kernel,
            binding.args,
            outs,
            prob_size=prob_size,
            vec_idx=vec_idx,
        )
        return outs[0] if len(outs) == 1 else outs
