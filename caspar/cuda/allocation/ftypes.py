# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from collections import Counter
import random
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Type

from symengine.lib import symengine_wrapper

import symforce.symbolic as sf


@dataclass(eq=False)
class Var:
    func: "Func"
    idx: int = field(default=0)
    # contribs: set["Func"] = field(default_factory=set)
    missing_contribs: Counter["Func"] = field(default_factory=Counter)

    def set_func(self, func: "Func") -> None:
        object.__setattr__(self, "func", func)

    def is_const(self) -> bool:
        return self.func.is_store()

    def __hash__(self) -> int:
        return hash((self.func, self.idx))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and hash(self) == hash(other)

    def __repr__(self) -> str:
        return str(self.func) + (f"[{self.idx}]" if self.idx else "")


class Func:
    args: tuple[Var, ...]
    outs: list[Var]
    n_outs = 1
    data: float | int | str | None = None

    missing_args: set[Var]
    acc_count: int

    _hash: int | None = None

    def __init__(self, *args: Var, data: Any = None, outs: list[Var] | None = None) -> None:
        self.args = args
        self.data = data
        self.missing_args = set()
        self.acc_count = 0

        assert isinstance(data, (float, int, str)) or data is None
        if outs is None:
            self.outs = [Var(self, i) for i in range(self.n_outs)]
        else:
            for out in outs:
                out.set_func(self)
            self.outs = outs

        assert isinstance(self.args, tuple)
        # assert isinstance(self.outs, tuple)
        assert all([isinstance(arg, Var) for arg in self.args])

    @property
    def n_args(self) -> int:
        return len(self.args)

    def rebind(self, var: Var, idx: int = 0) -> None:
        var.func = self
        var.idx = idx
        self.outs[idx] = var

    def __getitem__(self, idx: int) -> Var:
        return self.outs[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({','.join(map(str, self.args))})"

    def __hash__(self) -> int:
        if self._hash is None:
            if isinstance(self, Store):
                self._hash = random.randint(0, 1 << 64)
            else:
                self._hash = hash((self.__class__, self.args, self.data))
        return self._hash

    def __eq__(self, other: "Func") -> bool:
        return isinstance(other, Func) and hash(self) == hash(other)

    def is_write(self) -> bool:
        return isinstance(self, Write)

    def is_load(self) -> bool:
        return isinstance(self, Read)

    def is_store(self) -> bool:
        return isinstance(self, Store)

    def is_sum(self) -> bool:
        return isinstance(self, Sum)

    def is_minus(self) -> bool:
        return isinstance(self, Minus)

    def is_prod(self) -> bool:
        return isinstance(self, Prod)

    def is_div(self) -> bool:
        return isinstance(self, Div)

    def is_sincos(self) -> bool:
        return isinstance(self, SinCos)

    def is_cos(self) -> bool:
        return isinstance(self, Cos)

    def is_sin(self) -> bool:
        return isinstance(self, Sin)

    def is_norm(self) -> bool:
        return isinstance(self, Norm)

    def is_rnorm(self) -> bool:
        return isinstance(self, RNorm)

    def is_pow(self) -> bool:
        return isinstance(self, Pow)

    def is_square(self) -> bool:
        return isinstance(self, Square)

    def is_rcp(self) -> bool:
        return isinstance(self, Rcp)

    def is_sqrt(self) -> bool:
        return isinstance(self, Sqrt)

    def is_rsqrt(self) -> bool:
        return isinstance(self, RSqrt)

    def is_cbrt(self) -> bool:
        return isinstance(self, Cbrt)

    def is_rcbrt(self) -> bool:
        return isinstance(self, RCbrt)

    def is_squeeze(self) -> bool:
        return isinstance(self, Squeeze)

    def is_acc(self) -> bool:
        return isinstance(self, Accumulator) and self.n_args > 2

    def is_anypow(self) -> bool:
        return isinstance(self, Exponent)

    def is_zero_out(self) -> bool:
        return isinstance(self, (Write,))

    def is_two_out(self) -> bool:
        return isinstance(self, (SinCos,))

    def is_fma_none(self) -> bool:
        return isinstance(self, FmaNone)

    def is_fma_one(self) -> bool:
        return isinstance(self, FmaOne)

    def is_fma_many(self) -> bool:
        return isinstance(self, FmaMany)

    def is_fmaprod_two(self) -> bool:
        return isinstance(self, FmaProdTwo)

    def is_fmaprod_many(self) -> bool:
        return isinstance(self, FmaProdMany)

    def is_fma(self) -> bool:
        return isinstance(self, (FmaNone, FmaOne, FmaMany))

    def is_fmaprod(self) -> bool:
        return isinstance(self, (FmaProdTwo, FmaProdMany))

    def is_neg(self) -> bool:
        return isinstance(self, Neg)


Func_T = Type[Func]


class Accumulator(Func):
    n_outs = 1


class Write(Func):
    n_outs = 0


class Read(Func):
    def __repr__(self) -> str:
        return str(self.data)


class Store(Func):
    data: float | int

    def __repr__(self) -> str:
        return str(self.data)


class Sum(Accumulator): ...


class Minus(Func): ...


class Prod(Accumulator): ...


class Neg(Func): ...


class Div(Func): ...


class SinCos(Func):
    n_outs = 2


class Cos(Func): ...


class Sin(Func): ...


class Norm(Func): ...


class RNorm(Func): ...


class Exponent(Func):
    def exponent(self):
        if isinstance(self, Pow):
            return self.args[1]
        if isinstance(self, Square):
            return 2.0
        if isinstance(self, Rcp):
            return -1.0
        if isinstance(self, Sqrt):
            return 0.5
        if isinstance(self, RSqrt):
            return -0.5
        if isinstance(self, Cbrt):
            return 1 / 3
        if isinstance(self, RCbrt):
            return -1 / 3


class Pow(Exponent): ...


class Square(Exponent): ...


class Rcp(Exponent): ...


class Sqrt(Exponent): ...


class RSqrt(Exponent): ...


class Cbrt(Exponent): ...


class RCbrt(Exponent): ...


class Squeeze(Func): ...


class FmaProdTwo(Func): ...


class FmaProdMany(Func):
    def is_acc(self):
        return True


class FmaNone(Func):
    def is_acc(self):
        return True


class FmaOne(Func):
    def is_acc(self):
        return True


class FmaMany(Func):
    def is_acc(self):
        return True


acc_funcs = {Sum, Prod}
zero_out_funcs = {Write}
two_out_funcs = {SinCos}


TMAP: dict[Type[sf.Expr], Type[Func]] = {
    sf.Add: Sum,
    sf.Mul: Prod,
    sf.Pow: Pow,
    sf.Symbol: Read,
    symengine_wrapper.cos: Cos,
    symengine_wrapper.sin: Sin,
}
