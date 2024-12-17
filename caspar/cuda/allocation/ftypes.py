# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from dataclasses import dataclass, field
from functools import cached_property
import random
from typing import Type, Union, get_args
import symforce.symbolic as sf
from symengine.lib import symengine_wrapper


@dataclass(eq=False)
class Var:
    func: "Func"
    idx: int = field(default=0)

    def set_func(self, func: "Func"):
        object.__setattr__(self, "func", func)

    def is_const(self):
        return self.func.is_store()

    def __hash__(self):
        return hash((self.func, self.idx))

    def __eq__(self, other):
        return isinstance(other, Var) and hash(self) == hash(other)

    def __repr__(self):
        return str(self.func) + (f"[{self.idx}]" if self.idx else "")


class Func:
    args: tuple[Var, ...]
    outs: list[Var]
    n_outs = 1
    data: float | int | str | None = None
    _hash: int | None = None

    def __init__(self, *args: Var, data=None) -> None:
        self.args = args
        self.data = data
        assert isinstance(data, (float, int, str)) or data is None
        self.outs = [Var(self, i) for i in range(self.n_outs)]
        assert isinstance(self.args, tuple)
        # assert isinstance(self.outs, tuple)
        assert all([isinstance(arg, Var) for arg in self.args])

    @property
    def n_args(self) -> int:
        return len(self.args)

    def rebind(self, var: Var, idx: int = 0):
        var.func = self
        var.idx = idx
        self.outs[idx] = var

    def __getitem__(self, idx) -> Var:
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

    def __eq__(self, other):
        return isinstance(other, Func) and hash(self) == hash(other)

    def is_write(self) -> bool:
        return isinstance(self, Write)

    def is_load(self):
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

    def is_fma(self) -> bool:
        return isinstance(self, Fma)

    def is_fmaprod(self) -> bool:
        return isinstance(self, FmaProd)


Func_T = Type[Func]


class Accumulator(Func):
    n_outs = 1


class Write(Func):
    n_outs = 0


class Read(Func):
    def __repr__(self) -> str:
        return str(self.data)


class Store(Func):
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


class FmaProd(Func):
    def is_acc(self):
        return True


class Fma(Func):
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
