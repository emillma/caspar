# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING, ClassVar, Hashable
from typing import Any
from typing import Type

if TYPE_CHECKING:
    from .order import FData
    from .order import VData

from symengine.lib import symengine_wrapper

import symforce.symbolic as sf


class Var:
    func: "Func"
    idx: int
    # contribs: set["Func"] = field(default_factory=set)
    vopt: "VData"

    def __init__(self, func: "Func", idx: int, *, _: None) -> None:
        self.func = func
        self.idx = idx
        assert idx < func.n_outs

    def set_func(self, func: "Func") -> None:
        object.__setattr__(self, "func", func)

    def is_const(self) -> bool:
        return self.func.is_store()

    def __hash__(self) -> int:
        return hash((self.func, self.idx))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and self.func == other.func and self.idx == other.idx

    def __repr__(self) -> str:
        return str(self.func) + (f"[{self.idx}]" if self.idx else "")


class Func:
    args: tuple[Var, ...]
    data: Hashable
    unique_id = 0  # used to create duplicate instances

    outs: list[Var]

    n_outs = 1
    fopt: "FData"
    _hash: int
    # _instances: ClassVar[dict[tuple, "Func"]] = {}

    # def __new__(cls, *args: Var, data: Any = None, unique_id: int = 0) -> "Func":
    #     if (key := (cls, args, data, unique_id)) in cls._instances:
    #         return cls._instances[key]
    #     return cls._instances.setdefault(key, super().__new__(cls))

    def __init__(self, *args: Var, data: Any = None, unique_id: int = 0) -> None:
        self.args = args
        self.data = data
        self.unique_id = unique_id
        self._hash = hash((self.__class__, self.args, self.data, self.unique_id))
        self.outs = [Var(self, i, _=None) for i in range(self.n_outs)]

        assert isinstance(data, (float, int, str)) or data is None
        assert all([isinstance(arg, Var) for arg in self.args])

    @property
    def n_args(self) -> int:
        return len(self.args)

    def rebind(self, var: Var, idx: int = 0) -> None:
        # if var.func == self:
        #     return
        var.func = self
        var.idx = idx
        self.outs[idx] = var

    def update_args(self, *args: Var) -> None:
        assert args == self.args
        assert hash(args) == hash(self.args)
        self.args = args

    def __getitem__(self, idx: int) -> Var:
        return self.outs[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(...)"
        return f"{self.__class__.__name__}({','.join(map(str, self.args))})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and self.args == other.args
            and self.data == other.data
            and self.unique_id == other.unique_id
        )

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

    def __init__(self, *args, data=None):
        super().__init__(*args, data=data)
        assert len(args) >= 2


class Write(Func):
    def print(self, _: list[Var], args: list[Var]) -> str:
        return f"{self.data} = {args[0]}"

    n_outs = 0


class Read(Func):
    def print(self, outs: list[Var], _: list[Var]) -> str:
        return f"{outs[0]} = {self.data}"

    def __repr__(self) -> str:
        return str(self.data)


class Store(Func):
    data: float

    def print(self, outs: list[Var], _: list[Var]) -> str:
        return f"{outs[0]} = {self.data}"

    def __repr__(self) -> str:
        return str(self.data)


class Sum(Accumulator):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = ({args[0]} + {args[1]})"


class Minus(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = ({args[0]} - {args[1]})"


class Prod(Accumulator):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = ({args[0]} * {args[1]})"


class Neg(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = -{args[0]}"


class Div(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = ({args[0]} / {args[1]})"


class SinCos(Func):
    n_outs = 2

    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]}, {outs[1]} = sincos({args[0]})"


class Cos(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = cos({args[0]})"


class Sin(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = sin({args[0]})"


class Norm(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = norm({args[0]})"


class RNorm(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = rnorm({args[0]})"


class Exponent(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = pow({args[0]}, {args[1]})"


class Pow(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = pow({args[0]}, {args[1]})"


class Square(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = {args[0]}*{args[0]}"


class Rcp(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = 1.0/{args[0]}"


class Sqrt(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = sqrt({args[0]})"


class RSqrt(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = rsqrt({args[0]})"


class Cbrt(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = cbrt({args[0]})"


class RCbrt(Exponent):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = rcbrt({args[0]})"


class FmaProd(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        if len(args) == 2:
            return f"{outs[0]} = {args[0]}*{args[1]}"
        else:
            return f"{outs[0]} = fma({args[0]}, {args[1]}, {args[2]})"


class FmaProdTwo(FmaProd): ...


class FmaProdMany(FmaProd):
    def is_acc(self):
        return True


class Fma(Func):
    def print(self, outs: list[Var], args: list[Var]) -> str:
        return f"{outs[0]} = {args[0]} + {args[1]}"

    def is_acc(self) -> bool:
        return True


class FmaNone(Fma): ...


class FmaOne(Fma): ...


class FmaMany(Fma): ...


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
