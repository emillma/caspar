# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from fractions import Fraction
from functools import cache
from itertools import combinations, product
import time
from typing import Type
import symforce.symbolic as sf
from . import ftypes
from .ftypes import Func, Var, Var


@cache
def get_pow_map() -> dict[Fraction, tuple | Fraction]:
    options = {
        Fraction(-1, 1): 2,
        Fraction(2, 1): 1,
        Fraction(1, 2): 2,
        Fraction(1, 3): 2,
        Fraction(-1, 3): 2,
        Fraction(-1, 2): 2,
    }
    costs = options.copy() | {Fraction(1, 1): 0}
    exp_maps = {k: k for k in costs}
    anynew = True

    def add_if_new(e0, cost0, e1, cost1, multiply):
        exp_new = e0 + e1 if multiply else e0 * e1
        cost = cost0 + cost1 + 1 if multiply else cost0 + cost1
        if cost < costs.get(exp_new, 6):
            costs[exp_new] = cost
            exp_maps[exp_new] = (multiply, e0, e1)
            return True
        return False

    while anynew:
        anynew = False
        for (e0, cost0), (e1, cost1) in list(product(costs.items(), costs.items())):
            anynew |= add_if_new(e0, cost0, e1, cost1, True)
        for (e0, cost0), (e1, cost1) in list(product(options.items(), costs.items())):
            anynew |= add_if_new(e0, cost0, e1, cost1, False)
    return exp_maps


EXP_MAP = get_pow_map()
OP_MAP = {
    Fraction(-1, 1): ftypes.Rcp,
    Fraction(2, 1): ftypes.Square,
    Fraction(1, 2): ftypes.Sqrt,
    Fraction(1, 3): ftypes.Cbrt,
    Fraction(-1, 3): ftypes.RCbrt,
    Fraction(-1, 2): ftypes.RSqrt,
}


def fix_pow(func: Func) -> Func:
    def inner(base: Var, exponent: Fraction) -> Var:
        if (exp := EXP_MAP.get(exponent, None)) is None:
            return ftypes.Pow(base, ftypes.Store(data=float(exponent))[0])[0]

        if isinstance(exp, Fraction):
            if exp == 1:
                return base
            else:
                return OP_MAP[exp](base)[0]

        mul, e0, e1 = exp
        if mul:
            v0 = inner(base, e0)
            v1 = inner(base, e1)
            return ftypes.Prod(v0, v1)[0]
        else:
            v1 = inner(base, e1)
            return inner(v1, e0)

    if not func.args[1].is_const():
        return func
    exponent = Fraction(float(func.args[1].func.data)).limit_denominator()
    return inner(func.args[0], exponent).func


def find_shared_args(funcs: list[Func]) -> dict[Func, set[Var]]:
    if not funcs:
        return {}
    ftyp = funcs[0].__class__

    argmap: dict[Func, set[Var]] = {func: set(func.args) for func in funcs}
    intersects: dict[tuple[Var, ...], list[Func]] = {}

    for (f0, set0), (f1, set1) in combinations(argmap.items(), 2):
        inter_set = set0 & set1
        if len(inter_set) >= 2:
            intersects.setdefault(tuple(inter_set), []).extend([f0, f1])

    while intersects:
        inter_tup = max(intersects, key=len)
        new_thing = ftyp(*inter_tup)
        targets = intersects.pop(inter_tup)
        for target in targets:
            argset = argmap[target]
            argset -= set(inter_set)
            argset.add(new_thing[0])

    return argmap


def fix_accum(FType: Type[Func], nested_args: tuple[tuple | Var, ...]) -> Func:
    args = []
    for thing in nested_args:
        if isinstance(thing, Var):
            args.append(thing)
        elif isinstance(thing, tuple):
            args.append(fix_accum(FType, thing)[0])
    return FType(*args) if len(args) > 1 else args[0].func
