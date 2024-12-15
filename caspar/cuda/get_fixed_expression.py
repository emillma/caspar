# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from symforce import symbolic as sf
from functools import lru_cache
from symforce.ops import LieGroupOps as Ops


def all_pow2(args):
    return all(isinstance(arg, sf.Pow) and arg.args[1] == 2 for arg in args)


def contains_mul(expr):
    return any(arg.is_Mul or (arg.is_Function and 1) for arg in expr.args)


fma = sf.Function("fma")
add2 = sf.Function("add2")
mul2 = sf.Function("mul2")
rcp = sf.Function("rcp")
div = sf.Function("div")

sqrt = sf.Function("sqrt")
rsqrt = sf.Function("rsqrt")
norm2 = sf.Function("norm2")
norm3 = sf.Function("norm3")
norm4 = sf.Function("norm4")

rnorm2 = sf.Function("rnorm2")
rnorm3 = sf.Function("rnorm3")
rnorm4 = sf.Function("rnorm4")
ternary = sf.Function("ternary")


@lru_cache(maxsize=None)
def fix_subexpr(expr: sf.Expr):

    args = getattr(expr, "args", [])
    if expr.is_Symbol:
        return expr

    elif expr.is_Number:
        return expr.evalf()

    elif expr.is_Add:
        args = sorted(args, key=lambda x: (x.is_Number, not x.is_Mul, not x.is_Pow))
        if args[0].is_Mul:
            arg0 = fix_subexpr(args[0].args[0])
            arg1 = fix_subexpr(sf.Mul(*args[0].args[1:]))
            arg2 = fix_subexpr(sf.Add(*args[1:]))
            return fma(arg0, arg1, arg2)
        elif args[0].is_Pow and args[0].args[1] == 2:
            arg0 = fix_subexpr(args[0].args[0])
            arg2 = fix_subexpr(sf.Add(*args[1:]))
            return fma(arg0, arg0, arg2)
        arg0 = fix_subexpr(args[0])
        arg1 = fix_subexpr(sf.Add(*args[1:]))
        return add2(arg0, arg1)

    elif expr.is_Mul:
        # TODO: handle sign as copysign
        # TODO: handle pow-1 as div
        args = sorted(args, key=lambda x: x.is_Pow and x.args[1] == -1)

        # SHOULD NOT DIVIDE IF DENOMINATOR IS USED MULTIPLE TIMES
        # if args[-1].is_Pow and args[-1].args[1] == -1:
        #     arg0 = fix_subexpr(sf.Mul(*args[:-1]))
        #     arg1 = fix_subexpr(args[-1].args[0])
        #     return div(arg0, arg1)
        # else:
        arg0 = fix_subexpr(args[0])
        arg1 = fix_subexpr(sf.Mul(*args[1:]))
        return mul2(arg0, arg1)

    elif expr.is_Pow and args[1].is_Number:
        pow = float(args[1])
        if args[1] == -1:
            return rcp(fix_subexpr(args[0]))
        elif pow == 2:
            fixed = fix_subexpr(args[0])
            return mul2(fixed, fixed)
        elif pow == -2:
            fixed = fix_subexpr(args[0])
            return rcp(mul2(fixed, fixed))
        elif pow == 3:
            fixed = fix_subexpr(args[0])
            return mul2(fixed, mul2(fixed, fixed))
        elif pow == -3:
            fixed = fix_subexpr(args[0])
            return rcp(mul2(fixed, mul2(fixed, fixed)))
        elif pow == 4:
            fixed = fix_subexpr(args[0])
            return mul2(mul2(fixed, fixed), mul2(fixed, fixed))
        elif pow == -4:
            fixed = fix_subexpr(args[0])
            return rcp(mul2(mul2(fixed, fixed), mul2(fixed, fixed)))
        elif pow == 0.5:
            if all_pow2(args[0].args) and 2 <= (n := len(args[0].args)) <= 4:
                args = [fix_subexpr(arg.args[0]) for arg in args[0].args]
                return [None, None, norm2, norm3, norm4][n](*args)
            else:
                return sqrt(fix_subexpr(args[0]))
        elif pow == -0.5:
            if all_pow2(args[0].args) and 2 <= (n := len(args[0].args)) <= 4:
                args = [fix_subexpr(arg.args[0]) for arg in args[0].args]
                return [None, None, rnorm2, rnorm3, rnorm4][n](*args)
            else:
                return rsqrt(fix_subexpr(args[0]))
    elif isinstance(expr, sf.Piecewise):
        assert len(args) == 4
        return ternary(fix_subexpr(args[1]), fix_subexpr(args[0]), fix_subexpr(args[2]))

    caller = getattr(expr, "func", type(expr))
    return caller(*[fix_subexpr(arg) for arg in args])


def get_fixed_expressions(exprs: list[sf.Expr]):
    return [fix_subexpr(sf.sympify(expr)) for expr in Ops.to_storage(exprs)]
