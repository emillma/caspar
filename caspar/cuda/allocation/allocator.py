# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement, product
from typing import Generator, Type
import symforce.symbolic as sf
from collections import Counter
from symengine.lib import symengine_wrapper
import numpy as np
from fractions import Fraction

from . import ftypes
from .ftypes import Func, Var, Var, TMAP, Func_T
from . import fixers


class Problem:
    def __init__(self, exprs: list[sf.Expr]):
        expr_map: dict[sf.Expr, Var] = {}

        def translate(expr: sf.Expr) -> Var:
            if expr.is_Number or isinstance(expr, (int, float)):
                return ftypes.Store(data=float(expr))[0]
            if (out := expr_map.get(expr)) is not None:
                return out

            FType = TMAP[type(expr)]
            if expr.is_Symbol:
                out = expr_map.setdefault(expr, FType(data=expr.name)[0])
            else:
                args = (translate(arg) for arg in expr.args)
                out = expr_map.setdefault(expr, FType(*args)[0])
            return out

        mapped = [translate(expr) for expr in exprs]
        root_vars = [var for var in mapped if isinstance(var, Var)]
        self.root_funcs = [ftypes.Write(rv, data=i) for i, rv in enumerate(root_vars)]
        ls = list(self.root_funcs)
        assert next(iter(self.root_funcs)) in self.root_funcs

        for expr in exprs:
            translate(expr)

        self.fix_pow()
        self.expand_prods()
        self.collect_pows()
        self.fix_sums()
        # self.fix_minus()
        self.fix_prods()
        self.fix_div()
        self.fix_fma()
        self.fix_sincos()
        # self.squeeze()
        None

    def funcs(self, ftype: Type[Func] | None = None) -> Generator[Func, None, None]:
        """Depth-first traversal of the function graph."""
        visited: set[Func] = set()
        to_visit = list(self.root_funcs)
        while to_visit:
            func = to_visit.pop(-1)
            if func in visited:
                continue
            visited.add(func)
            if ftype is None or isinstance(func, ftype):
                yield func
            to_visit.extend(v.func for v in func.args)

    def vars(self) -> Generator[Var, None, None]:
        visited: set[Var] = set()
        for func in self.funcs():
            for arg in func.args:
                if arg in visited:
                    continue
                visited.add(arg)
                yield arg

    def dependencies(self) -> dict[Var, list[Func]]:
        deps: dict[Var, list[Func]] = {}
        for func in self.funcs():
            for arg in func.args:
                deps.setdefault(arg, []).append(func)
        return deps

    def fix_pow(self) -> None:  # a**-(2/3) -> rcbrt(a)**2
        for pow in self.funcs(ftypes.Pow):
            new_pow = fixers.fix_pow(pow)
            new_pow.rebind(pow.outs[0])

    def expand_prods(self):  # a*(b*c) -> a*b*c
        def prod_gen(arg: Var):
            if not arg.func.is_prod():
                yield arg
            else:
                for arg in arg.func.args:
                    yield from prod_gen(arg)

        for prod in self.funcs(ftypes.Prod):
            args = [b for a in prod.args for b in prod_gen(a)]
            new_prod = ftypes.Prod(*args)
            new_prod.rebind(prod.outs[0])

    def collect_pows(self) -> None:
        for mul in self.funcs(ftypes.Prod):
            pows: list[ftypes.Exponent]
            pows = [v.func for v in mul.args if v.func.is_anypow()]

            exps: dict[tuple[Func_T, float], list[Var]] = {}
            for p in pows:
                exps.setdefault((type(p), p.exponent()), []).append(p.args[0])
            new_pows = []
            for (FType, exp), args in exps.items():
                base = ftypes.Prod(*args)[0] if len(args) > 1 else args[0]
                if FType is ftypes.Pow:
                    new_pows.append(FType(base, ftypes.Store(data=exp)[0])[0])
                else:
                    new_pows.append(FType(base)[0])
            other = (a for a in mul.args if not a.func.is_anypow())
            if len(new_args := (*new_pows, *other)) == 1:
                new_func = new_args[0].func
            else:
                new_func = ftypes.Prod(*new_args)
            new_func.rebind(mul.outs[0])

    def fix_prods(self):
        prods = [p for p in self.funcs(ftypes.Prod)]
        mul_map = fixers.find_shared_args([p.args for p in prods])
        for prod in prods:
            new_prod = fixers.fix_accum(ftypes.Prod, mul_map[prod.args])
            new_prod.rebind(prod.outs[0])

    def fix_div(self) -> None:
        for var, funcs in self.dependencies().items():
            if not (
                var.func
                and len(funcs) == 1
                and funcs[0].is_prod()
                and var.func.is_rcp()
            ):
                continue
            others = [a for a in funcs[0].args if a != var]
            new_prod_var = ftypes.Prod(*others)[0] if len(others) > 1 else others[0]
            new_div = ftypes.Div(new_prod_var, var.func.args[0])
            new_div.rebind(funcs[0].outs[0])

    def fix_sums(self):
        sums = [s for s in self.funcs(ftypes.Sum)]
        sum_map = fixers.find_shared_args([s.args for s in sums])
        for sum in sums:
            new_sum = fixers.fix_accum(ftypes.Sum, sum_map[sum.args])
            new_sum.rebind(sum.outs[0])

    def fix_minus(self):
        for func in self.funcs(ftypes.Prod):
            if -1.0 in func.lit_args:
                new_prod = ftypes.Prod(*(a for a in func.args if a != -1.0))
                new_neg = ftypes.Neg(new_prod[0])
                new_neg.rebind(func.outs[0])
        for func in self.funcs(ftypes.SUM):
            None

    def fix_sincos(self):
        sin = {}
        cos = {}
        shared = {}
        for func in self.funcs():
            if func.is_sin():
                if func.args[0] in cos:
                    shared[func.args[0]] = (func, cos.pop(func.args[0]))
                else:
                    sin[func.args[0]] = func
            elif func.is_cos():
                if func.args[0] in sin:
                    shared[func.args[0]] = (sin.pop(func.args[0]), func)
                else:
                    cos[func.args[0]] = func

        for base, (s, c) in shared.items():
            new_sincos = ftypes.SinCos(base)
            new_sincos.rebind(s[0])
            new_sincos.rebind(c[0], 1)

    def fix_fma(self) -> None:
        deps = self.dependencies()
        for sum in self.funcs(ftypes.Sum):
            unique_prods: list[ftypes.Prod] = []
            other = []
            for arg in sum.args:
                if arg.func.is_prod() or arg.func.is_square() and len(deps[arg]) == 1:
                    if arg.func.is_square():
                        new_prod = ftypes.Prod(arg.func.args[0], arg.func.args[0])
                        new_prod.rebind(arg)
                        arg = new_prod[0]
                    unique_prods.append(arg.func)
                else:
                    other.append(arg)
            if not unique_prods:
                continue

            fma_prods = (ftypes.FmaProd(*p.args)[0] for p in unique_prods)
            new_sum = ftypes.Fma(*other, *fma_prods)
            new_sum.rebind(sum.outs[0])

    def fix_norms(self):
        pass

    # def _fix_expr(self, expr: sf.Expr) -> Func:
    #     if expr in self._fixed_exprs:
    #         return self._fixed_exprs[expr]

    #     methods = {
    #         lambda x: x.is_Symbol or x.is_Number: self.fix_lit,
    #         lambda x: x.is_Add: self.fix_sum,
    #         lambda x: x.is_Mul: self.fix_mul,
    #         lambda x: x.is_Pow: self.fix_pow,
    #         lambda x: isinstance(x, symengine_wrapper.cos): self.fix_cos,
    #         lambda x: isinstance(x, symengine_wrapper.sin): self.fix_sin,
    #     }
    #     for method, fix in methods.items():
    #         if method(expr):
    #             return fix(expr)
    #     raise NotImplementedError

    # def fix_expr(self, expr: sf.Expr) -> Var:
    #     if expr in self._fixed_exprs:
    #         return self._fixed_exprs[expr]
    #     reg = self._fix_expr(expr)
    #     assert isinstance(reg, Var)
    #     return self._fixed_exprs.setdefault(expr, reg)

    # def fix_lit(self, expr: sf.Symbol | sf.Number) -> Var:
    #     if isinstance(expr, sf.Symbol):
    #         return Func(FTYPES.LOAD, lit_args=[expr.name])[0]
    #     elif isinstance(expr, sf.Number):
    #         return Func(FTYPES.STORE, lit_args=[expr.evalf()])[0]

    # def fix_accum(self, ftype: FTYPES, args: tuple[sf.Expr | tuple, ...]) -> Var:
    #     out = []
    #     for arg in args:
    #         if isinstance(arg, sf.Expr):
    #             out.append(self.fix_expr(arg))
    #         else:
    #             out.append(self.fix_accum(ftype, arg))
    #     return Func(ftype, out)[0]

    # def fix_sum(self, expr: sf.Add) -> Var:
    #     return self.fix_accum(FTYPES.SUM, self._sum_map[expr.args])

    # def fix_mul(self, expr: sf.Mul | sf.Pow) -> Var:
    #     if expr.is_Pow:
    #         return self.fix_pow(expr)
    #     return self.fix_accum(FTYPES.PROD, self._prod_map[expr.args])

    # def fix_pow(self, expr: sf.Pow) -> Var:
    #     base, exp = expr.args
    #     if is_fastpow(expr):
    #         reg = self.fix_expr(base)
    #         return {
    #             2: Func(FTYPES.PROD, [reg, reg])[0],
    #             3: Func(FTYPES.PROD, [reg, reg, reg])[0],
    #             4: Func(FTYPES.PROD, [Var(Func(FTYPES.PROD, [reg, reg]))] * 2)[0],
    #         }[exp]
    #     if not exp.is_Number:
    #         return Func(FTYPES.POW, [self.fix_expr(base), self.fix_expr(exp)])[0]
    #     expf = float(exp)

    #     if is_squared_sum(base) and expf in (0.5, -0.5):
    #         powregs = [self.fix_expr(x.args[0]) for x in base.args if is_pow2(x)]
    #         litregs = [self.fix_expr(x) for x in base.args if x.is_Number]
    #         assert len(powregs) + len(litregs) == len(base.args)
    #         if expf == 0.5:
    #             return Func(FTYPES.NORM, [*powregs, *litregs])[0]
    #         else:
    #             return Func(FTYPES.RNORM, [*powregs, *litregs])[0]

    #     if expf == -1:
    #         return Func(FTYPES.RCP, [self.fix_expr(base)])[0]
    #     if expf == 0.5:
    #         return Func(FTYPES.SQRT, [self.fix_expr(base)])[0]
    #     if expf == -0.5:
    #         return Func(FTYPES.RSQRT, [self.fix_expr(base)])[0]
    #     raise NotImplementedError

    # def fix_cos(self, expr: sf.Expr) -> Var:
    #     if sf.sin(expr.args[0]) in self.unique_exprs:
    #         return Func(FTYPES.SINCOS, [self.fix_expr(expr.args[0])])[1]
    #     return Func(FTYPES.COS, [self.fix_expr(expr.args[0])])[0]

    # def fix_sin(self, expr: sf.Expr) -> Var:
    #     if sf.cos(expr.args[0]) in self.unique_exprs:
    #         return Func(FTYPES.SINCOS, [self.fix_expr(expr.args[0])])[0]
    #     return Func(FTYPES.SIN, [self.fix_expr(expr.args[0])])[0]
