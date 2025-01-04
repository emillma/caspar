# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.

from collections import Counter
from typing import Generator
from typing import Type

from symengine.lib import symengine_wrapper

import symforce.symbolic as sf

from . import fixers
from . import ftypes
from .ftypes import TMAP
from .ftypes import Func
from .ftypes import Var


class Problem:
    def __init__(self, exprs: list[sf.Expr]):
        expr_map: dict[sf.Expr, Var] = {}

        def translate(expr: sf.Expr) -> Var:
            if (out := expr_map.get(expr)) is not None:
                return out

            if expr.is_Number or isinstance(expr, (int, float)):
                return expr_map.setdefault(expr, ftypes.Store(data=float(expr))[0])

            FType = TMAP[type(expr)]
            if expr.is_Symbol:
                return expr_map.setdefault(expr, FType(data=expr.name)[0])
            else:
                args = [translate(arg) for arg in expr.args]
                func = FType(*args)
                return expr_map.setdefault(expr, func[0])

        mapped = [translate(expr) for expr in exprs]
        root_vars = [var for var in mapped if isinstance(var, Var)]
        self.root_funcs = [ftypes.Write(rv, data=i) for i, rv in enumerate(root_vars)]
        ls = list(self.root_funcs)
        assert next(iter(self.root_funcs)) in self.root_funcs

        self.fix_pow()
        self.expand_prods()
        self.collect_pows()
        self.fix_sums()
        # self.fix_minus()
        self.fix_prods()
        self.fix_div()
        self.fix_sincos()
        self.fix_norms()
        # self.fix_fma()
        self.split_store()
        self.split_acc()
        self.make_unique()

        assert len(set(self.funcs())) == len(list(self.funcs()))

    def make_unique(self) -> None:
        to_visit: list[Func] = list(self.root_funcs)
        unique_arg: dict[Var, Var] = {}
        while to_visit:
            func = to_visit.pop(-1)
            update = False
            for arg in func.args:
                if arg not in unique_arg:
                    unique_arg[arg] = arg
                    to_visit.append(arg.func)
                    continue
                if unique_arg[arg] is not arg:
                    update = True
            if update:
                func.update_args(*(unique_arg[a] for a in func.args))

    def check_unique(self) -> None:
        for func in self.funcs():
            for out in func.outs:
                assert out.func is func

            for arg in func.args:
                assert arg.func[arg.idx] is arg

        to_visit: list[Func] = list(self.root_funcs)
        unique_funcs: dict[Func, Func] = {}
        while to_visit:
            func = to_visit.pop(-1)
            assert all(func[i].func is func for i in range(func.n_outs))
            if func in unique_funcs:
                assert unique_funcs[func] is func
                continue
            unique_funcs[func] = func
            to_visit.extend(v.func for v in func.args)

    def funcs(self, ftype: Type[Func] | None = None) -> Generator[Func, None, None]:
        """Depth-first traversal of the function graph."""
        visited: set[int] = set()
        to_visit = list(self.root_funcs)
        while to_visit:
            func = to_visit.pop(-1)
            if id(func) in visited:
                continue
            visited.add(id(func))
            if ftype is None or isinstance(func, ftype):
                yield func
            to_visit.extend(v.func for v in func.args)

    def vars(self) -> Generator[Var, None, None]:
        visited: set[Var] = set()
        for func in self.funcs():
            for arg in func.outs:
                if arg in visited:
                    continue
                visited.add(arg)
                yield arg

    def contribs(self) -> dict[Var, list[Func]]:
        contribs: dict[Var, list[Func]] = {}
        for func in self.funcs():
            for arg in func.args:
                contribs.setdefault(arg, []).append(func)
        return contribs

    def fix_pow(self) -> None:  # a**-(2/3) -> rcbrt(a)**2
        for pow in self.funcs(ftypes.Pow):
            new_pow = fixers.fix_pow(pow)
            new_pow.rebind(pow.outs[0])

    def expand_prods(self) -> None:  # a*(b*c) -> a*b*c
        def prod_gen(arg: Var) -> Generator[Var, None, None]:
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
        ptypes = [ftypes.Square, ftypes.Rcp, ftypes.Sqrt, ftypes.RSqrt, ftypes.Cbrt, ftypes.RCbrt]
        to_check = list(self.funcs(ftypes.Prod))
        for prod in to_check:
            args = []
            for ptype in ptypes:
                instances = [p for p in prod.args if isinstance(p.func, ptype)]
                if len(instances) == 0:
                    continue
                elif len(instances) == 1:
                    args.append(instances[0])
                else:
                    base = ftypes.Prod(*[p.func.args[0] for p in instances])[0]
                    to_check.append(base.func)
                    args.append(ptype(base)[0])

            common: dict[Var, list[Var]] = {}
            for arg in (a for a in prod.args if a.func.is_pow()):
                common.setdefault(arg.func.args[1], []).append(arg.func.args[0])
            for exp, bases in common.items():
                if len(bases) == 1:
                    args.append(ftypes.Pow(bases[0], exp)[0])
                else:
                    base = ftypes.Prod(*bases)[0]
                    to_check.append(base.func)
                    args.append(ftypes.Pow(base, exp)[0])

            args += [a for a in prod.args if not isinstance(a.func, ftypes.Exponent)]
            if len(args) == 1:
                new_prod = args[0].func
            else:
                new_prod = ftypes.Prod(*args)
            new_prod.rebind(prod.outs[0])

    def fix_prods(self) -> None:
        mul_map = fixers.find_shared_args(list(self.funcs(ftypes.Prod)))
        for func, new_args in mul_map.items():
            new_prod = ftypes.Prod(*new_args)
            new_prod.rebind(func.outs[0])

        # for prod in prods:
        #     new_prod.rebind(prod.outs[0])

    def fix_div(self) -> None:
        for var, contribs in self.contribs().items():
            if not (len(contribs) == 1 and contribs[0].is_prod() and var.func.is_rcp()):
                continue
            others = [a for a in contribs[0].args if a != var]
            new_prod_var = ftypes.Prod(*others)[0] if len(others) > 1 else others[0]
            new_div = ftypes.Div(new_prod_var, var.func.args[0])
            new_div.rebind(contribs[0].outs[0])

    def fix_sums(self) -> None:
        sum_map = fixers.find_shared_args(list(self.funcs(ftypes.Sum)))
        for func, new_args in sum_map.items():
            new_prod = ftypes.Sum(*new_args)
            new_prod.rebind(func.outs[0])

    def fix_minus(self) -> None:
        for func in self.funcs(ftypes.Prod):
            funcs = (a.func for a in func.args)
            neg = next((f for f in funcs if f.is_store() and f.data == -1), None)
            if neg is not None:
                args = [a for a in func.args if a.func is not neg]
                if len(args) == 1:
                    new_func = args[0].func
                else:
                    new_func = ftypes.Prod(*(a for a in func.args if a.func is not neg))
                new_neg = ftypes.Neg(new_func[0])
                new_neg.rebind(func.outs[0])

        for func in self.funcs(ftypes.Sum):
            negs = [a for a in func.args if a.func.is_neg()]
            if not negs:
                continue
            other = [a for a in func.args if not a.func.is_neg()]
            if len(negs) == 1:
                neg_part = negs[0].func.args[0]
            else:
                neg_part = ftypes.Sum(*(a.func.args[0] for a in negs))[0]

            if len(other) == 0:
                new_minus: Func = ftypes.Neg(neg_part)
            elif len(other) == 1:
                new_minus = ftypes.Minus(other[0], neg_part)
            else:
                other_part = ftypes.Sum(*other)[0]
                new_minus = ftypes.Minus(other_part, neg_part)
            new_minus.rebind(func.outs[0])

    def fix_sincos(self) -> None:
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

    def fix_norms(self) -> None:
        for root_typ in [ftypes.Sqrt, ftypes.RSqrt]:
            for root in self.funcs(root_typ):
                if not (inner := root.args[0].func).is_sum():
                    continue
                if all(
                    a.func.is_square() or (a.func.is_store() and a.func.data >= 0)
                    for a in inner.args
                ):
                    store_vals = [a.func.data for a in inner.args if a.func.is_store()]
                    new_lits = [ftypes.Store(data=v**0.5)[0] for v in store_vals]
                    other = [a.func.args[0] for a in inner.args if not a.func.is_store()]
                    norm_tyb = ftypes.Norm if root_typ is ftypes.Sqrt else ftypes.RNorm
                    new_func = norm_tyb(*new_lits, *other)
                    new_func.rebind(root.outs[0])

    def fix_fma(self) -> None:
        contribs = self.contribs()
        for sum in self.funcs(ftypes.Sum):
            unique_prods: list[ftypes.Prod] = []
            other = []
            for arg in sum.args:
                if (arg.func.is_prod() or arg.func.is_square()) and len(contribs[arg]) == 1:
                    if arg.func.is_square():
                        new_prod = ftypes.Prod(arg.func.args[0], arg.func.args[0])
                        new_prod.rebind(arg)
                        arg = new_prod[0]
                    unique_prods.append(arg.func)
                else:
                    other.append(arg)
            if not unique_prods:
                continue

            fma_prods = []
            for p in unique_prods:
                cls = ftypes.FmaProdTwo if len(p.args) == 2 else ftypes.FmaProd
                fma_prods.append(cls(*p.args)[0])

            fmacls = [ftypes.FmaNone, ftypes.Fma][len(other) > 0]
            new_sum: Func = fmacls(*other, *fma_prods)
            new_sum.rebind(sum.outs[0])

    def split_store(self) -> None:
        nstores: Counter[Func] = Counter()
        for func in self.funcs():
            if not any(a.func.is_store() for a in func.args):
                continue
            args = []
            for arg in func.args:
                if arg.func.is_store():
                    nstores[arg.func] += 1
                    new_store = ftypes.Store(data=arg.func.data, unique_id=nstores[arg.func])
                    args.append(new_store[0])
                else:
                    args.append(arg)
            new_func = func.__class__(*args)
            new_func.rebind(func.outs[0])

    def split_acc(self) -> None:
        i = 1
        for ftype in [ftypes.Sum, ftypes.Prod]:
            for accumulator in self.funcs(ftype):
                args = [ftypes.Contribute(arg, unique_id=i)[0] for arg in accumulator.args]
                ftype(*args).rebind(accumulator.outs[0])
                i += 1
