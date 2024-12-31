# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations
from itertools import product
import time
from typing import Iterable

from . import ftypes
from .ftypes import Func
from .ftypes import Var


@dataclass
class FData:
    missing_args: set[Var] = field(default_factory=set)
    acc_count: int = field(default=0)

    fma_parent: Func = field(default=None)
    fma_waiting: Var = field(default=None)

    state: int = field(default=0)  # 0: not started, 1: started, 2: finished

    reg_preassure: int = field(default=-1)
    removable: int = field(default=0)
    firable: int = field(default=0)
    aff1: int = field(default=0)
    aff2: int = field(default=0)

    def __lt__(self, other: "FData") -> bool:
        return (
            self.reg_preassure < other.reg_preassure
            or self.removable < other.removable
            or self.firable < other.firable
            or self.aff1 < other.aff1
            or self.aff2 < other.aff2
        )

    def is_not_started(self) -> bool:
        return self.state == 0

    def is_started(self) -> bool:
        return self.state == 1

    def is_finished(self) -> bool:
        return self.state == 2

    def start(self) -> None:
        self.state = 1

    def finish(self) -> None:
        self.state = 2


@dataclass
class VData:
    missing_contribs: Counter[Func] = field(default_factory=Counter)
    live: bool = field(default=False)

    register: int = field(default=-1)

    def is_live(self) -> bool:
        return self.live


def prepare(
    funcs: list[Func],
) -> tuple[
    dict[Func, dict[Func, int]],
    dict[Func, dict[Func, int]],
]:
    aff1: dict[Func, dict[Func, int]] = {}
    for func in (f for f in funcs if not f.is_acc()):
        for arg0, arg1 in product(func.args, func.args):
            if arg0 is arg1:
                continue  # avoid self-affinity
            aff1.setdefault(arg0.func, {}).setdefault(arg1.func, 0)
            aff1[arg0.func][arg1.func] += 1

    aff2: dict[Func, dict[Func, int]] = {}
    for func, func2aff in aff1.items():
        for (func0, aval0), (func1, aval1) in product(func2aff.items(), func2aff.items()):
            if func0 is func1:
                continue  # avoid self-affinity
            aff2.setdefault(func0, {}).setdefault(func1, 0)
            aff2[func0][func1] += aval0 + aval1

    return aff1, aff2


class Solver:
    def __init__(
        self,
        funcs: list[Func],
        aff1: dict[Func, dict[Func, int]],
        aff2: dict[Func, dict[Func, int]],
    ):
        self.aff1 = aff1
        self.aff2 = aff2

        self.args: set[Var] = {arg for func in funcs for arg in func.outs}

        for arg in self.args:
            arg.vopt = VData()
            arg.vopt.missing_contribs == None

        for func in funcs:
            func.fopt = FData(reg_preassure=-func.n_outs)
            func.fopt.missing_args = set(func.args)
            func.fopt.acc_count = 0
            for arg in func.args:
                assert arg in self.args
                arg.vopt.missing_contribs.update([func])

        self.funcs = funcs
        self.ready = {f for f in funcs if not f.fopt.missing_args}
        self.not_ready = {k for k in funcs if k not in self.ready}

        # self.fma2fmaprods: dict[Func, set[ftypes.FmaProd]] = {}
        for fma in (f for f in funcs if f.is_fma()):
            for fmaprod in (a.func for a in fma.args if a.func.is_fmaprod()):
                fmaprod.fopt.fma_parent = fma

        self.ops: list = []
        self.max_stack = 0
        self.current_stack = 0

    def allocate(self, add: list[Var]) -> None:
        """Add a variable to stack."""

        for var in add:
            var.vopt.register = self.current_stack
            self.current_stack += 1
            self.max_stack = max(self.max_stack, self.current_stack)

    def pop_stack(self, var: Var) -> None:
        """Remove a variable from stack."""
        self.current_stack -= 1

    def use_var(self, func: Func, var: Var) -> None:
        """Use a variable in a function."""
        # print("Remove contrib: ", func, var)
        assert var not in func.fopt.missing_args
        var.vopt.missing_contribs[func] -= 1
        if func.is_acc():
            func.fopt.acc_count += 1

        if var.vopt.missing_contribs.total() == 0:
            self.pop_stack(var)

    def check_if_ready(self, func: Func) -> None:
        """Check if a function is ready"""
        if func not in self.not_ready:
            return
        elif func.is_fma():
            if func.is_fma_none() or func.is_fma_one():
                ready = any(arg.func.outs[0].vopt.is_live() for arg in func.args)
            elif func.is_fma_many():
                ready = len(func.fopt.missing_args) <= func.n_args - 2

        elif func.is_fmaprod():
            fma = next(iter(func[0].vopt.missing_contribs))
            if func.is_fmaprod_two():
                ready = not func.fopt.missing_args
                if fma.is_fma_one():
                    ready = ready and fma.args[0].vopt.is_live()
                elif fma.is_fma_many():
                    ready = ready and fma.fopt.is_started()
            else:
                ready = len(func.fopt.missing_args) <= func.n_args - 2
        elif func.is_acc():
            ready = len(func.fopt.missing_args) <= func.n_args - 2

        else:
            ready = not func.fopt.missing_args

        if ready:
            self.not_ready.remove(func)
            self.ready.add(func)

    def do_func(self, func: Func) -> None:
        """Do a function."""
        for v in func.args:
            self.use_var(func, v)
        self.allocate(func.outs)
        self.ops.append((func, *func.args))
        self.finish_func(func)

    def finish_func(self, func: Func) -> None:
        """Finish a function."""
        # print("Finish: ", func)
        func.fopt.finish()
        for out in func.outs:
            out.vopt.live = True
            for contrib in out.vopt.missing_contribs.copy():
                contrib.fopt.missing_args.remove(out)
                if contrib.is_acc() and contrib.fopt.is_started():
                    self.accumulate(contrib, out, contrib.outs[0])
                self.check_if_ready(contrib)

        # Update scores
        for other in (f for f in self.aff1.get(func, {}) if f.fopt.is_not_started()):
            other.fopt.aff1 += self.aff1[func][other]
            for arg in other.args:
                if all(f is other for f in arg.vopt.missing_contribs):
                    other.fopt.reg_preassure += 1

            for out in other.outs:
                if all(f.fopt.is_started() for f in out.vopt.missing_contribs):
                    other.fopt.removable += 1
                for contrib in out.vopt.missing_contribs:
                    if func.fopt.missing_args <= set(other.outs):
                        other.fopt.firable += 1

        for other in (f for f in self.aff2.get(func, {}) if f.fopt.is_not_started()):
            other.fopt.aff2 += self.aff2[func][other]

    def start_fma(self, func: Func) -> None:
        """Start an FMA function."""
        # print("Start fma: ", func)
        if func.is_fma_none():
            self.start_accumulate(func)
        if func.is_fma_one():
            self.start_accumulate(func)
        if func.is_fma_many():
            self.start_accumulate(func)

        for prod in (arg.func for arg in func.args if arg.func.is_fmaprod()):
            if (var := prod.fopt.fma_waiting) is not None:
                self.accumulate(prod, var, prod.outs[0])
                prod.fopt.fma_waiting = None
            else:
                self.check_if_ready(prod)

    def start_fmaprod(self, func: Func) -> None:
        """Start accumulating a function."""
        # print("Start fmaprod: ", func)
        fma = func.fopt.fma_parent
        if func.is_fmaprod_two():
            if fma.is_fma_none() and not fma.fopt.is_started():
                self.do_func(func)

            else:
                self.use_var(func, func.args[0])
                self.use_var(func, func.args[1])
                self.ops.append((func, func.args[0], func.args[1], fma.outs[0]))
                fma.fopt.missing_args.remove(func.outs[0])
                func.outs[0].vopt.missing_contribs[fma] -= 1
                fma.fopt.acc_count += 1
                if fma.fopt.acc_count == len(fma.args):
                    self.finish_func(fma)

        else:
            self.start_accumulate(func)

    def start_accumulate(self, func: Func) -> None:
        """Start accumulating a function."""
        # print("Start accumulate: ", func)
        func.fopt.start()
        live_args = [v for v in func.args if v.vopt.is_live()]
        first = max(live_args, key=lambda v: v.vopt.missing_contribs.keys() <= {func})
        self.use_var(func, first)
        for i, v in enumerate(a for a in live_args if a is not first):
            self.accumulate(func, v, first if i == 0 else func.outs[0])
        self.allocate(func.outs)

    def accumulate(self, func: Func, var: Var, prev: Var) -> None:
        # print("Accumulate: ", func, var)

        if (
            func.is_fmaprod()
            and func.fopt.acc_count == len(func.args) - 1
            and ((fma := func.fopt.fma_parent).fopt.is_started() or not fma.is_fma_none())
        ):
            if not fma.fopt.is_started():
                func.fopt.fma_waiting = var
                self.check_if_ready(fma)
            else:
                self.use_var(func, var)
                fma.fopt.missing_args.remove(func.outs[0])
                self.use_var(fma, func.outs[0])
                self.ops.append((func, var, prev, fma.outs[0]))
                func.fopt.finish()
                if fma.fopt.acc_count == len(fma.args):
                    self.finish_func(fma)
        else:
            self.use_var(func, var)
            if not var.func.is_fmaprod():
                self.ops.append((func, var, prev))  # this is already done for fma_prods

            if func.fopt.acc_count == len(func.args):
                self.finish_func(func)

    def reorder(self) -> None:
        t0 = time.perf_counter()
        while self.not_ready or self.ready:
            func = max(self.ready, key=lambda f: f.fopt)
            # print(func.__class__.__name__)
            self.ready.remove(func)
            if func.is_fma():
                self.start_fma(func)
            elif func.is_fmaprod():
                self.start_fmaprod(func)
            elif func.is_acc():  # accumulate
                self.start_accumulate(func)
            else:
                self.do_func(func)
        print("Time: ", time.perf_counter() - t0)
        print(self.max_stack)

    def format_reordering(self) -> None:
        ssa_regmap = {}
        regmap = ssa_regmap
        count = 0
        print("")
        fma_prod_couts: Counter[Func] = Counter()
        for op in self.ops:
            func, *args = op
            outs = func.outs
            if func.is_fmaprod():
                fma_prod_couts[func] += 1
                if fma_prod_couts[func] == len(func.args) - 1:
                    outs = func.fopt.fma_parent.outs
            arg_str = [f"r{regmap[a]}" for a in args]
            for out in outs:
                ssa_regmap[out] = count
                count += 1
            out_str = [f"r{regmap[a]}" for a in outs]
            print(f"{func.print(out_str, arg_str):<50}")
