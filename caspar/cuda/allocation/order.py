# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.

from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations
from itertools import product
from pprint import pprint
import time
from typing import Iterable

from . import ftypes
from .ftypes import Func
from .ftypes import Var


@dataclass
class FData:
    func: Func
    missing_args: set[Var] = field(default_factory=set)
    acc_count: int = field(default=0)

    fma_parent: Func = field(default=None)
    fma_need_one: bool = field(init=False)
    fma_prev: Var = field(default=None)
    fma_last: Var = field(default=None)

    state: int = field(default=-1)  # -1: not_ready, 0: not started, 1: started, 2: finished

    reg_preassure: int = field(init=False)
    removable: list[bool] = field(default=0)
    firable: int = field(default=0)
    priority: int = field(default=-(2**32))
    aff2: int = field(default=0)
    aff1: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.reg_preassure = -self.func.n_outs
        self.fma_need_one = self.func.is_fma_none()

    def __lt__(self, other: "FData") -> bool:
        return (
            self.reg_preassure < other.reg_preassure or self.aff1 < other.aff1
            # or self.priority < other.priority
            # or self.aff1 < other.aff1
            # or self.aff2 < other.aff2
            # or self.firable < other.firable
        )

    def key(self) -> tuple:
        return (self.reg_preassure, self.aff1, self.priority)

    def is_not_ready(self) -> bool:
        return self.state < 0

    def is_not_started(self) -> bool:
        return self.state < 1

    def is_started(self) -> bool:
        return self.state == 1

    def is_finished(self) -> bool:
        return self.state == 2

    def ready(self) -> None:
        self.state = 0

    def start(self) -> None:
        self.state = 1

    def finish(self) -> None:
        # assert self.m
        self.state = 2

    def update_aff1(self, val: int, priority: int, done: set[Func] = None) -> None:
        self.aff1 += val
        self.priority = max(self.priority, priority)
        if done is None:
            done = set()
        elif self.func in done:
            return
        for arg in self.missing_args:
            arg.func.fopt.update_aff1(val=val, priority=priority, done=done)
        done.add(self.func)

    def update_aff2(self, val: int = 1, done: set[Func] = None) -> None:
        self.aff2 += val
        # if done is None:
        #     done = set()
        # elif self.func in done:
        #     return
        # for arg in self.missing_args:
        #     arg.func.fopt.update_aff2(done=done)
        # done.add(self.func)


@dataclass
class VData:
    var: Var
    missing_contribs: Counter[Func] = field(default_factory=Counter)
    live: bool = field(default=False)
    virtual: bool = field(init=False)
    register: int = field(default=-1)

    def __post_init__(self) -> None:
        self.virtual = self.var.func.is_fmaprod_two()

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
            arg.vopt = VData(arg)

        for func in funcs:
            func.fopt = FData(func)
            func.fopt.missing_args = set(func.args)
            func.fopt.acc_count = 0
            for arg in func.args:
                assert arg in self.args
                arg.vopt.missing_contribs.update([func])

        self.funcs = funcs
        self.ready: list[Func] = []
        for f in sorted(funcs, key=lambda f: f.is_store()):
            self.check_if_ready(f)

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
            var.vopt.live = True
            if not var.vopt.virtual:
                assert var.vopt.register == -1
                var.vopt.register = self.current_stack
                self.current_stack += 1
                self.max_stack = max(self.max_stack, self.current_stack)

    def pop_stack(self, var: Var) -> None:
        """Remove a variable from stack."""
        assert var.vopt.is_live()
        var.vopt.live = False
        if not var.vopt.virtual:
            self.current_stack -= 1
        assert self.current_stack >= 0

    def use_var(self, func: Func, var: Var) -> None:
        """Use a variable in a function."""
        # print("Remove contrib: ", func, var)
        assert var not in func.fopt.missing_args
        var.vopt.missing_contribs -= Counter([func])

        if var.vopt.missing_contribs.total() == 0:
            self.pop_stack(var)

        elif sum(n > 0 for n in var.vopt.missing_contribs.values()) == 1:
            next(iter(var.vopt.missing_contribs)).fopt.reg_preassure += 1

    def check_if_ready(self, func: Func) -> None:
        """Check if a function is ready"""
        if not func.fopt.is_not_ready():
            return
        elif func.is_acc():
            ready = len(func.fopt.missing_args) <= func.n_args - 2
        # elif func.is_store():
        #     ready = any(not f.fopt.is_not_ready() for f in func[0].vopt.missing_contribs)
        else:
            ready = not func.fopt.missing_args

        if ready:
            func.fopt.ready()
            self.ready.insert(0, func)

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
            for contrib in out.vopt.missing_contribs:
                contrib.fopt.update_aff1(1, -self.turn)

        for out in func.outs:
            out.vopt.live = True
            for contrib in out.vopt.missing_contribs.copy():
                contrib.fopt.missing_args.remove(out)
                if contrib.is_acc() and contrib.fopt.is_started():
                    self.accumulate(contrib, out, contrib.outs[0])
                self.check_if_ready(contrib)

    def start_accumulate(self, func: Func) -> None:
        """Start accumulating a function."""
        # print("Start accumulate: ", func)

        func.fopt.start()

        live_args = [v for v in func.args if v.vopt.is_live()]
        first = max(live_args, key=lambda v: v.vopt.missing_contribs.keys() <= {func})
        if not func.is_fmaprod_two():
            self.use_var(func, first)
        func.fopt.acc_count += 1

        for i, v in enumerate(a for a in live_args if a is not first):
            self.accumulate(func, v, first if i == 0 else func.outs[0])
        self.allocate(func.outs)

    def accumulate(self, func: Func, var: Var, prev: Var) -> None:
        func.fopt.acc_count += 1

        if func.is_fmaprod():
            if func.fopt.acc_count == func.n_args:
                if func.fopt.fma_parent.fopt.fma_need_one:
                    self.use_var(func, var)
                    self.ops.append((func, var, prev))
                    self.finish_func(func)
                else:
                    func.fopt.fma_prev = prev
                    func.fopt.fma_last = var
            else:
                self.use_var(func, var)
                self.ops.append((func, var, prev))

        if func.is_fma() and var.func.is_fmaprod():
            if not var.vopt.virtual:
                self.use_var(var.func, var.func.fopt.fma_prev)
            self.use_var(var.func, var.func.fopt.fma_last)
            self.use_var(func, var)
            self.ops.append((func, var.func.fopt.fma_last, var.func.fopt.fma_prev, prev))

        else:
            self.use_var(func, var)
            self.ops.append((func, var, prev))

        if func.fopt.acc_count == func.n_args:
            self.finish_func(func)

    def reorder(self) -> None:
        t0 = time.perf_counter()
        for self.turn in range(len(self.funcs)):
            # print(_)
            not_ready = [f for f in self.ready if f.fopt.is_not_ready()]
            func = max(self.ready, key=lambda f: f.fopt)
            assert func.fopt.state == 0
            costs = {f: f.fopt.key() for f in self.ready}
            # pprint(costs)
            print(func)
            self.ready.remove(func)
            # print(func.__class__.__name__)
            if func.is_acc():  # accumulate
                self.start_accumulate(func)
            else:
                self.do_func(func)

        assert all(f.fopt.is_finished() for f in self.funcs)
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

            arg_str = [f"r{a.vopt.register}" for a in args]
            for out in outs:
                ssa_regmap[out] = count
                count += 1
            out_str = [f"r{a.vopt.register}" for a in outs]
            print(f"{func.print(out_str, arg_str):<50}")
        print(self.max_stack)
