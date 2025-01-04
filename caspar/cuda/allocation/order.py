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
    missing_args: set[Var] = field(init=False)

    acc_parent: Func = field(init=False)
    acc_waiting: Func = field(init=False)
    acc_count: int = field(default=0)

    fma_parent: Func = field(default=None)
    fma_prev: Var = field(default=None)
    fma_last: Var = field(default=None)

    state: int = field(default=0)  # 0: not ready, 1: ready, 2: done

    reg_preassure: int = field(init=False)
    removable: list[bool] = field(default=0)
    firable: int = field(default=0)
    priority: int = field(default=-(2**32))
    aff2: int = field(default=0)
    aff1: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.missing_args = set(self.func.args)

        self.reg_preassure = -self.func.n_outs

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
        return self.state == 0

    def is_ready(self) -> bool:
        return self.state == 1

    def is_finished(self) -> bool:
        return self.state == 1

    def mark_ready(self) -> None:
        self.state = 1

    def mark_done(self) -> None:
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
    live: bool = field(default=False)
    missing_contribs: Counter[Func] = field(default_factory=Counter)
    register: int = field(default=-1)
    register_ssa: int = field(default=-1)

    def __post_init__(self) -> None:
        self.virtual = self.var.func.is_fmaprod_two()

    def is_live(self) -> bool:
        return self.live


class Solver:
    def __init__(self, funcs: list[Func]):
        self.args: set[Var] = {arg for func in funcs for arg in func.outs}

        for arg in self.args:
            arg.vopt = VData(arg)

        for func in funcs:
            func.fopt = FData(func)
            for arg in func.args:
                assert arg in self.args
                arg.vopt.missing_contribs.update([func])

        for func in (f for f in funcs if f.is_accumulator()):
            for arg in func.args:
                arg.func.fopt.acc_parent = func

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
        self.ssa_count = 0

    def allocate(self, variables: list[Var]) -> None:
        """Add a variable to stack."""

        for var in variables:
            assert not var.vopt.live and not var.is_virtual() and var.vopt.register == -1

            var.vopt.live = True

            var.vopt.register = self.current_stack
            self.current_stack += 1
            var.vopt.register_ssa = self.ssa_count
            self.ssa_count += 1
            self.max_stack = max(self.max_stack, self.current_stack)

    def pop_stack(self, var: Var) -> None:
        """Remove a variable from stack."""
        assert var.vopt.is_live() or var.is_virtual()
        var.vopt.live = False
        if not var.vopt.virtual:
            self.current_stack -= 1
        assert self.current_stack >= 0

    def use_var(self, func: Func, var: Var) -> None:
        """Use a variable in a function."""
        # print("Remove contrib: ", func, var)
        assert not var.is_virtual()
        var.vopt.missing_contribs -= Counter([func])

        if var.vopt.missing_contribs.total() == 0 and not var.is_virtual():
            self.pop_stack(var)

        elif sum(n > 0 for n in var.vopt.missing_contribs.values()) == 1:
            next(iter(var.vopt.missing_contribs)).fopt.reg_preassure += 1

    def check_if_ready(self, func: Func) -> None:
        """Check if a function is ready"""
        assert func.fopt.is_not_ready()
        if not func.fopt.is_not_ready():
            return
        else:
            ready = not func.fopt.missing_args

        if ready:
            func.fopt.mark_ready()
            self.ready.insert(0, func)

    def do_func(self, func: Func) -> None:
        """Do a function."""
        for v in func.args:
            self.use_var(func, v)
        self.allocate(func.outs)
        self.ops.append(func)
        for out in func.outs:
            self.finish_var(out)

    def finish_var(self, out: Var) -> None:
        """Finish a function."""

        for contrib in out.vopt.missing_contribs:
            contrib.fopt.update_aff1(1, -self.turn)

        out.vopt.live = True
        for contrib in out.vopt.missing_contribs.copy():
            contrib.fopt.missing_args.remove(out)
            self.check_if_ready(contrib)

    def do_contribute(self, func: ftypes.Contribute) -> None:
        acc = func.fopt.acc_parent

        if acc.fopt.acc_count == 0:
            acc.fopt.acc_waiting = func

        elif acc.fopt.acc_count == 1:
            self.use_var(acc.fopt.acc_waiting, acc.fopt.acc_waiting.args[0])
            self.use_var(func, func.args[0])

            self.allocate(acc.outs)
            acc.fopt.missing_args.remove(acc.fopt.acc_waiting[0])
            acc.fopt.missing_args.remove(func[0])

            self.ops.append((acc, acc.fopt.acc_waiting.args[0], func.args[0]))
        else:
            self.use_var(func, func.args[0])
            acc.fopt.missing_args.remove(func[0])
            self.ops.append((acc, acc[0], func.args[0]))

        acc.fopt.acc_count += 1
        if acc.fopt.acc_count == acc.n_args:
            self.check_if_ready(acc)

    def do_accumulator(self, func: Func) -> None:
        """Start accumulating a function."""
        # print("Start accumulate: ", func)

        self.finish_var(func.outs[0])
        # self.allocate(func.outs)
        # for acc in (f for f in func[0].vopt.missing_contribs if f.is_contrib()):
        #     acc.fopt.missing_args.remove(func[0])
        #     self.check_if_ready(acc)

    def reorder(self) -> None:
        t0 = time.perf_counter()
        for self.turn in range(len(self.funcs)):
            # print(_)
            not_ready = [f for f in self.funcs if f.fopt.is_not_ready()]
            costs = {f: f.fopt.key() for f in self.ready}
            func = max(self.ready, key=lambda f: f.fopt)

            # pprint(costs)
            # print(func)
            self.ready.remove(func)
            # print(func.__class__.__name__)
            if func.is_contrib():  # accumulate
                self.do_contribute(func)
            elif func.is_accumulator():
                self.do_accumulator(func)
            else:
                self.do_func(func)

        # assert all(f.fopt.is_finished() for f in self.funcs)
        print("Time: ", time.perf_counter() - t0)
        print(self.max_stack)

    def format_reordering(self) -> None:
        ssa_regmap = {}
        regmap = ssa_regmap
        count = 0
        print("")

        for func in self.ops:
            if isinstance(func, tuple):
                func, *args = func
            else:
                args = func.args

            arg_str = [f"r{a.vopt.register_ssa}" for a in args]

            out_str = [f"r{a.vopt.register_ssa}" for a in func.outs]
            print(f"{func.print(out_str, arg_str):<50}")
        print(self.max_stack)
