# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from itertools import combinations
from itertools import pairwise
from itertools import product
from pprint import pprint
from typing import TYPE_CHECKING, Any
from typing import Iterable

import symforce.symbolic as sf

from . import ftypes
from .allocator import Func
from .allocator import Problem
from .allocator import Var
from .allocator import ftypes
from .ftypes import Func
from .ftypes import Var


def prepare(funcs: list[Func]) -> tuple[dict[Func, dict[Var, int]], dict[Func, dict[Var, int]]]:
    aff1: dict[Func, dict[Var, int]] = {}
    for func in (f for f in funcs if not f.is_acc()):
        done = set()
        for var0, aff_var in product(func.args, func.args):
            if (aff_func := var0.func, aff_var) in done or var0 == aff_var:
                continue
            done.add((aff_func, aff_var))
            aff1.setdefault(aff_func, {}).setdefault(aff_var, 0)
            aff1[aff_func][aff_var] += 1

    aff2: dict[Func, dict[Var, int]] = {}
    for func, arg2aff in aff1.items():
        done = set()
        for arg0, aff_var in combinations(arg2aff, 2):
            if (aff_func := arg0.func, aff_var) in done:
                continue
            done.add((aff_func, aff_var))
            aff2.setdefault(aff_func, {}).setdefault(arg0, 0)
            aff2[aff_func][arg0] += 1

    return aff1, aff2


class Solver:
    def __init__(
        self,
        funcs: list[Func],
        aff1: dict[Func, dict[Var, int]],
        aff2: dict[Func, dict[Var, int]],
    ):
        self.aff1 = aff1
        self.aff2 = aff2

        self.missing_arg = {f: set(f.args) for f in funcs}
        self.missing_acc = {f: set(f.args) for f in funcs if f.is_acc()}
        assert all(len(v) == f.n_args for f, v in self.missing_acc.items())

        self.missing_contrib: dict[Var, set[Func]] = {}
        for fma, var in ((f, v) for f in funcs for v in f.args):
            self.missing_contrib.setdefault(var, set()).add(fma)

        self.ready = {k for k, v in self.missing_arg.items() if not v}
        self.not_ready = {k for k in self.missing_arg if k not in self.ready}

        # self.fma2fmaprods: dict[Func, set[ftypes.FmaProd]] = {}
        self.fma_parents: dict[Func, Func] = {}
        self.fma_waiting: dict[ftypes.FmaProd, Var] = {}
        self.fma_target: dict[ftypes.Fma, Var | None] = {}
        # self.fma_pure: set[Func] = {
        #     f for f in funcs if f.is_fma() and all(arg.func.is_fmaprod() for arg in f.args)
        # }
        for fma in (f for f in funcs if f.is_fma()):
            for fmaprod in (a.func for a in fma.args if a.func.is_fmaprod()):
                # self.fma2fmaprods.setdefault(fma, set()).add(fmaprod)
                self.fma_parents[fmaprod] = fma

        self.reg_count = 0
        self.available_regs: list[int] = []
        self.regmap: dict[Var, int] = {}
        self.live_vars: set[Var] = set()
        self.started: set[Func] = set()
        self.ops: list = []

    def allocate_regs(self, vars: Iterable[Var]) -> None:
        for var in vars:
            if not self.available_regs:
                self.available_regs.append(self.reg_count)
                self.reg_count += 1
            self.regmap[var] = self.available_regs.pop()

    def remove_contrib(self, func: Func, var: Var) -> None:
        print("Remove contrib: ", func, var)
        assert var not in self.missing_arg[func]
        self.missing_contrib[var].remove(func)
        if func.is_acc():
            self.missing_acc[func].remove(var)

        if not self.missing_contrib[var]:
            self.available_regs.append(self.regmap[var])
            self.live_vars.remove(var)

    def check_if_ready(self, func: Func) -> None:
        if func not in self.not_ready:
            return

        if func.is_acc():
            n_min = 2
            if func.is_fma():
                n_min = min(sum(not arg.func.is_fmaprod() for arg in func.args), n_min)
            ready = len(self.missing_arg[func]) <= func.n_args - n_min

        else:
            ready = not self.missing_arg[func]

        if ready:
            self.not_ready.remove(func)
            self.ready.add(func)

    def do_func(self, func: Func) -> None:
        for v in func.args:
            self.remove_contrib(func, v)
        self.allocate_regs(func.outs)
        self.ops.append((func, *func.args))
        self.finish_func(func)

    def finish_func(self, func: Func) -> None:
        print("Finish: ", func)
        for out in func.outs:
            self.live_vars.add(out)
            for contrib in self.missing_contrib[out].copy():
                self.missing_arg[contrib].remove(out)
                if contrib.is_acc() and contrib in self.started:
                    self.accumulate(contrib, out)
                self.check_if_ready(contrib)

    def start_accumulate(self, func: Func) -> None:
        print("Start accumulate: ", func)
        self.started.add(func)
        live_args = [v for v in func.args if v in self.live_vars]
        if len(live_args) == 0:
            assert isinstance(func, ftypes.Fma)
            self.fma_target[func] = None
        elif len(live_args) == 1:
            assert isinstance(func, ftypes.Fma)
            self.fma_target[func] = live_args[0]

        elif len(live_args) >= 2:
            first = max(live_args, key=lambda v: self.missing_contrib[v] <= {func})
            self.remove_contrib(func, first)
            self.allocate_regs(func.outs)
            for i, v in enumerate(a for a in live_args if a is not first):
                self.accumulate(func, v, first if i == 0 else None)

        if func.is_fma():
            for prod in (a for a in func.args if a.func.is_fmaprod()):
                assert isinstance(prod.func, ftypes.FmaProd)
                if prod.func in self.fma_waiting:
                    self.accumulate(prod.func, self.fma_waiting.pop(prod.func))

    def accumulate(self, func: Func, var: Var, prev: Var | None = None) -> None:
        print("Accumulate: ", func, var)
        prev = func.args[0] if prev is None else prev
        if func.is_fmaprod() and len(self.missing_acc[func]) == 1:
            assert isinstance(func, ftypes.FmaProd)
            parent = self.fma_parents[func]
            assert isinstance(parent, ftypes.Fma)

            if parent not in self.started:
                assert isinstance(func, ftypes.FmaProd)
                self.fma_waiting[func] = var
                self.check_if_ready(self.fma_parents[func])
                return
            else:
                if parent not in self.fma_target:
                    self.ops.append((func, prev, var))
                elif self.fma_target[parent] is None:
                    self.allocate_regs(parent.outs)
                    self.ops.append((func, prev, var, parent[0]))
                else:
                    self.remove_contrib(parent, self.fma_target[parent])
                    self.allocate_regs(parent.outs)
                    self.ops.append((func, prev, var, self.fma_target[parent]))
                self.finish_func(func)
                return
        if not (func.is_fma() and var.func.is_fmaprod()):
            self.ops.append((func, prev, var))
        self.remove_contrib(func, var)
        if len(self.missing_acc[func]) == 0:
            self.finish_func(func)
            self.started.remove(func)

    def score(self, func: Func) -> tuple[int, ...]:
        # if func.is_load() and func.lit_args[0].data == "c":
        #     return -100, 0, 0, 0, 0, 0
        reg_preassure = (
            sum(self.missing_contrib[var] <= {func} for var in self.live_vars) - func.n_outs
        )
        removable = all(
            (f.is_acc() and f in self.started)
            for out in func.outs
            for f in self.missing_contrib[out]
        )

        a1 = a2 = finishable = 0

        if func.is_fma():
            finishable += sum(arg.func in self.fma_waiting for arg in func.args)
        if not self.missing_arg[func]:
            for other in set(f for out in func.outs for f in self.missing_contrib[out]):
                if self.missing_arg[other] <= set(func.outs):
                    finishable += 1
            a1 = sum(self.aff1.get(func, {}).get(r, 0) for r in self.live_vars)
            a2 = sum(self.aff2.get(func, {}).get(r, 0) for r in self.live_vars)

        return (
            reg_preassure,
            removable,
            finishable,
            a1,
            a2,
            -int(func.is_acc() and bool(self.missing_arg[func])),
        )

    def reorder(self) -> None:
        while self.not_ready or self.ready:
            scores = {call: self.score(call) for call in self.ready}
            func = max(self.ready, key=self.score)
            self.ready.remove(func)

            if func.is_acc():  # accumulate
                self.start_accumulate(func)
            else:
                self.do_func(func)

    def format_reordering(self) -> None:
        new_ordrer = []
        accs: dict[Func, list] = {}

        print("")
        for op in self.ops:
            func, *args = op
            print(
                "".join(f"{str(self.regmap[a]):3}" for a in func.outs),
                f"{str(func):<40}",
                "".join(f"{str(self.regmap[a]):3}" for a in args),
            )
