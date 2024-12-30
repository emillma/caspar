# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.

from collections import Counter
from itertools import combinations
from itertools import product
from typing import Iterable

from . import ftypes
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
        self.args: set[Var] = {arg for func in funcs for arg in func.args}

        for arg in self.args:
            arg.missing_contribs = Counter()

        for func in funcs:
            func.missing_args = set(func.args)
            func.acc_count = 0
            for arg in func.args:
                arg.missing_contribs[func] += 1
        self.funcs = funcs
        self.ready = {f for f in funcs if not f.missing_args}
        self.not_ready = {k for k in funcs if k not in self.ready}

        # self.fma2fmaprods: dict[Func, set[ftypes.FmaProd]] = {}
        self.fma_waiting: dict[ftypes.Func, Var] = {}

        self.reg_count = 0
        self.regmap: dict[Var, int] = {}
        self._stack: list[int] = []
        self.live_vars: set[Var] = set()
        self.started_acc: set[Func] = set()
        self.ops: list = []
        self.max_stack = 0
        self.current_stack = 0

    def allocate(self, add: list[Var]) -> None:
        """Add a variable to stack."""
        for var in add:
            if var in self.regmap:
                continue
            if not self._stack:
                self.max_stack = self.max_stack + 1
                self._stack.append(self.max_stack - 1)
            self.regmap[var] = self._stack.pop()

    def pop_stack(self, var: Var) -> None:
        """Remove a variable from stack."""
        self._stack.append(self.regmap[var])

    def use_var(self, func: Func, var: Var) -> None:
        """Use a variable in a function."""
        print("Remove contrib: ", func, var)
        assert var not in func.missing_args
        var.missing_contribs[func] -= 1
        if func.is_acc():
            func.acc_count += 1

        if var.missing_contribs.total() == 0:
            self.pop_stack(var)

    def check_if_ready(self, func: Func) -> None:
        """Check if a function is ready"""
        if func not in self.not_ready:
            return
        elif func.is_fma():
            if func.is_fma_none() or func.is_fma_one():
                ready = any(arg.func.outs[0] in self.live_vars for arg in func.args)
            elif func.is_fma_many():
                ready = len(func.missing_args) <= func.n_args - 2

        elif func.is_fmaprod():
            parent = next(iter(func[0].missing_contribs))
            if func.is_fmaprod_two():
                ready = not func.missing_args
                if parent.is_fma_one():
                    ready = ready and parent.args[0] in self.live_vars
                elif parent.is_fma_many():
                    ready = ready and parent in self.started_acc
            else:
                ready = len(func.missing_args) <= func.n_args - 2
        elif func.is_acc():
            ready = len(func.missing_args) <= func.n_args - 2

        else:
            ready = not func.missing_args

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
        print("Finish: ", func)
        for out in func.outs:
            self.live_vars.add(out)
            for contrib in out.missing_contribs.copy():
                contrib.missing_args.remove(out)
                if contrib.is_acc() and contrib in self.started_acc:
                    self.accumulate(contrib, out, contrib.outs[0])
                self.check_if_ready(contrib)

    def start_fma(self, func: Func) -> None:
        """Start an FMA function."""
        print("Start fma: ", func)
        if func.is_fma_none():
            self.start_accumulate(func)
        if func.is_fma_one():
            self.start_accumulate(func)
        if func.is_fma_many():
            self.start_accumulate(func)

        for prod in (arg.func for arg in func.args if arg.func.is_fmaprod()):
            if prod in self.fma_waiting:
                self.accumulate(prod, self.fma_waiting.pop(prod), prod.outs[0])
            self.check_if_ready(prod)

    def start_fmaprod(self, func: Func) -> None:
        """Start accumulating a function."""
        print("Start fmaprod: ", func)
        parent = next(iter(func[0].missing_contribs))
        if func.is_fmaprod_two():
            if parent.is_fma_none() and parent not in self.started_acc:
                self.do_func(func)

            else:
                self.use_var(func, func.args[0])
                self.use_var(func, func.args[1])
                self.ops.append((func, func.args[0], func.args[1], parent.outs[0]))
                parent.acc_count += 1
                parent.missing_args.remove(func.outs[0])
                func.outs[0].missing_contribs[parent] -= 1
                if parent.acc_count == len(parent.args):
                    self.finish_func(parent)
                    self.started_acc.remove(parent)

        else:
            self.start_accumulate(func)

    def start_accumulate(self, func: Func) -> None:
        """Start accumulating a function."""
        print("Start accumulate: ", func)
        self.started_acc.add(func)
        live_args = [v for v in func.args if v in self.live_vars]
        first = max(live_args, key=lambda v: v.missing_contribs.keys() <= {func})
        self.use_var(func, first)
        for i, v in enumerate(a for a in live_args if a is not first):
            self.accumulate(func, v, first if i == 0 else func.outs[0])
        self.allocate(func.outs)

    def accumulate(self, func: Func, var: Var, prev: Var) -> None:
        print("Accumulate: ", func, var)

        if (
            func.is_fmaprod()
            and func.acc_count == len(func.args) - 1
            and (
                not (parent := next(iter(func[0].missing_contribs))).is_fma_none()
                and parent in self.started_acc
            )
        ):
            if parent not in self.started_acc:
                self.fma_waiting[func] = var
                self.check_if_ready(parent)
            else:
                parent.acc_count += 1
                parent.missing_args.remove(func.outs[0])
                func[0].missing_contribs[parent] -= 1
                self.use_var(func, var)
                self.pop_stack(func.outs[0])
                self.ops.append((func, var, prev, parent.outs[0]))
                self.started_acc.remove(func)
                if parent.acc_count == len(parent.args):
                    self.finish_func(parent)
                    self.started_acc.remove(parent)
        else:
            self.use_var(func, var)
            if not var.func.is_fmaprod():
                self.ops.append((func, var, prev))  # this is already done for fma_prods

            if func.acc_count == len(func.args):
                self.finish_func(func)
                self.started_acc.remove(func)

    def score(self, func: Func) -> tuple[int, ...]:
        freed = sum(var.missing_contribs.keys() <= {func} for var in self.live_vars)
        reg_preassure = freed - func.n_outs

        removable = all(
            (f.is_acc() and f in self.started_acc)
            for out in func.outs
            for f in out.missing_contribs
        )

        a1 = a2 = finishable = 0

        if func.is_fma():
            finishable += sum(arg.func in self.fma_waiting for arg in func.args)
        if not func.missing_args:
            for other in set(f for out in func.outs for f in out.missing_contribs):
                if other.missing_args <= set(func.outs):
                    finishable += 1
            a1 = sum(self.aff1.get(func, {}).get(r, 0) for r in self.live_vars)
            a2 = sum(self.aff2.get(func, {}).get(r, 0) for r in self.live_vars)

        return (
            reg_preassure,
            removable,
            finishable,
            a1,
            a2,
            -int(func.is_acc() and bool(func.missing_args)),
        )

    def reorder(self) -> None:
        while self.not_ready or self.ready:
            scores = {call: self.score(call) for call in self.ready}
            func = max(self.ready, key=self.score)
            self.ready.remove(func)
            if func.is_fma():
                self.start_fma(func)
            elif func.is_fmaprod():
                self.start_fmaprod(func)
            elif func.is_acc():  # accumulate
                self.start_accumulate(func)
            else:
                self.do_func(func)

    def format_reordering(self) -> None:
        new_ordrer = []
        accs: dict[Func, list] = {}
        count = 0
        print("")
        for op in self.ops:
            func, *args = op
            if func.is_fmaprod() and len(args) == 3:
                outs = [args[-1]]
            else:
                outs = [a for a in func.outs]

            print(
                "".join(f"{self.regmap[a]:5}" for a in outs),
                f"{str(func):<40}",
                "".join(f"{str(self.regmap[a]):5}" for a in args),
            )
        print(self.max_stack)
