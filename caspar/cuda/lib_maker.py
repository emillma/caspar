# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from itertools import chain
from pathlib import Path
import hashlib

import numpy as np
import torch
import jinja2
import symforce.symbolic as sf
from symforce.ops import LieGroupOps as Ops
from .get_fixed_expression import get_fixed_expressions
from .cuda_printer import MyCudaCodePrinter
from .get_expr_order import get_best_order
from .kernel import Kernel
from .color import color

INCLUDE_FILES = list(Path(__file__).parent.joinpath("templates").glob("*.cuh"))
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent.joinpath("templates")),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.StrictUndefined,
)
printer = MyCudaCodePrinter()


def get_edges(exprs):
    todo = Ops.to_storage(exprs)
    graph = {}
    while todo:
        n_nodes = todo.pop()
        if n_nodes in graph or n_nodes.is_Number:
            continue
        args = list(set(a for a in getattr(n_nodes, "args", [])))
        graph[n_nodes] = args
        todo.extend(args)

    key_map = {n: i for i, n in enumerate(graph)}
    dep_edges = np.array(
        [
            (key_map[n], key_map[a])
            for n, args in graph.items()
            for a in args
            if not a.is_Number
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    return dep_edges, list(graph.keys())


def add_function(
    out_dir: Path,
    fname: str,
    args: dict[str, sf.Storage],
    outs: dict[str, sf.Storage],
    *,
    row_vecs: list[str] = [],
    vec_idx: list[str] = [],
    vec_idx_unq: list[str] = [],
    vec_idx_map: dict[str, str] = dict(),
    vec_idx_unq_map: dict[str, str] = dict(),
    atomic_add: list[str] = [],
    scalar: list[str] = [],
    unique: list[str] = [],
    double: list[str] = [],
    develop=False,
):

    if double == "all":
        double = list(args) + list(outs)

    for name in scalar:
        assert name in args
    for name in chain(row_vecs, vec_idx, vec_idx_unq, double, unique):
        assert name in args or name in outs
    for name in atomic_add:
        assert name in outs and name not in (vec_idx + vec_idx_unq)

    vec_idx = [
        a for a in (vec_idx + atomic_add) if a not in (unique + list(vec_idx_map))
    ]
    vec_idx_unq = [*vec_idx_unq]
    vec_idx_map = {k: k for k in vec_idx} | vec_idx_map
    vec_idx_unq_map = {k: k for k in vec_idx_unq} | vec_idx_unq_map

    for name in vec_idx_map.values():
        assert name in vec_idx
    for name in vec_idx_unq_map.values():
        assert name in vec_idx_unq

    args = {k: Ops.to_storage(v) for k, v in args.items()}
    done = set()
    for a in (a for storage in args.values() for a in storage):
        assert a not in done and a.is_Symbol
        done.add(a)

    outs = {k: Ops.to_storage(v) for k, v in outs.items()}
    hash_iter = (
        hash(e) for d in [args, outs] for s in d.values() for e in Ops.to_storage(s)
    )
    io_hash = "".join(hex(b) for h in hash_iter for b in h.to_bytes(8, "big"))
    hstr = f"{fname}{io_hash}{row_vecs}{vec_idx_map}{vec_idx_unq_map}{scalar}{atomic_add}{double}{unique}"
    out_file = out_dir / f"{fname}.cu"
    start = f"//  {hashlib.md5(hstr.encode()).hexdigest()}\n"
    if out_file.exists() and out_file.read_text().startswith(start):
        return

    args = {k: Ops.to_storage(v) for k, v in args.items()}
    outs = {k: get_fixed_expressions(v) for k, v in outs.items()}
    sizes = {k: len(v) for d in [args, outs] for k, v in d.items()}

    exprs = [e for vals in outs.values() for e in vals]
    edges, keys_inv = get_edges(exprs)
    # color(exprs)

    if edges.size == 0:
        max_rexisters, order = 1, []
    else:
        max_rexisters, order = get_best_order(edges, 320000)

    def getitem(name, row, col):
        if name in unique:
            return f"{name}[{row}]"
        elif name in row_vecs:
            return f"{name}[{row}+{name}_stride*{col}]"
        else:
            return f"{name}[{row}*{name}_stride+{col}]"

    def write_name(name, idx, stack_var: sf.Expr):
        t = "double" if name in double else "float"
        if name in atomic_add:
            if stack_var.is_zero:
                return ""
            elif name in unique:
                return (
                    f"write_unique<{t},{idx}>("
                    "warp, "
                    f"{stack_var}, "
                    "shared_out, "
                    f"{name});\n"
                )
            else:
                return (
                    f"write_together<{t},{idx},{str(name in row_vecs).lower()}>("
                    f"{vec_idx_map[name]}_group, "
                    f"{vec_idx_map[name]}_vec_idxs_shared, "
                    f"{vec_idx_map[name]}_ord, "
                    f"{stack_var}, "
                    f"shared_out, "
                    f"{name}, "
                    f"{name}_stride);\n"
                )
        elif name in vec_idx_unq:
            el = f"{vec_idx_unq_map[name]}_vec_idx"
            return f"{getitem(name, idx, el)} = {stack_var};\n"

        else:
            return f"{getitem(name, idx, 'gtrank')} = {stack_var};\n"

    def read_name(name, idx, stack_var: sf.Expr):
        t = "double" if name in double else "float"
        if name in vec_idx:
            return (
                f"{stack_var} = "
                f"read_together<{t}, {idx}, {str(name in row_vecs).lower()}>("
                f"{vec_idx_map[name]}_vec_idxs_shared,"
                f"{vec_idx_map[name]}_ord,"
                f"shared_in,"
                f"{name},"
                f"{name}_stride);\n"
            )
        elif name in vec_idx_unq:
            el = f"{vec_idx_unq_map[name]}_vec_idx"
            return f"{stack_var} = {getitem(name, idx, el)};\n"

        elif name in unique:
            return f"{stack_var} = read_unique<{t},{idx}>(shared_in, {name});\n"

        elif name in scalar:
            return f"{stack_var} = {name};\n"

        else:
            return f"{stack_var} = {getitem(name, idx, 'gtrank')};\n"

    arg_map = {}
    for name, arg in args.items():
        for j, expr in enumerate(arg):
            arg_map[expr] = (name, j)

    out_map = {}
    for name, arg in outs.items():
        for j, expr in enumerate(arg):
            out_map.setdefault(expr, []).append((name, j))

    stack = [sf.Symbol(f"stack[{i}]") for i in range(max_rexisters)]
    stored = dict()
    done = set()
    lines = ""
    use_double = any(name in double for name in chain(args))
    printer = MyCudaCodePrinter(double=use_double)
    for expr in [keys_inv[n] for n in order]:
        if expr in done:
            assert expr in stored
            stack.append(stored[expr])
            continue
        done.add(expr)

        stack_var = stored[expr] = stack.pop()
        if expr.is_Symbol:
            name, idx = arg_map[expr]
            lines += read_name(name, idx, stack_var)
        else:
            fargs = [stored.get(a, a) for a in expr.args]
            caller = getattr(expr, "func", type(expr))
            call = printer.doprint(caller(*fargs))
            lines += f"{stack_var} = {call}; //{max_rexisters  -len(stack)}\n"

        for name, idx in out_map.get(expr, []):
            lines += write_name(name, idx, stack_var)

    const_lines = ""
    for expr, outls in out_map.items():
        if expr in done:
            continue
        elif expr.is_Number:
            for name, idx in outls:
                lines += write_name(name, idx, expr)
        else:
            for name, idx in outls:
                # raise NotImplementedError
                name_in, idx_in = arg_map[expr]
                stack_var = stored[expr] = stack.pop()
                const_lines += read_name(name_in, idx_in, stack_var)
                const_lines += write_name(name, idx, stack_var)
                stack.append(stack_var)

    code = env.get_template("core.cu.jinja").render(
        fname=fname,
        arg_names=list(args),
        out_names=list(outs),
        size_dict=sizes,
        lines=lines,
        const_lines=const_lines,
        vec_idx=vec_idx,
        vec_idx_unq=vec_idx_unq,
        atomic_add=atomic_add,
        stack_size=max_rexisters,
        scalar=scalar,
        unique=unique,
        double=double,
        include_files=INCLUDE_FILES,
        stack_type="double" if use_double else "float",
        enumerate=enumerate,
        set=set,
    )

    out_file.write_text(start + code)


def call_generated(kernel: Kernel, args, outs, prob_size, vec_idx=[]):
    assert not isinstance(vec_idx, torch.Tensor)
    kargs = []
    for a in chain(args, outs):
        kargs.append(a)
        if isinstance(a, torch.Tensor):
            assert a.dim() <= 2
            kargs.append(a.stride(0))
    kargs.extend(vec_idx)

    out = kernel.launch_n(prob_size)(*kargs, prob_size, True)
    return out
