# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import numpy as np
import numba as nb
from multiprocessing import cpu_count

# nb.config.DISABLE_JIT = True

# TODO burn this file and write the search algorithm in C++


@nb.njit
def get_order(
    args: np.ndarray,
    args_n: np.ndarray,
    args_ptr: np.ndarray,
    funcs: np.ndarray,
    funcs_n: np.ndarray,
    funcs_ptr: np.ndarray,
    roots: np.ndarray,
    quit_if_worse=0xFFFFFFFF,
):
    n_nodes = len(args_ptr)
    order = []
    to_load = []
    roots = list(roots)

    is_loaded = np.zeros(n_nodes)
    missing_args = args_n.copy()
    missing_funcs = funcs_n.copy()
    total = 0
    total_max = 0
    while roots:
        to_load.append(roots.pop(0))

        while to_load:
            node = to_load[0]
            if is_loaded[node]:
                to_load.pop(0)

            elif missing_args[node]:
                for arg in args[args_ptr[node] : args_ptr[node + 1]]:
                    if not is_loaded[arg]:
                        to_load.insert(0, arg)
            else:
                to_load.pop(0)
                is_loaded[node] = True
                for arg in args[args_ptr[node] : args_ptr[node + 1]]:
                    missing_funcs[arg] -= 1
                    if not missing_funcs[arg]:
                        total -= 1
                        order.append(arg)

                order.append(node)
                total += 1
                if total >= quit_if_worse:
                    return quit_if_worse, order
                total_max = max(total, total_max)

                if not missing_funcs[node]:
                    total -= 1
                    order.append(node)

                for func in funcs[funcs_ptr[node] : funcs_ptr[node + 1]]:
                    if not is_loaded[func]:
                        to_load.append(func)
                        missing_args[func] -= 1

    return total_max, order


@nb.njit
def get_order_random(
    edges,
    n_per_thread,
    n_nodes,
):
    args = edges[:, 1].copy()
    args_n = np.bincount(edges[:, 0], minlength=n_nodes)
    args_ptr = np.cumsum(np.array([0, *args_n]))

    funcs = edges[np.argsort(edges[:, 1]), 0].copy()
    funcs_n = np.bincount(edges[:, 1], minlength=n_nodes)
    funcs_ptr = np.cumsum(np.array([0, *funcs_n]))
    roots = np.where(args_n == 0)[0]

    new_names = np.arange(n_nodes)
    total_best = 0xFFFFFFFF
    order_best = [1]

    for i in range(n_per_thread):
        np.random.shuffle(new_names)
        for j in range(n_nodes):
            args_slice = args[args_ptr[j] : args_ptr[j + 1]]
            args_slice[:] = args_slice[np.argsort(new_names[args_slice])]
            funcs_slice = funcs[funcs_ptr[j] : funcs_ptr[j + 1]]
            funcs_slice[:] = funcs_slice[np.argsort(new_names[funcs_slice])]
        roots = roots[np.argsort(new_names[roots])]
        total, ord = get_order(
            args, args_n, args_ptr, funcs, funcs_n, funcs_ptr, roots, total_best
        )
        if total < total_best:
            total_best = total
            order_best = ord

    return total_best, order_best


@nb.njit(parallel=True, cache=True)
def get_best_order(
    edges: np.ndarray,
    attempts: int,
    cpu_count: int = cpu_count(),
):
    n_nodes = np.amax(edges) + 1
    n_per_thread = attempts // cpu_count

    totals = np.empty(cpu_count, np.int64)
    orders = [[0]] * cpu_count
    for i in nb.prange(cpu_count):
        total, order = get_order_random(
            edges=edges,
            n_per_thread=n_per_thread,
            n_nodes=n_nodes,
        )
        totals[i] = total
        orders[i] = order
    best = np.argmin(totals)
    return totals[best], orders[best]
