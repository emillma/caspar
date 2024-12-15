# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from itertools import chain
import numpy as np
import torch
from functools import cache
from .dlls import get_launcher, check

from ctypes import (
    pointer,
    c_void_p,
    c_float,
    c_double,
    cast,
    c_int64,
    c_bool,
    POINTER,
    c_char,
)


def arg_cast(arg):
    if isinstance(arg, bool):
        return cast(pointer(c_bool(arg)), c_void_p)

    elif isinstance(arg, int):
        return cast(pointer(c_int64(arg)), c_void_p)

    elif isinstance(arg, float):
        return cast(pointer(c_double(arg)), c_void_p)

    elif arg is None:
        return cast(pointer(c_void_p()), c_void_p)

    elif isinstance(arg, torch.Tensor):
        assert arg.is_cuda and (arg.stride(-1) == 1 or arg.size(-1) <= 1)
        return cast(pointer(c_int64(arg.data_ptr())), c_void_p)
    else:
        raise ValueError(f"Unsupported type {type(arg)}")


class Kernel:
    _kernels: dict[str, "Kernel"] = {}
    _initialized = set()

    @classmethod
    def reset_timing_data(cls):
        for kernel in cls._kernels.values():
            kernel.event_pairs = []

    def __new__(cls, *, name: str, **kwargs):
        if name not in cls._kernels:
            cls._kernels[name] = super().__new__(cls)
        return cls._kernels[name]

    def __init__(self, *, kernel: c_void_p, name: str, cooperative=False):
        if name in self.__class__._initialized:
            return
        self.kernel = kernel
        self.event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.launcher = get_launcher(cooperative)
        self.name = name
        self.__class__._initialized.add(name)

    def __hash__(self):
        return hash(self.name)

    @cache
    def launch_n(self, n, tpb=1024, shared_mem=0, stream=None):

        bpg = (n - 1) // tpb + 1 if n > 0 else 0
        return self.get_launcher(bpg, tpb, shared_mem, stream)

    @cache
    def get_launcher(self, bpg, tpb, shared_mem=0, stream=None):
        assert stream is None
        bpg = (bpg, 1, 1) if isinstance(bpg, int) else bpg
        tpb = (tpb, 1, 1) if isinstance(tpb, int) else tpb
        assert tpb[0] == 1024

        def inner(*args):
            if False:
                for arg in (a for a in args if isinstance(a, torch.Tensor)):
                    if arg.dtype in (torch.float32, torch.float64):
                        assert arg.isfinite().all()
            if any(e == 0 for e in chain(bpg, tpb)):
                return
            cuargs = (c_void_p * len(args))(*map(arg_cast, args))

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            self.event_pairs.append((start_event, end_event))
            start_event.record()
            check(
                self.launcher(
                    self.kernel,
                    *bpg,
                    *tpb,
                    shared_mem,
                    (stream or torch.cuda.current_stream())._as_parameter_,
                    cuargs,
                    None,
                )
            )
            end_event.record()

        return inner

    def __repr__(self):
        return f"Kernel({self.kernel})"
