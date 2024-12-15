# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from ctypes import cdll

from ctypes import pointer, c_void_p, c_char_p
from pathlib import Path

_cuda = cdll.LoadLibrary("libcuda.so")
_cudart = cdll.LoadLibrary("libcudart.so")


def get_launcher(cooperative: bool):
    return _cuda.cuLaunchCooperativeKernel if cooperative else _cuda.cuLaunchKernel


def check(ret_code: int):
    if ret_code:
        while code := _cudart.cudaGetLastError():
            print(_cudart.cudaGetErrorString(code))
        out = c_char_p()
        _cuda.cuGetErrorString(ret_code, pointer(out))
        raise ValueError(out.value.decode())


def get_module(cubin: Path):
    module = c_void_p()
    check(_cuda.cuModuleLoad(pointer(module), str(cubin).encode()))
    return module


def unload_module(module):
    check(_cuda.cuModuleUnload(module))


def get_function(module, name: str):
    kernel = c_void_p()
    check(_cuda.cuModuleGetFunction(pointer(kernel), module, name.encode()))
    return kernel
