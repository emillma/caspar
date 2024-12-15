# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from subprocess import run
from pathlib import Path
from ctypes import pointer, c_void_p, cast
import torch

from .kernel import Kernel
from .dlls import get_module, get_function, unload_module

torch.cuda.init()
torch.cuda.synchronize()


def any_newer(file: Path, compare: list[Path]):
    ftime = file.stat().st_mtime
    return any(f.stat().st_mtime > ftime for f in compare)


class CuLib:

    def __init__(self, code_dir: Path, cufiles=None, force=False):
        self.funcions = {}

        code_dir = Path(code_dir)
        bin_f = code_dir.joinpath("out.cubin")
        # TODO: use cufiles if provided
        cufiles = list(code_dir.glob("*.cu"))

        if force or not bin_f.exists() or any_newer(bin_f, cufiles):
            cmd = (
                "nvcc -fatbin -arch=native -dlink -Xptxas=-v "
                # "-G "
                "--use_fast_math --optimize=3 "
                f"-o={bin_f} {' '.join(map(str, cufiles))}"
            )
            run(cmd.split(), check=True)
        self.module = get_module(bin_f)
        torch.cuda.synchronize()

    def __del__(self):
        unload_module(self.module)

    def get_kernel(self, func_name: str, cooperative=False):
        if func_name not in self.funcions:
            self.funcions[func_name] = get_function(self.module, func_name)
        func = self.funcions[func_name]
        return Kernel(kernel=func, name=func_name, cooperative=cooperative)
