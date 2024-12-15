# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
from pathlib import Path
from caspar.cuda.lib_loader import CuLib


mytools = CuLib(Path(__file__).parent / "src")
# summarize_f32 = mytools.get_kernel("summarize_f32")
# summarize_f64 = mytools.get_kernel("summarize_f64")
get_n_matches = mytools.get_kernel("get_n_matches")
get_n_ops = mytools.get_kernel("get_n_ops")
get_ops = mytools.get_kernel("get_ops")
evaluate = mytools.get_kernel("evaluate")
# evaluate = None
3953532
