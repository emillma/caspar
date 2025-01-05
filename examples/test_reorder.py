import itertools
from random import randint
from typing import Iterable

import symforce

symforce.set_epsilon_to_symbol()
# Setup
import numpy as np
import sympy

import symforce.symbolic as sf
from caspar.cuda.allocation import Problem
from caspar.cuda.allocation import Solver
from symforce.values import Values

symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")

# Set epsilon to a symbol for safe code generation.  For more information, see the Epsilon tutorial:
# https://symforce.org/tutorials/epsilon_tutorial.html

import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util
from symforce.values import Values


def az_el_from_point(a: sf.Matrix55, b: sf.Matrix55) -> sf.Matrix55:
    """
    Transform a nav point into azimuth / elevation angles in the
    camera frame.

    Args:
        nav_T_cam (sf.Pose3): camera pose in the world
        nav_t_point (sf.Matrix): nav point
        epsilon (Scalar): small number to avoid singularities

    Returns:
        sf.Matrix: (azimuth, elevation)
    """
    return a * b


_old = sf.cse


def cse(*args, **kwargs):
    return _old(*args, **kwargs)


sf.cse = cse
az_el_codegen = codegen.Codegen.function(
    func=az_el_from_point,
    config=codegen.CppConfig(),
)
az_el_codegen_data = az_el_codegen.generate_function()


print("Files generated in {}:\n".format(az_el_codegen_data.output_dir))
for f in az_el_codegen_data.generated_files:
    print("  |- {}".format(f))


def tmp_symbols() -> Iterable[sf.Symbol]:
    for i in itertools.count():
        yield sf.Symbol(f"_tmp{i}")


if __name__ == "__main__":
    for i in range(1):
        letters = [chr(randint(97, 122)) for _ in range(6)]
        letters = [chr(97 + i) for i in range(6)]
        a, b, c, d, e, f = sf.symbols(" ".join(letters))
        exprs = [a * b + c]
        # exprs = [sf.sin((a + 1) * 2)]
        # exprs = [sf.sin(a + b) + (c * 3)]

        # for i in a:
        #     a.append(i + 1)
        # exprs = [a * a + b + c + d]
        # exprs = [sf.sin(sf.sin(sf.sin(a)))]
        # sf.sympify
        # sympy.factor(sf.sympify(exprs[0]).expand())
        A = sf.Matrix22.symbolic("a")
        B = sf.Matrix22.symbolic("b")

        B.subs((B[0], B[1]), (B[1], B[0]))
        # exprs = A.inv().to_storage()
        exprs = Values(foo=(A * B), bar=sf.Pose3.symbolic("pose").to_tangent())
        # exprs = Values(bar=exprs, baz=sf.Pose2.symbolic("pose"))
        funcs = list(Problem(exprs).funcs())
        # aff1, aff2 = prepare(funcs)
        reorderer = Solver(funcs)
        reorderer.reorder()
        reorderer.format_reordering()
        # reorderer.get_cse(tmp_symbols())
