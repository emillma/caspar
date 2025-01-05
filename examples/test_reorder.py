import itertools
from random import randint, random
from typing import Iterable

import symforce

symforce.set_epsilon_to_symbol()
symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")


import symforce.symbolic as sf
from caspar.cuda.allocation import Problem
from caspar.cuda.allocation import Solver
from symforce import codegen
from symforce.ops import StorageOps as Ops
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


def verify(input: Values, output: Values, lines: list[str]) -> None:
    locals().update(
        {k: Ops.to_storage(v) if Ops.storage_dim(v) > 1 else v for k, v in inputs.items()}
    )
    locals().update({k: [None] * Ops.storage_dim(v) for k, v in exprs.items()})

    rsqrt = lambda x: x**-0.5
    min = sf.Min
    max = sf.Max
    acos = sf.acos
    sign = sf.sign
    sqrt = sf.sqrt
    atan2 = sf.atan2
    norm = lambda *x: sf.Add(*[i**2 for i in x]) ** 0.5
    for line in lines:
        exec(line)

    for k, v in output.items():
        for va, vb in zip(Ops.to_storage(locals()[k]), Ops.to_storage(v)):
            if not va == vb:
                vmap = {s: random() for s in va.free_symbols}
                assert abs(va.subs(vmap).evalf() - vb.subs(vmap).evalf()) < 1e-12


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
        inputs = Values(a=A, b=B, p=sf.Pose3.symbolic("pose"), eps=sf.epsilon())
        exprs = Values(
            # foo=(A * B),
            bar=sf.Pose3.symbolic("pose").to_tangent(),
            normed=A.row(0).norm(0),
        )
        # inputs = Values(eps=sf.epsilon())
        # exprs = Values(out=sf.epsilon())
        # exprs = Values(bar=exprs, baz=sf.Pose2.symbolic("pose"))

        prob = Problem(inputs, exprs)

        reorderer = Solver(list(prob.funcs()))
        reorderer.reorder()
        lines = reorderer.format_reordering()

        verify(inputs, exprs, lines)
        # reorderer.get_cse(tmp_symbols())
