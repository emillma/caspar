# CASPAR - Copyright 2024, Emil Martens, SFI Autoship, NTNU
# This source code is under the Apache 2.0 license found in the LICENSE file.
import logging
from sympy.printing.printer import Printer
from sympy.printing.c import C11CodePrinter

import symforce.symbolic as sf


class MyCudaCodePrinter(Printer):
    node_calls = set()
    unimplemented_calls = set()

    def __init__(self, *args, double=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.double = double
        self.postfix = "" if double else "f"
        self.prefix = "d" if double else "f"

    def _print_sign(self, expr):
        return f"copysign{self.postfix}(1.0{self.postfix}, {self._print(expr.args[0])})"

    def _print_sin(self, expr):
        return f"sin{self.postfix}({self._print(expr.args[0])})"

    def _print_asin(self, expr):
        return f"asin{self.postfix}({self._print(expr.args[0])})"

    def _print_cos(self, expr):
        return f"cos{self.postfix}({self._print(expr.args[0])})"

    def _print_acos(self, expr):
        return f"acos{self.postfix}({self._print(expr.args[0])})"

    def _print_tan(self, expr):
        return f"tan{self.postfix}({self._print(expr.args[0])})"

    def _print_atan(self, expr):
        return f"atan{self.postfix}({self._print(expr.args[0])})"

    def _print_atan2(self, expr):
        return f"atan2{self.postfix}({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_Abs(self, expr):
        return f"fabs{self.postfix}({self._print(expr.args[0])})"

    def _print_Min(self, expr):
        return f"fmin{self.postfix}({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_Max(self, expr):
        return f"fmax{self.postfix}({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_Pow(self, expr):
        return f"pow{self.postfix}({self._print(expr.args[0])}, {self._print(expr.args[1])})"

    def _print_log(self, expr):
        return f"log{self.postfix}({self._print(expr.args[0])})"

    def _print_exp(self, expr):
        return f"exp{self.postfix}({self._print(expr.args[0])})"

    def _print_Piecewise(self, expr):
        raise NotImplementedError("Piecewise not implemented")

    # Numbers
    def _print_Number(self, expr):
        return f"{float(expr.evalf()):.8e}{self.postfix}"

    # Custom functions
    def _print_Function(self, expr):
        args = [self._print(arg) for arg in expr.args]
        return getattr(self, f"_custom_{expr.get_name()}")(*args)

    # def _custom_nonzero(self, expr):
    #     x = expr.args[0]
    #     return f'asm volatile ("slct.f32.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b + 1.f), "f"())'

    def _custom_rsqrt(self, a):
        return f"rsqrt{self.postfix}({a})"

    def _custom_sqrt(self, a):
        return f"sqrt{self.postfix}({a})"

    def _custom_fma(self, a, b, c):
        return f"fma{self.postfix}({a}, {b}, {c})"

    def _custom_add2(self, a, b):
        return f"({a} + {b})"

    def _custom_mul2(self, a, b):
        return f"({a} * {b})"

    def _custom_norm2(self, a, b, c):
        return f"hypot{self.postfix}({a}, {b}, {c})"

    def _custom_norm3(self, a, b, c):
        return f"norm3d{self.postfix}({a}, {b}, {c})"

    def _custom_norm4(self, a, b, c, d):
        return f"norm4d{self.postfix}({a}, {b}, {c}, {d})"

    def _custom_rnorm2(self, a, b):
        return f"rhypot{self.postfix}({a}, {b})"

    def _custom_rnorm3(self, a, b, c):
        return f"rnorm3d{self.postfix}({a}, {b}, {c})"

    def _custom_rnorm4(self, a, b, c, d):
        return f"rnorm4d{self.postfix}({a}, {b}, {c}, {d})"

    def _custom_ternary(self, a, b, c):
        return f"({a} ? {b} : {c})"

    # def _custom_div(self, a, b):
    #     return f"fdivide({a}, {b})"

    def _custom_rcp(self, a):
        return f"__{self.prefix}rcp_rn({a})"

    def _print_Symbol(self, a):
        return a.name

    def emptyPrinter(self, expr):
        logging.debug(f"No custom printer for type {type(expr)}")
        return super().emptyPrinter(expr)

    # Misc
    def doprint(self, expr):
        return super().doprint(expr)

    def _format_code(self, expr):
        return expr

    def __getattr__(self, attr):
        self.node_calls.add(attr)
        if getattr(sf, attr, None) is None and attr not in self.unimplemented_calls:
            self.unimplemented_calls.add(attr)
        raise AttributeError(attr)
