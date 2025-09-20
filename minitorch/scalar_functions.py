from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Computes the forward pass for multiplication.

        Args:
            ctx: Context for saving information used in the backward pass.
            a: First scalar input.
            b: Second scalar input.

        Returns:
            Product of a and b.
        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backward pass for multiplication.

        Args:
            ctx: Context containing saved input values.
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Tuple containing gradients with respect to inputs a and b.
        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Computes the forward pass for inverse.

        Args:
            ctx: Context for saving information used in the backward pass.
            a: Scalar input.

        Returns:
            The reciprocal of a.
        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backward pass for inverse.

        Args:
            ctx: Context containing saved input values.
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradient with respect to the input a.
        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Computes the forward pass for negation.

        Args:
            ctx: Context for saving information.
            a: Scalar input.

        Returns:
            Negated value of a.
        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backward pass for negation.

        Args:
            ctx: Context (not used for negation).
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradient with respect to input a.
        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = \text{sigmoid}(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Computes the forward pass for the sigmoid function.

        Args:
            ctx: Context for saving information used in the backward pass.
            a: Scalar input.

        Returns:
            Sigmoid of a.
        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backward pass for sigmoid.

        Args:
            ctx: Context containing saved forward output.
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradient with respect to input a.
        """
        (out,) = ctx.saved_values  # out = sigmoid(a)
        return d_output * out * (1 - out)


class ReLU(ScalarFunction):
    r"""ReLU function $f(x) = \max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Computes the forward pass for ReLU.

        Args:
            ctx: Context for saving information used in the backward pass.
            a: Scalar input.

        Returns:
            Result of applying ReLU to a.
        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backward pass for ReLU.

        Args:
            ctx: Context containing saved input values.
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradient with respect to input a.
        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    r"""Exp function $f(x) = \exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Computes the forward pass for exponential.

        Args:
            ctx: Context for saving information used in the backward pass.
            a: Scalar input.

        Returns:
            Exponential of a.
        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backward pass for exponential.

        Args:
            ctx: Context containing saved forward output.
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradient with respect to input a.
        """
        (out,) = ctx.saved_values  # out = exp(a)
        return d_output * out


class LT(ScalarFunction):
    """Less-than function $f(x, y) = 1.0$ if $x < y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Computes the less-than indicator.

        Args:
            ctx: Context (not used).
            a: First input.
            b: Second input.

        Returns:
            1.0 if a < b, else 0.0.
        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backward pass for less-than.

        Args:
            ctx: Context (not used).
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradients with respect to inputs a and b, which are 0.0 since this function is not differentiable.
        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x, y) = 1.0$ if $x == y$ else $0.0$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Computes the equality indicator.

        Args:
            ctx: Context (not used).
            a: First input.
            b: Second input.

        Returns:
            1.0 if a == b, else 0.0.
        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backward pass for equality.

        Args:
            ctx: Context (not used).
            d_output: Gradient of the output with respect to some scalar.

        Returns:
            Gradients with respect to inputs a and b, which are 0.0 since this function is not differentiable.
        """
        return 0.0, 0.0
