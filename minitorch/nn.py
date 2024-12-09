from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor_ops import SimpleOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw
    # First, reshape to split height dimension: (batch, channel, new_height, kh, width)
    new = input.contiguous()
    new = new.view(batch, channel, new_height, kh, width)

    # Next, reshape to split width dimension: (batch, channel, new_height, kh, new_width, kw)
    new = new.view(batch, channel, new_height, kh, new_width, kw)

    # Finally, merge the kernel dimensions and rearrange to get:
    # (batch, channel, new_height, new_width, kh * kw)
    new = new.permute(0, 1, 2, 4, 3, 5)

    new = new.contiguous()
    new = new.view(batch, channel, new_height, new_width, kh * kw)

    return new, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Average pooling 2D function.

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling kernel

    Returns:
    -------
        Tensor of averages

    """
    # Reshape input for pooling using tile function
    tiled, new_height, new_width = tile(input, kernel)

    # Calculate sum over the pooling window (last dimension)
    kh, kw = kernel
    kernel_size = kh * kw

    # Sum over the pooling window dimension and divide by window size
    out = (tiled.sum(dim=4) / float(kernel_size)).contiguous()

    # Reshape to remove the extra dimension
    return out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3])


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of max.

        Args:
        ----
            ctx : Context
            a : tensor
            dim : int dimension to reduce along

        """
        max_result = SimpleOps.reduce(operators.max, start=-float("inf"))(
            a, int(dim.item())
        )

        ctx.save_for_backward(a, dim, max_result)
        return max_result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass of max.

        Args:
        ----
            ctx : Context
            grad_output : gradient of output

        Returns:
        -------
            : gradient of input, gradient of dimension (always 0)

        """
        input, dim, max_result = ctx.saved_values

        # Create mask where input equals max_result
        mask = input == max_result

        # Sum up the mask to count number of maxes
        num_maxes = mask.sum(dim=dim)

        # Broadcast grad_output / num_maxes back to input shape
        grad_input = mask * (grad_output / num_maxes)

        return grad_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Maximum over dimension `dim`"""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Applies the softmax function along a dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to compute softmax over

    Returns:
    -------
        Tensor of same shape as input with softmax applied along dim

    """
    # Subtract max for numerical stability (prevents overflow)
    shifted = input - max(input, dim).detach()
    exp_values = shifted.exp()
    sum_exp = exp_values.sum(dim=dim)
    # Reshape sum_exp to ensure proper broadcasting
    if dim == -1 or dim == input.dims - 1:
        new_shape = list(input.shape)
        new_shape[dim] = 1
        sum_exp = sum_exp.view(*new_shape)
    return exp_values / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Applies the log softmax function along a dimension.

    Args:
    ----
        input: input tensor
        dim: dimension to compute log softmax over

    Returns:
    -------
        Tensor of same shape as input with log softmax applied along dim

    """
    # Subtract max for numerical stability
    shifted = input - max(input, dim).detach()
    # Compute exp and sum

    # Use LogSumExp trick: x - (max + log(sum(exp(x - max))))

    exp_values = shifted.exp()
    sum_exp = exp_values.sum(dim=dim)
    # Log of softmax: log(exp(x)/sum(exp)) = x - log(sum(exp))
    # the max is added back in implicitly by using shifted
    return shifted - sum_exp.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D max pooling over an input tensor.

    Args:
    ----
        input: input tensor of shape (batch, channel, height, width)
        kernel: tuple of (kernel_height, kernel_width)

    Returns:
    -------
        Tensor with max pooling applied

    """
    # Use tile function to reshape input for pooling
    tiled, new_height, new_width = tile(input, kernel)

    # Apply max over the pooling window (last dimension)
    out = max(tiled, 4)

    # Reshape to remove the extra dimension
    return out.view(out.shape[0], out.shape[1], out.shape[2], out.shape[3])


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Applies dropout to input tensor.

    Args:
    ----
        input: input tensor
        rate: dropout rate (probability of setting values to 0)
        ignore: if True, don't apply dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore or rate == 0.0:
        return input
    if rate == 1.0:
        return input * 0.0

    # Create random mask with probability (1 - rate) of 1s
    mask = rand(input.shape)
    mask = (mask > rate) / (1.0 - rate)
    return input * mask


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: input tensor
        dim: dimension to compute argmax over

    Returns:
    -------
        A tensor of the same size as input with 1's in the position of the maximum value
        along dim and 0's everywhere else.

    """
    # Create a tensor of zeros with same shape as input
    max_vals = max(input, dim)

    # Create a tensor that's True where input equals the max value (broadcasting)
    return input == max_vals
