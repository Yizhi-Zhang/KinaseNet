from torch.nn import *
import torch as torch
from torch.nn import ReLU
from torch import Tensor as T


class ReLUgradat0(torch.autograd.Function):
    """
    source: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


class ReLU0(ReLU):

    def forward(self, input: T) -> T:
        return ReLUgradat0.apply(input)
