import torch
from torch import nn

class ComplexConv2d(nn.Module):
    """
    A 3D convolutional operator that supports complex values.
    Based on implementation described by:
        EK Cole, et al. "Analysis of deep complex-valued CNNs for MRI reconstruction," arXiv:2004.01738.
    """
    def __init__(self, in_chans, out_chans, kernel_size):
        super(ComplexConv2d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        # The complex convolution operator has two sets of weights: X, Y
        self.conv_r = nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)
        self.conv_i = nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding, bias=False)

    def forward(self, input):
        # The input has real and imaginary parts: a, b
        # The output of the convolution (Z) can be written as:
        #   Z = (X + iY) * (a + ib)
        #     = (X*a - Y*b) + i(X*b + Y*a)

        # Compute real part of output
        output_real = self.conv_r(input.real)
        output_real = output_real - self.conv_i(input.imag)
        # Compute imaginary part of output
        output_imag = self.conv_r(input.imag)
        output_imag = output_imag + self.conv_i(input.real)

        return torch.complex(output_real, output_imag)
    

class Activation(nn.Module):
    """
    A generic class for activation layers.
    """
    def __init__(self, type='relu'):
        super(Activation, self).__init__()

        if type == 'none':
            self.activ = nn.Identity()
        elif type == 'relu':
            self.activ = nn.ReLU()
        elif type == 'leaky_relu':
            self.activ = nn.PReLU()
        else:
            raise ValueError('Invalid activation type: %s' % type)

    def forward(self, input):
        if input.is_complex():
            return torch.view_as_complex(self.activ(torch.view_as_real(input)))
        else:
            return self.activ(input)
        
        
class ConvBlock(nn.Module):
    """
    A 3D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv
    Based on implementation described by:
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """
    def __init__(self, in_chans, out_chans, kernel_size, act_type='relu', norm_type='none', is_complex=True):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.is_complex = is_complex
        self.name = 'ComplexConv2D' if is_complex else 'Conv2D'

        # Define normalization and activation layers
        activation = Activation(act_type)

        if is_complex:
            convolution = ComplexConv2d(in_chans, out_chans, kernel_size=kernel_size)
        else:
            convolution = Conv2d(in_chans, out_chans, kernel_size=kernel_size)

        # Define forward pass (pre-activation)
        self.layers = nn.Sequential(
            convolution,
            activation
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input)

    def __repr__(self):
        return f'{self.name}(in_chans={self.in_chans}, out_chans={self.out_chans})'