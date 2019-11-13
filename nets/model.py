from torch.nn import *
from torch import add, cat
import numpy as np
import torch

# This module is from the article "An RDAU-NET model for lesion segmentation in breast ultrasound images",
# PLOS ONE https://doi.org/10.1371/journal.pone.0221535


class ResUnit(Module):
    """
    To build a resdual block with two conditions: encode and decode
    it can be expressed in: y = F(x, {Wi}) + Ws * x, where x and y are the input and output vector and
    Wi is the weight of the corresponding layer. Ws is the weight of the correction term.
    """
    def __init__(self, in_channels: int, out_channels: int, if_encode: bool):
        super(ResUnit, self).__init__()
        self.if_encode = if_encode
        self.conv_encode_1 = Sequential(
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        )
        self.conv_encode_2 = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=2, padding=(0, 0)),
            BatchNorm2d(out_channels)
        )
        self.conv_decode_1 = Sequential(
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
            BatchNorm2d(out_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1)),
        )
        self.conv_decode_2 = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.if_encode:
            conv1 = self.conv_encode_1(x)
            conv2 = self.conv_encode_2(x)
            output = add(conv1, conv2)
            return output

        elif self.if_encode is False:
            conv1 = self.conv_decode_1(x)
            conv2 = self.conv_decode_2(x)
            output = add(conv1, conv2)
            return output


class AttentionGate(Module):
    """
    To build an attention gate block which can be difined as: h_output = alpha * h, where h_output is the output of
    the gate; h is one of the input of the gate; alpha is the computed attention coefficients, which can be defined as:
    alpha = sigma2 * { Wk * [ Wint * ( sigma1 * (Wh * h + Wg * g + b_h,g)) + b_int] + b_k}, where g and h represent the
    feature maps presented to the inputs of AG modules from the decoder and encoder piplines and Wg, Wh, Wint, Wk
    indicate the convolution kernels. Furthermore, sigma1 is the Relu activation function and sigma2 is the Sigmoid
    activation function.
    """
    def __init__(self, in_channels_g, in_channels_h):
        super(AttentionGate, self).__init__()
        self.W_g = Conv2d(in_channels_g, in_channels_g, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.W_h = Conv2d(in_channels_h, in_channels_g, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.alpha = Sequential(
            Conv2d(in_channels_g, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            Conv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        self.relu = ReLU(inplace=True)

    def forward(self, g, h):
        g1 = self.W_g(g)
        h1 = self.W_h(h)
        alpha = self.relu(g1 + h1)
        alpha = self.alpha(alpha)
        alpha = torch.sigmoid(alpha)
        return h * alpha


class DilationBlock(Module):
    """
    To build a 6-layer residual block which includes 6 dilation convolution layers with the dilation ratio r=1,2,4,8,
    16,32. The output is the sum of these layers.
    """
    def __init__(self, in_channels, out_channels):
        super(DilationBlock, self).__init__()
        self.conv1 = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=1, padding=(1, 1)),
            ReLU())
        self.conv2 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=2, padding=(2, 2)),
            ReLU())
        self.conv3 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=4, padding=(4, 4)),
            ReLU())
        self.conv4 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=8, padding=(8, 8)),
            ReLU())
        self.conv5 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=16, padding=(16, 16)),
            ReLU())
        self.conv6 = Sequential(
            Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=32, padding=(32, 32)),
            ReLU())

    def forward(self, x):
        conv1_output = self.conv1(x)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        conv6_output = self.conv6(conv5_output)
        output = conv1_output + conv2_output + conv3_output + conv4_output + conv5_output + conv6_output
        return output


class UpConv(Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.upconv = Sequential(
            UpsamplingNearest2d(scale_factor=2),
            ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upconv(x)


class UNet2(Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.resblock1 = ResUnit(1, 32, if_encode=False)
        self.resblock2 = ResUnit(32, 64, if_encode=True)
        self.resblock3 = ResUnit(64, 128, if_encode=True)
        self.resblock4 = ResUnit(128, 256, if_encode=True)
        self.resblock5 = ResUnit(256, 512, if_encode=True)
        self.resblock6 = ResUnit(512, 512, if_encode=True)

        self.resblock7 = ResUnit(768, 512, if_encode=False)
        self.resblock8 = ResUnit(768, 256, if_encode=False)
        self.resblock9 = ResUnit(384, 128, if_encode=False)
        self.resblock10 = ResUnit(192, 64, if_encode=False)
        self.resblock11 = ResUnit(96, 32, if_encode=False)

        self.ag5 = AttentionGate(64, 32)
        self.ag4 = AttentionGate(128, 64)
        self.ag3 = AttentionGate(256, 128)
        self.ag2 = AttentionGate(512, 256)
        self.ag1 = AttentionGate(256, 512)

        self.diconv = DilationBlock(512, 256)

        self.upconv = UpConv()

        self.conv = Conv2d(32, 1, (1, 1), (1, 1), (0, 0))

    def forward(self, x):
        encode1 = self.resblock1(x)
        encode2 = self.resblock2(encode1)
        encode3 = self.resblock3(encode2)
        encode4 = self.resblock4(encode3)
        encode5 = self.resblock5(encode4)
        encode6 = self.resblock6(encode5)
        # dilation conv path
        diconv = self.diconv(encode6)

        # up conv, attention gate, concatenate and decode path
        up_conv1 = self.upconv(diconv)
        ag1 = self.ag1(up_conv1, encode5)

        cat1 = cat((ag1, up_conv1), dim=1)
        decode1 = self.resblock7(cat1)

        up_conv2 = self.upconv(decode1)
        ag2 = self.ag2(up_conv2, encode4)

        cat2 = cat((ag2, up_conv2), dim=1)
        decode2 = self.resblock8(cat2)

        up_conv3 = self.upconv(decode2)
        ag3 = self.ag3(up_conv3, encode3)

        cat3 = cat((ag3, up_conv3), dim=1)
        decode3 = self.resblock9(cat3)

        up_conv4 = self.upconv(decode3)
        ag4 = self.ag4(up_conv4, encode2)

        cat4 = cat((ag4, up_conv4), dim=1)
        decode4 = self.resblock10(cat4)

        up_conv5 = self.upconv(decode4)
        ag5 = self.ag5(up_conv5, encode1)

        cat5 = cat((ag5, up_conv5), dim=1)
        decode5 = self.resblock11(cat5)

        output = self.conv(decode5)
        output = torch.sigmoid(output)
        return output

