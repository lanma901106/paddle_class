import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2DTranspose


class Encoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Encoder, self).__init__()
        # TODO: encoder contains:
        #       1 3x3conv + 1bn + relu +
        #       1 3x3conc + 1bn + relu +
        #       1 2x2 pool
        # return features before and after pool

    def forward(self, inputs):
        # TODO: finish inference part

        return x, x_pooled


class Decoder(Layer):
    def __init__(self, num_channels, num_filters):
        super(Decoder, self).__init__()
        # TODO: decoder contains:
        #       1 2x2 transpose conv (makes feature map 2x larger)
        #       1 3x3 conv + 1bn + 1relu +
        #       1 3x3 conv + 1bn + 1relu

    def forward(self, inputs_prev, inputs):
        # TODO: forward contains an Pad2d and Concat

        # Pad

        return x


class UNet(Layer):
    def __init__(self, num_classes=59):
        super(UNet, self).__init__()
        # encoder: 3->64->128->256->512
        # mid: 512->1024->1024

        # TODO: 4 encoders, 4 decoders, and mid layers contains 2 1x1conv+bn+relu

    def forward(self, inputs):
        x1, x = self.down1(inputs)
        print(x1.shape, x.shape)
        x2, x = self.down2(x)
        print(x2.shape, x.shape)
        x3, x = self.down3(x)
        print(x3.shape, x.shape)
        x4, x = self.down4(x)
        print(x4.shape, x.shape)

        # middle layers
        x = self.mid_conv1(x)
        x = self.mid_bn1(x)
        x = self.mid_conv2(x)
        x = self.mid_bn2(x)

        print(x4.shape, x.shape)
        x = self.up4(x4, x)
        print(x3.shape, x.shape)
        x = self.up3(x3, x)
        print(x2.shape, x.shape)
        x = self.up2(x2, x)
        print(x1.shape, x.shape)
        x = self.up1(x1, x)
        print(x.shape)

        x = self.last_conv(x)

        return x


def main():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        model = UNet(num_classes=59)
        x_data = np.random.rand(1, 3, 123, 123).astype(np.float32)
        inputs = to_variable(x_data)
        pred = model(inputs)

        print(pred.shape)


if __name__ == "__main__":
    main()
