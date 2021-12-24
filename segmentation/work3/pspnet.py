import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50


# pool with different bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):


class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()

        # stem: res.conv, res.pool2d_max

        # psp: 2048 -> 2048*2

        # cls: 2048*2 -> 512 -> num_classes

        # aux: 1024 -> 256 -> num_classes

    def forward(self, inputs):


# aux: tmp_x = layer3


def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data = np.random.rand(2, 3, 473, 473).astype(np.float32)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        model.train()
        pred, aux = model(x)
        print(pred.shape, aux.shape)


if __name__ == "__main__":
    main()
