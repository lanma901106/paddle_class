import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from backbone_models.resnet_multi_grid import ResNet50



class ASPPPooling(Layer):
    # TODO:


class ASPPConv(Layer):
    # TODO:



class ASPPModule(Layer):
    # TODO:


class DeepLabHead(fluid.dygraph.Sequential):
    def __init__(self, num_channels, num_classes):
        super(DeepLabHead, self).__init__(
                ASPPModule(num_channels, 256, [12, 24, 36]),
                Conv2D(256, 256, 3, padding=1),
                BatchNorm(256, act='relu'),
                Conv2D(256, num_classes, 1)
                )


class DeepLab(Layer):
    # TODO:






def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = DeepLab(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)



if __name__ == '__main__':
    main()
