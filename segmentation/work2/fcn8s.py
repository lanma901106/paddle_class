import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Conv2DTranspose
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from vgg import VGG16BN


class FCN8s(fluid.dygraph.Layer):
 # TODO: create fcn8s model




def main():
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 512, 512).astype(np.float32)
        x = to_variable(x_data)
        model = FCN8s(num_classes=59)
        model.eval()
        pred = model(x)
        print(pred.shape)


if __name__ == '__main__':
    main()
