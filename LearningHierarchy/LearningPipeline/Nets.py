import tensorflow as tf
from tensorflow.keras import layers as l
from tensorflow.keras import regularizers as reg
from tensorflow.keras import merge as m


class ResNet8(tf.keras.Model):
    # constants
    STRIDE = 2
    KERNEL_1 = 1
    KERNEL_3 = 3
    KERNEL_5 = 5
    POOL_SIZE = 3
    L2_REGULIZER_VAL = 1e-4
    DROP_PROBABILITY = 0.5

    def __init__(self, out_dim, f=0.25):
        super(ResNet8, self).__init__()
        # convolution 2D
        self.conv1 = l.Conv2D(int(32 * f), KERNEL_5,
                                strides=STRIDE,
                                padding='same')
        # max pooling 2D
        self.max_pool1 = l.MaxPool2D(pool_size=POOL_SIZE,
                                        strides=STRIDE)

        # activation
        self.activ1 = l.Activation('relu')
        # convolution 2D
        self.conv2 = l.Conv2D(int(32 * f), KERNEL_3, strides=STRIDE,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # activation
        self.activ2 = l.Activation('relu')
        # convolution 2D
        self.conv3 = l.Conv2D(int(32 * f), KERNEL_3,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # convolution 2D
        self.conv4 = l.Conv2D(int(32 * f), KERNEL_1,
                                strides=STRIDE,
                                padding='same')

        # activation
        self.activ3 = l.Activation('relu')
        # convolution 2D
        self.conv5 = l.Conv2D(int(64 * f), KERNEL_3, strides=STRIDE,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # activation
        self.activ4 = l.Activation('relu')
        # convolution 2D
        self.conv6 = l.Conv2D(int(64 * f), KERNEL_3,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # convolution 2D
        self.conv7 = l.Conv2D(int(64 * f), KERNEL_1,
                                strides=STRIDE,
                                padding='same')

        # activation
        self.activ5 = l.Activation('relu')
        # convolution 2D
        self.conv8 = l.Conv2D(int(128 * f), KERNEL_3, strides=STRIDE,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # activation
        self.activ6 = l.Activation('relu')
        # convolution 2D
        self.conv9 = l.Conv2D(int(128 * f), KERNEL_3,
                                padding='same', kernel_initializer="he_normal",
                                kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # convolution 2D
        self.conv10 = l.Conv2D(int(128 * f), KERNEL_1,
                                strides=STRIDE,
                                padding='same')

        self.out_block = tf.keras.Sequential([
                            l.Flatten(),
                            l.Activation('relu'),
                            l.Dropout(DROP_PROBABILITY),
                            l.Dense(int(256 * f)),
                            l.Activation('relu'),
                            l.Dense(out_dim)])

    def call(self, x):
        # Define the forward pass
        res1_1 = self.max_pool1(self.conv1(x))
        res1_2 = self.conv3(self.activ2(self.conv2(self.activ1(res1_1))))

        res2_1 = m.add([self.conv4(res_1_1), res_1_2])
        res2_2 = self.conv6(self.activ4(self.conv5(self.activ3(res2_1))))

        res3_1 = m.add([self.conv7(res2_1), res2_2])
        res3_2 = self.conv9(self.activ6(self.conv8(self.activ5(res3_1))))

        res = m.add([self.conv10(res3_1), res3_2])
        return self.out_block(res)
