import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow.keras.regularizers as reg


class ResNet8(tf.keras.Model):
    def __init__(self, out_dim, f=0.25):
        super(ResNet8, self).__init__()

        # constants
        STRIDE = 2
        KERNEL_1 = 1
        KERNEL_3 = 3
        KERNEL_5 = 5
        POOL_SIZE = 3
        L2_REGULIZER_VAL = 1e-4
        DROP_PROBABILITY = 0.5

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
        res2_1 = l.concatenate([self.conv4(res1_1), res1_2])
        res2_2 = self.conv6(self.activ4(self.conv5(self.activ3(res2_1))))
        res3_1 = l.concatenate([self.conv7(res2_1), res2_2])
        res3_2 = self.conv9(self.activ6(self.conv8(self.activ5(res3_1))))
        res = l.concatenate([self.conv10(res3_1), res3_2])
        return self.out_block(res)


class ResNet8b(tf.keras.Model):
    def __init__(self, out_dim, f=0.25):
        super(ResNet8b, self).__init__()

        # constants
        STRIDE = 2
        KERNEL_1 = 1
        KERNEL_3 = 3
        KERNEL_5 = 5
        POOL_SIZE = 3
        L2_REGULIZER_VAL = 1e-4
        DROP_PROBABILITY = 0.5


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

        self.out_block = tf.keras.Sequential([
                            l.Flatten(),
                            l.Activation('relu'),
                            l.Dropout(DROP_PROBABILITY),
                            l.Dense(int(32 * f)),
                            l.Activation('relu'),
                            l.Dense(out_dim)])

    def call(self, x):
        # Define the forward pass
        res1_1 = self.max_pool1(self.conv1(x))
        res1_2 = self.conv3(self.activ2(self.conv2(self.activ1(res1_1))))
        res2_1 = l.concatenate([self.conv4(res1_1), res1_2])
        res2_2 = self.conv6(self.activ4(self.conv5(self.activ3(res2_1))))
        res3_1 = l.concatenate([self.conv7(res2_1), res2_2])
        return self.out_block(res3_1)


class TCResNet8(tf.keras.Model):
    def __init__(self, out_dim, f=0.25):
        super(TCResNet8, self).__init__()

        # constants
        FIRST_CONV_KERNEL = [3, 1]
        ANYOTHER_CONV_KERNEL = [9, 1]
        SHORTCUT_CONV_KERNEL = [1, 1]
        L2_REGULIZER_VAL = 1e-4
        DROP_PROBABILITY = 0.5

        # begin
        self.conv1 = l.Conv2D(int(16 * f), FIRST_CONV_KERNEL, strides=1, padding='same',
                              use_bias=False)

        # for block1:
        self.conv2 = l.Conv2D(filters=int(24 * f), kernel_size=ANYOTHER_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ1_1 = l.Activation('relu')
        self.conv3 = l.Conv2D(int(24 * f), ANYOTHER_CONV_KERNEL, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv4 = l.Conv2D(int(24 * f), SHORTCUT_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_1_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ1_1_1 = l.Activation('relu')
        self.activ1_2 = l.Activation('relu')

        # for block2:
        self.conv5 = l.Conv2D(int(32 * f), ANYOTHER_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ2_1 = l.Activation('relu')
        self.conv6 = l.Conv2D(int(32 * f), ANYOTHER_CONV_KERNEL, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv7 = l.Conv2D(int(32 * f), SHORTCUT_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_1_1 = l.BatchNormalization()
        self.activ2_1_1 = l.Activation('relu')
        self.activ2_2 = l.Activation('relu')

        # for block3:
        self.conv8 = l.Conv2D(int(48 * f), ANYOTHER_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ3_1 = l.Activation('relu')
        self.conv9 = l.Conv2D(int(48 * f), ANYOTHER_CONV_KERNEL, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv10 = l.Conv2D(int(48 * f), SHORTCUT_CONV_KERNEL, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_1_1 = l.BatchNormalization()
        self.activ3_1_1 = l.Activation('relu')
        self.activ3_2 = l.Activation('relu')

        # avg pooling 2D
        self.avg_pool = l.AvgPool2D(strides=1)

        self.out_block = tf.keras.Sequential([
            l.Flatten(),
            l.Activation('relu'),
            l.Dropout(DROP_PROBABILITY),
            l.Dense(int(46 * f)),
            l.Activation('softmax'),
            l.Dense(out_dim)]
        )

    def call(self, x):
        # Define the forward pass
        # input: instead of wxhx1 use wx1xh
        # output: instead of 3x3x1xhxwx3 get 3x1xfxtx1xc
        L = x.shape[1]
        C = x.shape[2]
        # inputs = tf.reshape(x, [-1, L, 1, C])  # [N, L, 1, C]
        inputs = tf.reshape(x, [-1, L, 3, C])  # [N, L, 1, C]

        res0_1 = self.conv1(inputs)
        # through block1
        res1_1 = self.bn1_2(self.conv3(self.activ1_1(self.bn1_1(self.conv2(res0_1)))))
        res1_1_1 = self.activ1_1_1(self.bn1_1_1(self.conv4(res0_1)))
        res1_2 = l.concatenate([res1_1, res1_1_1])
        res1_3 = self.activ1_2(res1_2)
        # through block 2
        res2_1 = self.bn2_2(self.conv6(self.activ2_1(self.bn2_1(self.conv5(res1_3)))))
        res2_1_1 = self.activ2_1_1(self.bn2_1_1(self.conv7(res1_2)))
        res2_2 = l.concatenate([res2_1, res2_1_1])
        res2_3 = self.activ2_2(res2_2)
        # through block 3
        res3_1 = self.bn3_2(self.conv9(self.activ3_1(self.bn3_1(self.conv8(res2_3)))))
        res3_1_1 = self.activ3_1_1(self.bn3_1_1(self.conv10(res2_3)))
        res3_2 = l.concatenate([res3_1, res3_1_1])
        res3_3 = self.activ3_2(res3_2)
        self.avg_pool = l.AvgPool2D(pool_size=res3_3.shape[1:3], strides=1)  # Average Pooling
        res = self.avg_pool(res3_3)
        return self.out_block(res)
