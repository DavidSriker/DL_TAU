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
        # x.shape = TensorShape([None, 300, 200, 3])
        res1_1 = self.max_pool1(self.conv1(x))  # after conv1: TensorShape([None, 150, 100, 32]); after max_pool1: TensorShape([None, 74, 49, 32])
        res1_2 = self.conv3(self.activ2(self.conv2(self.activ1(res1_1))))  # after conv2: TensorShape([None, 37, 25, 32]); after conv3: TensorShape([None, 37, 25, 32])

        res2_1 = l.concatenate([self.conv4(res1_1), res1_2])  # after conv4: TensorShape([None, 37, 25, 32]); after concatenate: TensorShape([None, 37, 25, 64])
        res2_2 = self.conv6(self.activ4(self.conv5(self.activ3(res2_1))))  # after conv5: TensorShape([None, 19, 13, 64]); after conv6: TensorShape([None, 19, 13, 64])

        res3_1 = l.concatenate([self.conv7(res2_1), res2_2])  # after conv7: TensorShape([None, 19, 13, 64]); after concatenate: TensorShape([None, 19, 13, 128])
        res3_2 = self.conv9(self.activ6(self.conv8(self.activ5(res3_1))))  # after conv8: TensorShape([None, 10, 7, 128]); after conv9: TensorShape([None, 10, 7, 128])

        res = l.concatenate([self.conv10(res3_1), res3_2])  # after conv10: TensorShape([None, 10, 7, 128]); after concatenate: TensorShape([None, 10, 7, 256])
        return self.out_block(res)  # after outblock: TensorShape([None, 3]) where None is the size of the batch


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

        # # activation
        # self.activ5 = l.Activation('relu')
        # # convolution 2D
        # self.conv8 = l.Conv2D(int(128 * f), KERNEL_3, strides=STRIDE,
        #                         padding='same', kernel_initializer="he_normal",
        #                         kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # # activation
        # self.activ6 = l.Activation('relu')
        # # convolution 2D
        # self.conv9 = l.Conv2D(int(128 * f), KERNEL_3,
        #                         padding='same', kernel_initializer="he_normal",
        #                         kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        # # convolution 2D
        # self.conv10 = l.Conv2D(int(128 * f), KERNEL_1,
        #                         strides=STRIDE,
        #                         padding='same')

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

        # res2_1 = m.add([self.conv4(res_1_1), res_1_2])
        # res2_1 = m.add([self.conv4(res1_1), res1_2])
        res2_1 = l.concatenate([self.conv4(res1_1), res1_2])
        res2_2 = self.conv6(self.activ4(self.conv5(self.activ3(res2_1))))

        # res3_1 = m.add([self.conv7(res2_1), res2_2])
        res3_1 = l.concatenate([self.conv7(res2_1), res2_2])
        # res3_2 = self.conv9(self.activ6(self.conv8(self.activ5(res2_1))))
        #
        # # res = m.add([self.conv10(res3_1), res3_2])
        # res = l.concatenate([self.conv10(res2_1), res3_2])
        return self.out_block(res3_1)


class TCResNet8(tf.keras.Model):
    def __init__(self, out_dim, f=0.25):
        super(TCResNet8, self).__init__()

        # constants
        first_conv_kernel = [3, 1]
        anyother_conv_kernel = [9, 1]
        shortcut_conv_kernel = [1, 1]
        POOL_SIZE = 3
        L2_REGULIZER_VAL = 1e-4
        DROP_PROBABILITY = 0.5
        # SGD(momentum 0.9 mini-batch 100 samples, 30k iterations)
        # init_LR = 0.1 and divided by 10 every 10k epochs
        # xavier weight initialization
        # batch norm include trainable for scale and shift.... need to set values?


        # begin
        self.conv1 = l.Conv2D(int(16 * f), first_conv_kernel, strides=1, padding='same',
                              use_bias=False)

        # for block1:
        self.conv2 = l.Conv2D(filters=int(24 * f), kernel_size=anyother_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ1_1 = l.Activation('relu')
        self.conv3 = l.Conv2D(int(24 * f), anyother_conv_kernel, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv4 = l.Conv2D(int(24 * f), shortcut_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn1_1_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ1_1_1 = l.Activation('relu')
        self.activ1_2 = l.Activation('relu')

        # for block2:
        self.conv5 = l.Conv2D(int(32 * f), anyother_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ2_1 = l.Activation('relu')
        self.conv6 = l.Conv2D(int(32 * f), anyother_conv_kernel, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv7 = l.Conv2D(int(32 * f), shortcut_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn2_1_1 = l.BatchNormalization()
        self.activ2_1_1 = l.Activation('relu')
        self.activ2_2 = l.Activation('relu')

        # for block3:
        self.conv8 = l.Conv2D(int(48 * f), anyother_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_1 = l.BatchNormalization(scale=True, trainable=True)
        self.activ3_1 = l.Activation('relu')
        self.conv9 = l.Conv2D(int(48 * f), anyother_conv_kernel, strides=1, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_2 = l.BatchNormalization(scale=True, trainable=True)
        self.conv10 = l.Conv2D(int(48 * f), shortcut_conv_kernel, strides=2, padding='same',
                              use_bias=False, kernel_initializer="he_normal",
                              kernel_regularizer=reg.l2(L2_REGULIZER_VAL))
        self.bn3_1_1 = l.BatchNormalization()
        self.activ3_1_1 = l.Activation('relu')
        self.activ3_2 = l.Activation('relu')

        # logits = slim.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None, scope="fc")
        # logits = tf.reshape(logits, shape=(-1, logits.shape[3]), name="squeeze_logit")
        # ranges = slim.conv2d(net, 2, 1, activation_fn=None, normalizer_fn=None, scope="fc2")
        # ranges = tf.reshape(ranges, shape=(-1, ranges.shape[3]), name="squeeze_logit2")
        # endpoints["ranges"] = tf.sigmoid(ranges)

        # avg pooling 2D
        self.avg_pool = l.AvgPool2D(strides=1)

        self.out_block = tf.keras.Sequential([
            l.Flatten(),  # from 3,1,1,96 we get 3,96
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

        # Average Pooling
        # self.avg_pool.pool_size = res3_3.shape[1:3]
        # temp = tf.shape(res3_3).numpy()
        # self.avg_pool.pool_size = temp[1:3]
        # print(tf.shape(res3_3))
        # print(res3_3.shape)
        # self.avg_pool = l.AvgPool2D(pool_size=tf.shape(res3_3).numpy()[1:3], strides=1)
        self.avg_pool = l.AvgPool2D(pool_size=res3_3.shape[1:3], strides=1)
        res = self.avg_pool(res3_3)
        # reshape...?
        return self.out_block(res)  # this is a 3x3. which is actually out_dimXnumber of channels (rgb = 3 channels)
