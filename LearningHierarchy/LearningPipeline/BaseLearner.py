import sys
import time
import numpy as np
import tensorflow as tf

from LearningHierarchy.LearningPipeline.Nets import ResNet8
from LearningHierarchy.LearningPipeline.DataUtilities import *

import matplotlib.pyplot as plt

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


class TrajectoryLearner(object):
    def __init__(self, config):
        self.config = config
        return

    def dataLoading(self, validation=False):
        if not validation:
            iter = ImagesIterator(directory=self.config.train_dir)
        else:
            iter = ImagesIterator(directory=self.config.val_dir)
        data_iter = iter.generateBatches()
        return data_iter, iter.num_samples

    def loss(self, y_true, y_pred):
        coordinate_loss = tf.keras.losses.MSE(y_true=y_true[:, :2], y_pred=y_pred[:, :2])
        velocity_loss = tf.keras.losses.MSE(y_true=y_true[:, 2], y_pred=y_pred[:, 2])
        loss = coordinate_loss + self.config.gamma * velocity_loss
        return loss

    # def grad(self, image_batch, labels_batch, validation=False):
    #     with tf.GradientTape() as tape:
    #         pred = self.mdl.call(x=image_batch)
    #         loss, coordinate_loss, velocity_loss = self.loss(pred, labels_batch)
    #     if not validation:
    #         train_vars = self.mdl.trainable_variables
    #         train_grads = tape.gradient(loss, train_vars)
    #         grads_and_vars = zip(train_grads, train_vars)
    #
    #     point_loss = np.mean(coordinate_loss)
    #     vel_loss = np.mean(velocity_loss)
    #     train_loss = np.mean(loss)
    #
    #     if not validation:
    #         return pred, train_loss, point_loss, vel_loss, train_grads, train_vars, grads_and_vars
    #     else:
    #         return pred, train_loss, point_loss, vel_loss

    def save(self):
        model_name = 'model_test'
        print(" [*] Saving checkpoint to %s..." % self.config.checkpoint_dir)
        path = os.path.join(self.config.checkpoint_dir, model_name + '_epoch_{:}_'
                            .format(self.config.max_epochs))
        if not os.path.exists(path):
            os.mkdir(path)
        tf.saved_model.save(self.mdl, path)

    # def train(self, train_data, n_samples_train):
    def train(self):
        gpu_config = tf.config.experimental.list_physical_devices('GPU')
        if gpu_config:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpu_config:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    gpus = tf.config.experimetal.list_physical_devices('GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        train_data, n_samples_train = self.dataLoading()
        val_data, n_samples_val = self.dataLoading(validation=True)

        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, beta_1=self.config.beta1)

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # self.train_summary_writer = tf.summary.create_file_writer(self.config.checkpoint_dir)

        self.train_steps_per_epoch = int(tf.math.ceil(n_samples_train / self.config.batch_size))

        self.mdl = ResNet8(out_dim=self.config.output_dim, f=self.config.f)


        # if self.config.resume_train:
        #     print("Resume training from previous checkpoint")
        #     latest_pb_file = max(glob.glob(os.path.join(self.config.directory_pb_file, '*')), key=os.path.getmtime)
        #     self.config.pb_file("pb_file", os.path.join(latest_pb_file, "saved_model.pb"),
        #                          "Checkpoint file")
        #     self.mdl.load_weights(self.config.pb_file)


        # TODO: check if to put this compile in the self.config.resume_train 'else:' condition
        self.mdl.compile(optimizer=custom_optimizer,
                        loss=self.loss,
                         metrics=['accuracy', 'mse'])

        path_callback = os.path.join(self.config.checkpoint_dir, 'training\\')
        if not os.path.exists(path_callback):
            os.mkdir(path_callback)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_callback,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         save_freq=self.config.save_latest_period)  # add save_best option?

        self.history = self.mdl.fit(train_data, epochs=self.config.max_epochs, validation_data=val_data,
                                    callbacks=[cp_callback])


        # self.mdl.summary()

        self.save()

        print(1)

    def test(self):
        pass