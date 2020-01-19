import sys
import time
import numpy as np
import tensorflow as tf

from LearningHierarchy.LearningPipeline.Nets import ResNet8
from LearningHierarchy.LearningPipeline.DataUtilities import *

import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp

from datetime import datetime


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

    def save(self):
        model_name = 'model_test'
        print(" [*] Saving checkpoint to %s..." % self.config.checkpoint_dir)
        path = os.path.join(self.config.checkpoint_dir, model_name + '_epoch_{:}'
                            .format(self.config.max_epochs))
        if not os.path.exists(path):
            os.mkdir(path)
        tf.saved_model.save(self.mdl, path)
        self.mdl.save(path)  # does the exact same thing

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

        # self.train_steps_per_epoch = int(tf.math.ceil(n_samples_train / self.config.batch_size))

        self.mdl = ResNet8(out_dim=self.config.output_dim, f=self.config.f)

        if self.config.resume_train:
            print("Resume training from previous checkpoint")

            # # load .pb file
            # latest_pb_file = max([os.path.join(dp, f) for dp, dn, filenames in os.walk(self.config.directory_pb_file) for f in
            #           filenames if '.pb' in f], key=os.path.getmtime)
            # self.latest_pb_directory = os.path.dirname(latest_pb_file)
            # self.mdl_temp = tf.saved_model.load(self.latest_pb_directory)  # this includes weights and everything!
            # loss, accuracy, mse = self.mdl.evaluate(val_data)
            # print("Restored model, accuracy: {}%".format(100*acc))

            # load .pb file
            latest_ckpt_file = max([os.path.join(dp, f) for dp, dn, filenames in
                                    os.walk(os.path.join(self.config.checkpoint_dir, 'TrainingLog')) for f in
                                    filenames if '.ckpt' in f], key=os.path.getmtime)
            self.latest_ckpt_directory = os.path.dirname(latest_ckpt_file)
            self.mdl.load_weights(os.path.join(self.latest_ckpt_directory, '.ckpt'))

        self.mdl.compile(optimizer=custom_optimizer,
                        loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # set callbacks
        # saving the model callback
        path_save_callback_temp = os.path.join(self.config.checkpoint_dir, 'TrainingLog')
        if not os.path.exists(path_save_callback_temp):
            os.mkdir(path_save_callback_temp)
        path_save_callback = os.path.join(self.config.checkpoint_dir, 'TrainingLog', datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(path_save_callback):
            os.mkdir(path_save_callback)
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_save_callback, '.ckpt'),
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         # save_freq=self.config.save_latest_freq  # saves every few steps...
                                                         period=self.config.save_latest_period)  # saves every few epochs
        # tensorboard summary visualization callback
        path_summary_callback_temp = os.path.join(self.config.checkpoint_dir, 'TrainingLog', 'Logs')
        if not os.path.exists(path_summary_callback_temp):
            os.mkdir(path_summary_callback_temp)
        path_summary_callback = os.path.join(self.config.checkpoint_dir, 'TrainingLog', 'Logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(path_summary_callback):
            os.mkdir(path_summary_callback)
        summary_callback = tf.keras.callbacks.TensorBoard(log_dir=path_summary_callback, histogram_freq=1,
                                                          write_images=True,
                                                          # update_freq=self.config.summary_freq_epoch)
                                                            update_freq='epoch')
        # # custum tensorboard summary visualization cal
        # metrics_path = os.path.join(path_summary_callback, 'metrics')
        # if not os.path.exists(metrics_path):
        #     os.mkdir(metrics_path)
        # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
        # file_writer = tf.summary.create_file_writer(path_summary_callback + "\\metrics")
        # with file_writer.as_default():
        #     hp.hparams_config(
        #         hparams=[HP_OPTIMIZER],
        #         metrics=[hp.Metric('accuracy', display_name='Accuracy')]  # , hp.Metric('mse', display_name='MSE')]
        #     )

        # hprm_callback = hp.KerasCallback(metrics_path, hp.hparams)

        # file_writer.set_as_default()
        # learning_rate_temp = self.config.learning_rate

        # def lr_schedule(epoch):
        #     if epoch > 2:
        #         learning_rate = learning_rate_temp*0.75
            # if epoch > 10:
            #     learning_rate = learning_rate_temp*0.5
            # if epoch > 20:
            #     learning_rate = learning_rate_temp*0.25
            # if epoch > 50:
            #     learning_rate = learning_rate_temp*0.1

            # tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
            # return learning_rate
        #
        # lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        # self.history = self.mdl.fit(train_data, epochs=self.config.max_epochs, validation_data=val_data,
        #                             callbacks=[save_callback, summary_callback, lr_callback])
        self.history = self.mdl.fit(train_data, epochs=self.config.max_epochs, validation_data=val_data,
                                    callbacks=[save_callback, summary_callback])

        self.mdl.summary()

        self.save()

        print('-----------------finished to train-----------------')

    def test(self):

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

        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, beta_1=self.config.beta1)

        # get test dataset
        test_data, n_samples_test = self.dataLoading(validation=True)

        self.mdl_test = ResNet8(out_dim=self.config.output_dim, f=self.config.f)

        # load ckpt file
        latest_ckpt_file = max([os.path.join(dp, f) for dp, dn, filenames in
                                os.walk(os.path.join(self.config.checkpoint_dir, 'TrainingLog')) for f in
                                filenames if '.ckpt' in f], key=os.path.getmtime)
        self.latest_ckpt_directory = os.path.dirname(latest_ckpt_file)
        self.mdl_test.load_weights(os.path.join(self.latest_ckpt_directory, '.ckpt'))

        self.mdl_test.compile(optimizer=custom_optimizer,
                         loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # evaluate
        results = self.mdl_test.evaluate(test_data)
        print('test loss, test acc, test mse:', results)

        print('-----------------finished to evaluate-----------------')