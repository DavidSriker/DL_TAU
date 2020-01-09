import os
import time
from itertools import count
import random
import tensorflow as tf
import numpy as np
from keras.utils.generic_utils import Progbar
from .nets import resnet8 as prediction_network
from DataUtilities import *

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

TEST_PHASE = 0
TRAIN_PHASE = 1

class TrajectoryLearner(object):
    def __init__(self):
        pass

    def trainInit(self):
        is_training_ph = tf.TensorSpec([], tf.bool, name="is_training")
        with tf.name_scope("data_loading"):
            # generate training and validation batches ( we do not need labels)
            train_batch, n_samples_train = self.generateBatches(self.config.train_dir)
            val_batch, n_samples_test = self.generateBatches(self.config.val_dir, validation=True)
            current_batch = tf.cond(pred=is_training_ph, true_fn=lambda: train_batch, false_fn=lambda: val_batch)  # Return true_fn() if the predicate pred is true else false_fn().
            image_batch, pnt_batch = current_batch[0], current_batch[1]

        with tf.GradientTape() as tape:
            with tf.name_scope("trajectory_prediction"):
                pred_pnt = prediction_network(image_batch, output_dim=self.config.output_dim, f=self.config.f)

            with tf.name_scope("compute_loss"):
                point_loss = tf.keras.losses.MSE(labels=pnt_batch[:,:2], predictions=pred_pnt[:,:2])
                vel_loss = tf.keras.losses.MSE(labels=pnt_batch[:, 2], predictions=pred_pnt[:,2])

                train_loss = point_loss + 0.1 * vel_loss

        with tf.name_scope("metrics"):
            _, var = tf.nn.moments(x=pred_pnt, axes=-1)
            std = tf.math.sqrt(var)
            point_rmse = tf.math.sqrt(point_loss)
            vel_rmse = tf.math.sqrt(vel_loss)

        with tf.name_scope("train_op"):
            train_vars = prediction_network.trainable_variables
            optimizer  = tf.keras.optimizers.AdamOptimizer(self.config.learning_rate, self.config.beta1)
            train_grads = tape.gradient(train_loss, train_vars)
            self.grads_and_vars = zip(train_grads, train_vars)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary), maybe add images
        self.train_steps_per_epoch = int(tf.math.ceil(n_samples_train/self.config.batch_size))
        self.val_steps_per_epoch = int(tf.math.ceil(n_samples_test/self.config.batch_size))
        self.pred_pnt = pred_pnt
        self.gt_pnt = pnt_batch
        self.point_rmse = point_rmse
        self.vel_rmse = vel_rmse
        self.pred_stds = std
        self.image_batch = image_batch
        self.is_training = is_training_ph
        self.total_loss = train_loss
        self.val_loss_eval = point_loss
        self.optimizer = optimizer

    def collectSummaries(self, step, summ_freq, n_epochs):
        # Unlike TF 1.x, the summaries are emitted directly to the writer;
        # there is no separate "merge" op and no separate add_summary() call,
        # which means that the step value must be provided at the callsite
        with tf.summary.record_if(step % summ_freq == 0):
            pnt_error_sum = tf.summary.scalar("point_rmse", self.point_rmse, step=self.global_step)
            vel_error_sum = tf.summary.scalar("vel_rmse", self.vel_rmse, step=self.global_step)
            image_sum = tf.summary.image("image", self.image_batch, step=self.global_step)
        self.validation_error = tf.TensorSpec(shape=[], dtype=tf.float32)
        # self.val_error_log = tf.summary.scalar("Validation_Error", self.validation_error, step=self.global_step)
        with tf.summary.record_if(step % self.train_steps_per_epoch == 0):
            self.val_error_log = tf.summary.scalar("Validation_Error", self.validation_error, step=self.n_epochs)
        # train_summary_writer.flush()

    def save(self, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            # self.saved_model.save(os.path.join(checkpoint_dir, model_name + '.latest'))
            self.ckpt.save(os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            # self.saved_model.save(os.path.join(checkpoint_dir, model_name), global_step=step)
            self.ckpt.save(os.path.join(checkpoint_dir, model_name), global_step=step)

    def train(self, config):
        """
        High level train function.
        Args:
            self
            config: Configuration dictionary
        Returns:
            None
        """

        self.config = config
        coord = tf.train.Coordinator()
        self.trainInit()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) \
                                       for v in tf.trainable_variables])
        gpu_config = tf.config.experimental.list_physical_devices('GPU')
        # device_name = tf.test.gpu_device_name()
        # print(tf.test.is_gpu_available())
        # print('Found GPU at: {}'.format(device_name))
        if gpu_config:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpu_config:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        self.ckpt = tf.train.Checkpoint(step=self.global_step, optimizer=self.optimizer, model=prediction_network)
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.config.checkpoint_dir, max_to_keep=100)
        self.train_summary_writer = tf.summary.create_file_writer(self.config.checkpoint_dir)


        # with self.train_summary_writer.as_default():
        # LK - the following two lines can be neglected if doing errors
        # with self.train_summary_writer.managed_session(config=gpu_config).as_default:
        with train_summary_writer.as_default():
            print("Number of params: {}".format(parameter_count))
            if self.config.resume_train:
                print("Resume training from previous checkpoint")
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

            progbar = Progbar(target=self.train_steps_per_epoch)

            n_epochs = 0

            @tf.function
            def fetchesResults(self, feed_dict):
                if feed_dict:
                    start_time = time.time()
                    fetches = {"train": self.train_op, "global_step": self.global_step,
                               "incr_global_step": self.incr_global_step}
                    if step % config.summary_freq == 0:
                        fetches["vel_rmse"] = self.vel_rmse
                        fetches["pnt_rmse"] = self.point_rmse
                        fetches["stds"] = self.pred_stds
                        # fetches["summary"] = self.step_sum_op

            @tf.function
            def fetchesResultsValLoss(self, feed_dict):
                if not feed_dict:
                    # start_time_val = time.time()
                    fetches = {"loss_val": self.val_loss_eval, "global_step_val": self.global_step,
                               "incr_global_step_val": self.incr_global_step}

            # @tf_function
            # def fetchesResultsValError_log(self, feed_dict):
            #     if self.validation_error == val_error:
            #         # start_time_val = time.time()
            #         self.val_error_log = val_error  # or log()? or append to list of data-log?
            #         fetches = {"loss_val_error_log": self.val_error_log, "global_step_val": self.global_step,
            #                    "incr_global_step_val": self.incr_global_step}


            for step in count(start=1):
                if coord.should_stop():
                    break

                self.collectSummaries(step, config.summary_freq, n_epochs)

                # Runs a series of operations
                # results = sess.run(fetches, feed_dict={self.is_training: True})
                # Every v1.Session.run call should be replaced by a Python function.
                    # The feed_dict and v1.placeholders become function arguments.
                    # The fetches become the function's return value.
                # During conversion eager execution allows easy debugging with standard Python tools like pdb.
                results = fetchesResults(feed_dict={self.is_training: True})

                progbar.update(step % self.train_steps_per_epoch)

                gs = results["global_step"]

                if step % config.summary_freq == 0:
                    # sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = tf.math.ceil( gs /self.train_steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.train_steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it point_rmse: %.3f " \
                          "vel_rmse: %.6f, point_std: %.6f, vel_std: %.6f"
                       % (train_epoch, train_step, self.train_steps_per_epoch, \
                          time.time() - start_time, results["pnt_rmse"],
                          results["vel_rmse"],
                          np.mean(results["stds"][:2]),
                          results["stds"][2] ))

                if step % self.train_steps_per_epoch == 0:
                    n_epochs += 1
                    actual_epoch = int(gs / self.train_steps_per_epoch)

                    # self.save(sess, config.checkpoint_dir, actual_epoch)
                    self.save(config.checkpoint_dir, actual_epoch)
                    progbar = Progbar(target=self.train_steps_per_epoch)

                    # Evaluate val accuracy
                    val_error = 0
                    for i in range(self.val_steps_per_epoch):
                        # loss = sess.run(self.val_loss_eval, feed_dict={self.is_training: False})
                        results_val_loss = fetchesResultsValLoss(feed_dict={self.is_training: False})
                        loss = results_val_loss["loss_val"]
                        # loss = tf.cond(pred={self.is_training_ph: False}, true_fn=lambda: print('something went wrong'), false_fn=lambda: self.val_loss_eval)
                        # loss = tf.cond(pred={self.is_training_ph == False}, true_fn=lambda: print('something went wrong'), false_fn=lambda: self.val_loss_eval)
                        val_error += loss
                    val_error = val_error / self.val_steps_per_epoch
                    # Log to Tensorflow board
                    # val_sum = sess.run(self.val_error_log, feed_dict ={self.validation_error: val_error})
                    # results_val_error_log = fetches_results_val_error_log(feed_dict{self.validation_error: val_error})
                    # val_sum = results_val_error_log["loss_val_error_log"]
                    # val_sum = tf.cond(pred={self.validation_error: val_error}, true_fn=lambda: print('something went wrong'), false_fn=lambda: self.val_error_log)
                    # val_sum = tf.cond(pred={self.validation_error == val_error}, true_fn=lambda: print('something went wrong'), false_fn=lambda: self.val_error_log)

                    # sv.summary_writer.add_summary(val_sum, n_epochs)
                    # tf.summary.scalar("val_sum", val_sum, n_epochs)  # LK - no need for val_sum, because it only take summary tensor to summary numpy level. and we dont need sess.run for this anymore...

                    print("Epoch [{}] Validation Loss: {}".format(
                        actual_epoch, val_error))
                    if (n_epochs == self.config.max_epochs):
                        break
                train_summary_writer.flush()

    def testInit(self):
        """
           This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or for prediction of trajectories.
        """
        image_height, image_width = self.config.test_img_height, self.config.test_img_width

        self.num_channels = 3
        # input_uint8 = tf.compat.v1.placeholder(tf.uint8, [None, image_height, image_width, self.num_channels],
        #                                        name='raw_input')
        input_uint8 = tf.TensorSpec([None, image_height, image_width, self.num_channels], tf.uint8, name="raw_input")

        input_mc = self.preprocessImage(input_uint8)

        # pnt_batch = tf.compat.v1.placeholder(tf.float32, [None, self.config.output_dim], name='gt_labels')
        pnt_batch = tf.TensorSpec([None, self.config.output_dim], tf.float32, name='gt_labels')

        with tf.name_scope("trajectory_prediction"):
            pred_pnt = prediction_network(input_mc, output_dim=self.config.output_dim, f=self.config.f)

        with tf.name_scope("compute_loss"):
            # point_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:, :2], predictions=pred_pnt[:, :2])
            point_loss = tf.keras.losses.MSE(labels=pnt_batch[:, :2], predictions=pred_pnt[:, :2])
            # vel_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:, 2], predictions=pred_pnt[:, 2])
            vel_loss = tf.keras.losses.MSE(labels=pnt_batch[:, 2], predictions=pred_pnt[:, 2])

            total_loss = point_loss + vel_loss

        with tf.name_scope("metrics"):
            _, var = tf.nn.moments(x=pred_pnt, axes=-1)
            std = tf.math.sqrt(var)

        self.inputs_img = input_uint8
        self.pred_pnt = pred_pnt
        self.gt_pnt = pnt_batch
        self.pred_stds = std
        self.point_loss = point_loss
        self.total_loss = total_loss
        self.vel_loss = vel_loss

    def setup_inference(self, config, mode):
        """Sets up the inference graph.
        Args:
            mode: either 'loss' or 'prediction'. When 'loss', it will be used for
            computing a loss (gt trajectories should be provided). When
            'prediction', it will just make predictions (to be used in simulator)
            config: config dictionary. it should have target size and trajectories
        """
        self.mode = mode
        self.config = config
        self.testInit()

    # def inference(self, inputs, sess):
    def inference(self, inputs):
        results = {}
        fetches = {}
        if self.mode == 'loss':
            # fetches["vel_loss"] = self.vel_loss
            # fetches["pnt_loss"] = self.point_loss
            # fetches["stds"] = self.pred_stds
            @tf.function
            def fetches_results_test(self, feed_dict):
                if self:
                fetches["vel_loss"] = self.vel_loss
                fetches["pnt_loss"] = self.point_loss
                fetches["stds"] = self.pred_stds

            # results = sess.run(fetches, feed_dict={self.inputs_img: inputs['images'],
            #                                        self.gt_pnt: inputs['gt_labels']})
            results = ?????fetches_results(feed_dict={self.is_training: True})

        if self.mode == 'prediction':
            # results['predictions'] = sess.run(self.pred_pnt, feed_dict = {
            #     self.inputs_img: inputs['images']})

        return results
