#Base model for learning

import os
import time
# import math
from itertools import count
import random
import tensorflow as tf
import numpy as np
from keras.utils.generic_utils import Progbar
from .nets import resnet8 as prediction_network
# from .data_utils import DirectoryIterator
from DataUtilities import ImagesIterator  # LK - DataUtilities???
# import tensorflow_compression as tfc  # tf.Variable is not supported in eager mode, instead use tfc.Variable meaning tensorflow constant variable??
# tf.enable_eager_execution()  # is it a default in tf2.x? or should we call this? i know that tf.Session and tf.enable_eager_execution are gone!

# LK - maybe add this: tf.enable_eager_execution()
# tf_upgrade_v2 --infile C:\Users\user\Desktop\ddlearning_tau\base_learner.py --outfile C:\Users\user\Desktop\ddlearning_tau\BaseLearner.py
# https://stackoverflow.com/questions/47251280/capacity-of-queue-in-tf-data-dataset
# You may want to wrap your counter in a class as Variables in Eager get deleted when they run out of scope.

TEST_PHASE = 0
TRAIN_PHASE = 1

class TrajectoryLearner(object):
    def __init__(self):
        pass

    def readFromDisk(self, inputs_queue):
        """Consumes the inputs queue.
        Args:
            filename_and_label_tensor: A scalar string tensor.
        Returns:
            Two tensors: the decoded images, and the labels.
        """
        pnt_seq = tf.cast(inputs_queue[1], dtype=tf.float32)
        file_content = tf.io.read_file(inputs_queue[0])  # file_content = tf.read_file(inputs_queue[0])
        image_seq = tf.image.decode_jpeg(file_content, channels=3)  # Decode a JPEG-encoded image to a uint8 tensor. # image_seq = tf.io.decode_jpeg(file_content, channels=3)  # Aliases: tf.image.decode_jpeg # This op also supports decoding PNGs and non-animated GIFs since the interface is the same, though it is cleaner to use

        return image_seq, pnt_seq

    def preprocessImage(self, image):
        """ Preprocess an input image
        Args:
            Image: A uint8 tensor
        Returns:
            image: A preprocessed float32 tensor.
        """
        image = tf.image.resize(image,  # tf.image.resize_images
                [self.config.img_height, self.config.img_width])  # ResizeMethod. Defaults to bilinear
        image = tf.cast(image, dtype=tf.float32)
        image = tf.divide(image, 255.0)
        return image

    def getFilenamesList(self, directory):
        # Load labels, velocities and image filenames. The shuffling will be done after
        # iterator = DirectoryIterator(directory, shuffle=False)
        iterator = ImagesIterator(directory=directory, shuffle=False, batch_s=self.config.batch_size)
        # tf.keras.preprocessing.image.DirectoryIterator
        return iterator.filenames, iterator.ground_truth

    def buildTrainGraph(self):
        # is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=(), name="is_training")
        is_training_ph = tf.TensorSpec([], tf.bool, name="is_training")
        # with tf.compat.v1.name_scope("data_loading"):
        with tf.name_scope("data_loading"):
            # generate training and validation batches ( we do not need labels)
            train_batch, n_samples_train = self.generateBatches(self.config.train_dir)
            val_batch, n_samples_test = self.generateBatches(self.config.val_dir, validation=True)

            # image_batch, pnt_batch, n_samples = self.generateBatches(self.config.train_dir)
            current_batch = tf.cond(pred=is_training_ph, true_fn=lambda: train_batch, false_fn=lambda: val_batch)  # Return true_fn() if the predicate pred is true else false_fn().

            # current_batch = tf.cond(pred=self.is_training,true_fn=lambda: train_batch, false_fn=lambda: val_batch)  # Return true_fn() if the predicate pred is true else false_fn().

            image_batch, pnt_batch = current_batch[0], current_batch[1]

        with tf.GradientTape() as tape:
            # with tf.compat.v1.name_scope("trajectory_prediction"):
            with tf.name_scope("trajectory_prediction"):
                pred_pnt = prediction_network(image_batch, output_dim=self.config.output_dim, f=self.config.f)

            # with tf.compat.v1.name_scope("compute_loss"):
            with tf.name_scope("compute_loss"):
                # point_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:,:2],
                #                                           predictions=pred_pnt[:,:2])
                point_loss = tf.keras.losses.MSE(labels=pnt_batch[:,:2], predictions=pred_pnt[:,:2])

                # vel_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:, 2],
                #                                         predictions=pred_pnt[:,2])
                vel_loss = tf.keras.losses.MSE(labels=pnt_batch[:, 2], predictions=pred_pnt[:,2])

                train_loss = point_loss + 0.1 * vel_loss

        # with tf.compat.v1.name_scope("metrics"):
        with tf.name_scope("metrics"):
            _, var = tf.nn.moments(x=pred_pnt, axes=-1)  # calculates the mean and variance of x
            std = tf.math.sqrt(var)  # std = tf.sqrt(var)
            point_rmse = tf.math.sqrt(point_loss)  # point_rmse = tf.sqrt(point_loss)
            vel_rmse = tf.math.sqrt(vel_loss)  # vel_rmse = tf.sqrt(vel_loss)

        # with tf.compat.v1.name_scope("train_op"):
        # tf 2.x implementation:
        with tf.name_scope("train_op"):
            train_vars = [var for var in tf.trainable_variables()] # train_vars = [var for var in tf.compat.v1.trainable_variables()]
            # optimizer  = tf.compat.v1.train.AdamOptimizer(self.config.learning_rate,  # Optimizer that implements the Adam algorithm.
            #                                  self.config.beta1)
            optimizer  = tf.keras.optimizers.AdamOptimizer(self.config.learning_rate,
                                             self.config.beta1)
            # self.grads_and_vars = optimizer.compute_gradients(train_loss,  # optimizer.compute_gradients used tf.gradients. this is as in TF1.x.
            #                                               var_list=train_vars)

        with tf.name_scope("train_op")
            train_grads = tape.gradient(train_loss, train_vars)
            self.grads_and_vars = zip(train_grads, train_vars)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.incr_global_step = tf.assign(self.global_step, self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary), maybe add
        # images

        self.train_steps_per_epoch = \
            int(tf.math.ceil(n_samples_train/self.config.batch_size))
        self.val_steps_per_epoch = \
            int(tf.math.ceil(n_samples_test/self.config.batch_size))
        self.pred_pnt = pred_pnt
        self.gt_pnt = pnt_batch
        self.point_rmse = point_rmse
        self.vel_rmse = vel_rmse
        self.pred_stds = std
        self.image_batch = image_batch
        self.is_training = is_training_ph
        self.total_loss = train_loss
        self.val_loss_eval = point_loss

    def generateBatches(self, data_dir, validation=False):
        seed = random.randint(0, 2**31 - 1)
        # Load the list of training files into queues
        file_list, pnt_list= self.getFilenamesList(data_dir)
        # Convert to tensors before passing
        # inputs_queue = tf.compat.v1.train.slice_input_producer([file_list, pnt_list],seed=seed, shuffle=not validation)
        inputs_queue = tf.data.Dataset.from_tensor_slices([file_list, pnt_list]).shuffle(not validation, seed=seed)  # Return the objects of sliced elements
        def mapReadImage(inputs_queue)
            image_seq, pnt_seq = self.readFromDisk(inputs_queue)
            # Resize images to target size and preprocess them
            image_seq = self.preprocessImage(image_seq)
            return image_seq, pnt_seq
        # Form training batches
        # image_batch, pnt_batch = tf.compat.v1.train.batch([image_seq,
        #      pnt_seq],
        #      batch_size=self.config.batch_size,`
        #      # This should be 1 for validation, but makes training significantly slower.
        #      # Since we are anyway not interested in the absolute value of the metrics, we keep it > 1.
        #      num_threads=self.config.num_threads,
        #      capacity=self.config.capacity_queue,
        #      allow_smaller_final_batch=validation)`
        dset = tf.data.Dataset.batch(batch_size=self.config.batch_size, drop_remainder=not validation). # LK - need to change this to prefetch. drop remainder instead of allow_small
         # This should be 1 for validation, but makes training significantly slower.
         # Since we are anyway not interested in the absolute value of the metrics, we keep it > 1.
             # map(mapReadImage, num_parallel_calls=self.config.num_threads, capacity=self.config.capacity_queue)  # LK - do i need to creat single _imread function???, maybe delete the key word capacity?
             map(mapReadImage, num_parallel_calls=self.config.num_threads).prefetch(2) #  LK - i think config.capacity_queue is not defined! there for instead use dset.prefetch(2)

        image_batch, pnt_batch = dset


        return [image_batch, pnt_batch], len(file_list)





    def collect_summaries(self):

        pnt_error_sum = tf.summary.scalar("point_rmse", self.point_rmse)  # tf.compat.v1.summary.scalar("point_rmse", self.point_rmse)
        vel_error_sum = tf.summary.scalar("vel_rmse", self.vel_rmse)  # tf.compat.v1.summary.scalar("vel_rmse", self.vel_rmse)
        image_sum = tf.summary.image("image", self.image_batch)  # tf.compat.v1.summary.image("image", self.image_batch)
        self.step_sum_op = tf.compat.v1.summary.merge([pnt_error_sum, vel_error_sum, image_sum])
        self.validation_error = tf.compat.v1.placeholder(tf.float32, [])
        self.val_error_log = tf.summary.scalar("Validation_Error", self.validation_error)  # tf.compat.v1.summary.scalar("Validation_Error", self.validation_error)

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        """High level train function.
        Args:
            config: Configuration dictionary
        Returns:
            None
        """
        self.config = config
        self.buildTrainGraph()
        self.collect_summaries()
        with tf.compat.v1.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) \
                                        for v in tf.compat.v1.trainable_variables()])
        self.saver = tf.compat.v1.train.Saver([var for var in \
            tf.compat.v1.trainable_variables()] +  [self.global_step], max_to_keep=100)
        sv = tf.compat.v1.train.Supervisor(logdir=config.checkpoint_dir, save_summaries_secs=0, saver=None)

        # LK - example from migrate.ipynb:
        # Example usage with eager execution, the default in TF 2.0:
        # writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
        # with writer.as_default():
        #   for step in range(100):
        #     # other model code would go here
        #     tf.summary.scalar("my_metric", 0.5, step=step)
        #     writer.flush()
        # so maybe i can change the "sv = " line to:
        # sv = tf.compat.v1.train.Supervisor(logdir=config.checkpoint_dir, save_summaries_secs=0, saver=None)

        gpu_config = tf.compat.v1.ConfigProto()
        # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        # for device in gpu_devices:
        #     tf.config.experimental.set_memory_growth(device, True)
        gpu_config.gpu_options.allow_growth=True
        with sv.managed_session(config=gpu_config) as sess:
            print("Number of params: {}".format(sess.run(parameter_count)))
            if config.resume_train:
                print("Resume training from previous checkpoint")
                checkpoint = tf.train.latest_checkpoint(
                                                config.checkpoint_dir)
                self.saver.restore(sess, checkpoint)

            progbar = Progbar(target=self.train_steps_per_epoch)

            n_epochs = 0

            for step in count(start=1):
                if sv.should_stop():
                    break
                start_time = time.time()
                fetches = { "train" : self.train_op,
                              "global_step" : self.global_step,
                              "incr_global_step": self.incr_global_step
                             }
                if step % config.summary_freq == 0:
                    fetches["vel_rmse"] = self.vel_rmse
                    fetches["pnt_rmse"] = self.point_rmse
                    fetches["stds"] = self.pred_stds
                    fetches["summary"] = self.step_sum_op

                # Runs a series of operations
                results = sess.run(fetches,
                                   feed_dict={self.is_training: True})

                progbar.update(step % self.train_steps_per_epoch)


                gs = results["global_step"]

                if step % config.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil( gs /self.train_steps_per_epoch)
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
                    self.save(sess, config.checkpoint_dir, actual_epoch)
                    progbar = Progbar(target=self.train_steps_per_epoch)
                    # Evaluate val accuracy
                    val_error = 0
                    for i in range(self.val_steps_per_epoch):
                        loss = sess.run(self.val_loss_eval, feed_dict={
                            self.is_training: False})
                        val_error += loss
                    val_error = val_error / self.val_steps_per_epoch
                    # Log to Tensorflow board
                    val_sum = sess.run(self.val_error_log, feed_dict ={
                        self.validation_error: val_error})
                    sv.summary_writer.add_summary(val_sum, n_epochs)
                    print("Epoch [{}] Validation Loss: {}".format(
                        actual_epoch, val_error))
                    if (n_epochs == self.config.max_epochs):
                        break


    def build_test_graph(self):
        """This graph will be used for testing. In particular, it will
           compute the loss on a testing set, or for prediction of trajectories.
        """
        image_height, image_width = self.config.test_img_height, \
                                    self.config.test_img_width

        self.num_channels = 3
        input_uint8 = tf.compat.v1.placeholder(tf.uint8, [None, image_height,
                                    image_width, self.num_channels],
                                    name='raw_input')


        input_mc = self.preprocessImage(input_uint8)

        pnt_batch = tf.compat.v1.placeholder(tf.float32, [None, self.config.output_dim],
                                          name='gt_labels')


        with tf.compat.v1.name_scope("trajectory_prediction"):
            pred_pnt = prediction_network(input_mc,
                    output_dim=self.config.output_dim, f=self.config.f)

        with tf.compat.v1.name_scope("compute_loss"):
            point_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:,:2],
                                                      predictions=pred_pnt[:,:2])

            vel_loss = tf.compat.v1.losses.mean_squared_error(labels=pnt_batch[:, 2],
                                                    predictions=pred_pnt[:, 2])
            total_loss = point_loss + vel_loss


        with tf.compat.v1.name_scope("metrics"):
            _, var = tf.nn.moments(x=pred_pnt, axes=-1)
            std = tf.sqrt(var)

        self.inputs_img = input_uint8
        self.pred_pnt = pred_pnt
        self.gt_pnt = pnt_batch
        self.pred_stds = std
        self.point_loss = point_loss
        self.total_loss = total_loss
        self.vel_loss = vel_loss

    def setupInference(self, config, mode):
        """Sets up the inference graph.
        Args:
            mode: either 'loss' or 'prediction'. When 'loss', it will be used for
            computing a loss (gt trajectories should be provided). When
            'prediction', it will just make predictions (to be used in simulator)
            config: config dictionary. it should have target size and trajectories
        """
        self.mode = mode
        self.config = config
        self.build_test_graph()

    def inference(self, inputs, sess):
        results = {}
        fetches = {}
        if self.mode == 'loss':
            fetches["vel_loss"] = self.vel_loss
            fetches["pnt_loss"] = self.point_loss
            fetches["stds"] = self.pred_stds

            results = sess.run(fetches,
                               feed_dict= {self.inputs_img: inputs['images'],
                                           self.gt_pnt: inputs['gt_labels']})
        if self.mode == 'prediction':
            results['predictions'] = sess.run(self.pred_pnt, feed_dict = {
                self.inputs_img: inputs['images']})

        return results
