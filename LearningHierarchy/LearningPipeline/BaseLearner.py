import sys
import time
import numpy as np
import tensorflow as tf
# import tensorflow.keras.utils.Progbar as Progbar
# from tensorflow.keras.utils import Progbar as Progbar
from LearningHierarchy.LearningPipeline.Nets import ResNet8
from LearningHierarchy.LearningPipeline.DataUtilities import *
# import gflags
# from LearningHierarchy.LearningPipeline.common_flags import FLAGS
# from LearningHierarchy.LearningPipeline.CommonFlags import *

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

TEST_PHASE = 0
TRAIN_PHASE = 1

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

    def grad(self, image_batch, labels_batch, validation=False):
        with tf.GradientTape() as tape:
            pred = self.mdl.call(x=image_batch)
            loss, coordinate_loss, velocity_loss = self.loss(pred, labels_batch)
        if not validation:
            train_vars = self.mdl.trainable_variables
            train_grads = tape.gradient(loss, train_vars)
            grads_and_vars = zip(train_grads, train_vars)

        point_loss = np.mean(coordinate_loss)
        vel_loss = np.mean(velocity_loss)
        train_loss = np.mean(loss)

        if not validation:
            return pred, train_loss, point_loss, vel_loss, train_grads, train_vars, grads_and_vars
        else:
            return pred, train_loss, point_loss, vel_loss

    def save(self):
        model_name = 'model_test'
        print(" [*] Saving checkpoint to %s..." % self.config.checkpoint_dir)
        path = os.path.join(self.config.checkpoint_dir, model_name + '_step_{:}'.format(self.global_step))
        # self.mdl.save(path)
        tf.saved_model.save(self.mdl, path)
        # self.mdl.save_weights(os.path.join(checkpoint_dir, model_name + 'step'.format(self.global_step) + '.ckpt'))

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

        # if self.config.resume_train:
        #     print("Resume training from previous checkpoint")
        #     # self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        #     self.mdl.load_weights(self.config.pb_file)

        # # progbar = Progbar(target=self.train_steps_per_epoch)
        # progbar = Progbar(target=self.config.max_epochs)

        # init_step = 0  # change manually if we restore a checkpoint and add +1.
        # step = init_step - 1

        self.mdl = ResNet8(out_dim=self.config.output_dim, f=self.config.f)

        self.mdl.compile(optimizer=custom_optimizer,
                        loss=self.loss)

        self.mdl.fit(train_data,
                     epochs=self.config.max_epochs,
                     validation_data=val_data)


        # for epoch in range(self.config.max_epochs):
        #     self.epoch = epoch
        #
        #     progbar.update(self.epoch % self.config.max_epochs)
        #
        #     train_loss_history = []
        #     val_loss_history = []
        #
        #     for image_batch, pnt_batch in train_data:
        #         start_time = time.time()
        #
        #         step += 1
        #         self.global_step = step
        #
        #         # progbar.update(step % self.train_steps_per_epoch)
        #
        #         # pred_pnt, train_loss, point_loss, vel_loss, train_grads, train_vars, self.grads_and_vars, self.mdl = \
        #         #     grad(self.config.output_dim, self.config.f, image_batch, pnt_batch, self.config.gamma)
        #         pred_pnt, train_loss, point_loss, vel_loss, train_grads, train_vars, self.grads_and_vars = \
        #             self.grad(image_batch, pnt_batch)
        #
        #         train_loss_history.append(train_loss)
        #
        #         # print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy()+1,
        #         #                                           train_loss))
        #
        #         self.train_op = custom_optimizer.apply_gradients(self.grads_and_vars)
        #
        #         self.global_step = step
        #         # self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)
        #
        #         # train_loss_name(train_loss)
        #         # train_accuracy(pnt_batch, pred_pnt)
        #
        #         # _, var = tf.nn.moments(x=pred_pnt, axes=-1)
        #         # std = tf.math.sqrt(var)
        #         # point_rmse = tf.math.sqrt(point_loss)
        #         # vel_rmse = tf.math.sqrt(vel_loss)
        #         #
        #         # self.pred_pnt = pred_pnt
        #         # self.gt_pnt = pnt_batch
        #         # self.point_rmse = point_rmse
        #         # self.vel_rmse = vel_rmse
        #         # self.pred_stds = std
        #         # self.image_batch = image_batch
        #         # self.total_loss = train_loss
        #         # # self.val_loss_eval = point_loss
        #         # self.optimizer = optimizer
        #         #
        #         # if step % self.train_steps_per_epoch == 0 and epoch > 0:
        #         # #     self.save(self.config.checkpoint_dir, epoch)
        #         #     save(self.global_step, self.mdl, self.config.checkpoint_dir)
        #         #
        #         # if step % self.config.summary_freq == 0 and step >= self.config.summary_freq:
        #         #     train_epoch = tf.math.ceil(self.global_step / self.train_steps_per_epoch)
        #         #     train_step = self.global_step - (train_epoch - 1) * self.train_steps_per_epoch
        #         #     print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it point_rmse: %.3f " \
        #         #           "vel_rmse: %.6f, point_std: %.6f, vel_std: %.6f"
        #         #           % (train_epoch, train_step, self.train_steps_per_epoch, \
        #         #              time.time() - start_time, self.point_rmse, self.vel_rmse,
        #         #              np.mean(self.pred_stds[:2]), self.pred_stds[2]))
        #         #
        #         #     tf.summary.scalar("point_rmse", self.point_rmse, step=self.global_step)
        #         #     tf.summary.scalar("vel_rmse", self.vel_rmse, step=self.global_step)
        #         #     tf.summary.image("image", self.image_batch, step=self.global_step)
        #         #
        #         # self.train_summary_writer.flush()
        #
        #     # print here the averaged loss for the entire epoch!... and print epoch:
        #
        #     for image_batch_val, pnt_batch_val in val_data:
        #         pred_pnt_val, val_loss, point_loss_val, vel_loss_val = \
        #             self.grad(image_batch_val, pnt_batch_val, validation=True)
        #
        #         val_loss_history.append(val_loss)
        #
        #     print("Epoch: {}, mean train loss: {}, mean val loss: {}".format(self.epoch, np.mean(train_loss_history)*100,
        #                                                                      np.mean(val_loss_history)*100))
        #     self.save()


# def fit():
#
#     # Utility main to load flags
#     try:
#         argv = FLAGS(sys.argv)  # parse flags
#     except gflags.FlagsError:
#         print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
#         sys.exit(1)
#
#     trl = TrajectoryLearner(FLAGS)
#     trl.train()
#
# fit()
