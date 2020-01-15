import sys
import time
import tensorflow as tf
# import tensorflow.keras.utils.Progbar as Progbar
from tensorflow.keras.utils import Progbar as Progbar
from LearningHierarchy.LearningPipeline.Nets import ResNet8 as prediction_network
from LearningHierarchy.LearningPipeline.DataUtilities import *
import gflags
from LearningHierarchy.LearningPipeline.common_flags import FLAGS

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

TEST_PHASE = 0
TRAIN_PHASE = 1

# config = []
# config['batch_size'] = 5

# global config

# checkpoint_dir = r"C:\Users\chen\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\checkpoint"
# resume_train = False
# summary_freq = 100
# max_epochs = 10
# train_steps_per_epoch = 1000
# path_training_data = r"C:\Users\chen\PycharmProjects\deepdroneracing\Deep_Learning_TAU\simulation_training_data\Training"
"""
High level train function.
Args:
    self
    config: Configuration dictionary
Returns:
    None
"""


def dataLoading(path):
    iter_data = ImagesIterator(path)
    data_iter = iter_data.generateBatches()

    return data_iter, iter_data.num_samples


def loss(pred_pnt, pnt_batch, gamma):
    # mean squared error or squared error?
    point_loss = tf.keras.losses.MSE(y_true=pnt_batch[:, :2], y_pred=pred_pnt[:, :2])
    # tf.losses.MSE
    # point_loss = np.mean(point_loss)
    vel_loss = tf.keras.losses.MSE(y_true=pnt_batch[:, 2], y_pred=pred_pnt[:, 2])
    # vel_loss = np.mean(vel_loss)

    train_loss = point_loss + gamma * vel_loss

    return train_loss, point_loss, vel_loss


# def grad(output_dim, f, image_batch, pnt_batch, gamma):
def grad(image_batch, pnt_batch, gamma, mdl):
    with tf.GradientTape() as tape:
        # mdl = prediction_network(out_dim=output_dim, f=f)
        # pred_pnt = mdl.call(x=image_batch)
        pred_pnt = mdl.call(x=image_batch)
        train_loss, point_loss, vel_loss = loss(pred_pnt, pnt_batch, gamma)
    # train_vars = model.trainable_variables
    train_vars = mdl.trainable_variables
    train_grads = tape.gradient(train_loss, train_vars)
    grads_and_vars = zip(train_grads, train_vars)

    point_loss = np.mean(point_loss)
    vel_loss = np.mean(vel_loss)
    train_loss = np.mean(train_loss)

    return pred_pnt, train_loss, point_loss, vel_loss, train_grads, train_vars, grads_and_vars, mdl


def save(global_step, model, checkpoint_dir):
    model_name = 'model'
    # ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, model=model)
    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=100)  # object deletes old checkpoints.
    print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
    model.save_weights(os.path.join(checkpoint_dir, model_name + 'step'.format(global_step) + '.ckpt'))
    # model._trackable_saver.save(os.path.join(checkpoint_dir, model_name + '.latest'))


class TrajectoryLearner(object):
    def __init__(self, config):
        self.config = config

        return
        # pass

    # def train(self, train_data, n_samples_train):
    def train(self):
        """
        High level train function.
        Args:
            self
            config: Configuration dictionary
        Returns:
            None
        """

        # self.config = config

        gpu_config = tf.config.experimental.list_physical_devices('GPU')
        # device_name = tf.test.gpu_device_name()
        # print(tf.test.is_gpu_available())
        # print('Found GPU at: {}'.format(device_name))
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

        train_data, n_samples_train = dataLoading(self.config.train_dir)
        val_data, n_samples_val = dataLoading(self.config.val_dir, validation=True)


        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, beta_1=self.config.beta1)

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        self.train_summary_writer = tf.summary.create_file_writer(self.config.checkpoint_dir)

        self.train_steps_per_epoch = int(tf.math.ceil(n_samples_train / self.config.batch_size))

        if self.config.resume_train:
            print("Resume training from previous checkpoint")
            # self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.mdl.load_weights(...)

        # progbar = Progbar(target=self.train_steps_per_epoch)
        progbar = Progbar(target=self.config.max_epochs)

        step = -1

        self.mdl = prediction_network(out_dim=self.config.output_dim, f=self.config.f)

        for epoch in range(self.config.max_epochs):
            self.epoch = epoch

            progbar.update(self.epoch % self.config.max_epochs)

            train_loss_history = []
            val_loss_history = []

            for image_batch, pnt_batch in train_data:
                start_time = time.time()

                step += 1
                self.global_step = step

                # progbar.update(step % self.train_steps_per_epoch)

                # pred_pnt, train_loss, point_loss, vel_loss, train_grads, train_vars, self.grads_and_vars, self.mdl = \
                #     grad(self.config.output_dim, self.config.f, image_batch, pnt_batch, self.config.gamma)
                pred_pnt, train_loss, point_loss, vel_loss, train_grads, train_vars, self.grads_and_vars, self.mdl = \
                    grad(image_batch, pnt_batch, self.config.gamma, self.mdl)

                train_loss_history.append(train_loss)

                print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy()+1,
                                                          train_loss))

                self.train_op = optimizer.apply_gradients(self.grads_and_vars)

                self.global_step = step
                # self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

                # train_loss_name(train_loss)
                # train_accuracy(pnt_batch, pred_pnt)

                _, var = tf.nn.moments(x=pred_pnt, axes=-1)
                std = tf.math.sqrt(var)
                point_rmse = tf.math.sqrt(point_loss)
                vel_rmse = tf.math.sqrt(vel_loss)

                self.pred_pnt = pred_pnt
                self.gt_pnt = pnt_batch
                self.point_rmse = point_rmse
                self.vel_rmse = vel_rmse
                self.pred_stds = std
                self.image_batch = image_batch
                self.total_loss = train_loss
                # self.val_loss_eval = point_loss
                self.optimizer = optimizer

                if step % self.train_steps_per_epoch == 0 and epoch > 0:
                #     self.save(self.config.checkpoint_dir, epoch)
                    save(self.global_step, self.mdl, self.config.checkpoint_dir)

                if step % self.config.summary_freq == 0 and step >= self.config.summary_freq:
                    train_epoch = tf.math.ceil(self.global_step / self.train_steps_per_epoch)
                    train_step = self.global_step - (train_epoch - 1) * self.train_steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it point_rmse: %.3f " \
                          "vel_rmse: %.6f, point_std: %.6f, vel_std: %.6f"
                          % (train_epoch, train_step, self.train_steps_per_epoch, \
                             time.time() - start_time, self.point_rmse, self.vel_rmse,
                             np.mean(self.pred_stds[:2]), self.pred_stds[2]))

                    tf.summary.scalar("point_rmse", self.point_rmse, step=self.global_step)
                    tf.summary.scalar("vel_rmse", self.vel_rmse, step=self.global_step)
                    tf.summary.image("image", self.image_batch, step=self.global_step)

                self.train_summary_writer.flush()

            # print here the averaged loss for the entire epoch!... and print epoch:



def fit():
    # c = {}
    # c["path_training_data"] = \
    #     r"C:\Users\chen\PycharmProjects\deepdroneracing\Deep_Learning_TAU\simulation_training_data\Training"

    # Utility main to load flags
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)

    # train_data, n_samples_train = dataLoading(c["path_training_data"])
    # TrajectoryLearner.train(self, config, train_data)
    trl = TrajectoryLearner(FLAGS)
    # trl.train(FLAGS, train_data, n_samples_train)
    trl.train()

fit()
