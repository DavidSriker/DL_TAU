from LearningHierarchy.LearningPipeline.Nets import ResNet8
from LearningHierarchy.LearningPipeline.DataUtilities import *
from datetime import datetime
import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp


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

    def setOptimizer(self, mode=0):
        if mode == 0:
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate,
                                                  beta_1=self.config.beta1)
        return

    def save(self):
        model_name = self.mdl.name
        print(" [*] Saving checkpoint to %s..." % self.config.checkpoint_dir)
        path = os.path.join(self.config.checkpoint_dir, model_name + '_epoch_{:}'
                            .format(self.config.max_epochs))
        if not os.path.exists(path):
            os.mkdir(path)
        # tf.saved_model.save(self.mdl, path)
        self.mdl.save(path)  # does the exact same thing
        return

    def setTensorboardSummaries(self):
        # tensor board summary visualization callback
        path_summary_callback_temp = os.path.join(self.config.checkpoint_dir, 'TrainingLog', 'Logs')
        if not os.path.exists(path_summary_callback_temp):
            os.mkdir(path_summary_callback_temp)
        path_summary_callback = os.path.join(self.config.checkpoint_dir, 'TrainingLog', 'Logs', self.mdl.name + "_" +
                                             datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if not os.path.exists(path_summary_callback):
            os.mkdir(path_summary_callback)
        self.summary_callback = tf.keras.callbacks.TensorBoard(log_dir=path_summary_callback, histogram_freq=1,
                                                          write_images=True,
                                                          # update_freq=self.config.summary_freq_epoch)
                                                          update_freq='epoch')
        return

    def setModelCallbacks(self):
        path_save_callback_temp = os.path.join(self.config.checkpoint_dir, 'TrainingLog')
        if not os.path.exists(path_save_callback_temp):
            os.mkdir(path_save_callback_temp)
        path_save_callback = os.path.join(self.config.checkpoint_dir, 'TrainingLog', self.mdl.name + "_" +
                                          datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        if not os.path.exists(path_save_callback):
            os.mkdir(path_save_callback)
        self.save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_save_callback, '.ckpt'),
                                                           save_weights_only=True,
                                                           verbose=False,
                                                           # save_freq=self.config.save_latest_freq  # saves every few steps...
                                                           period=self.config.save_latest_period)  # saves every few epochs
        return

    def setGPUs(self):
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
        return

    def train(self):
        self.setGPUs()
        # data setup
        train_data, n_samples_train = self.dataLoading()
        val_data, n_samples_val = self.dataLoading(validation=True)

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # network setup
        self.mdl = ResNet8(out_dim=self.config.output_dim, f=self.config.f)
        if self.config.resume_train:
            # load .pb file
            print("Resume training from previous checkpoint")
            latest_ckpt_file = max([os.path.join(dp, f) for dp, dn, filenames in
                                    os.walk(os.path.join(self.config.checkpoint_dir, 'TrainingLog')) for f in
                                    filenames if '.ckpt' in f], key=os.path.getmtime)
            self.latest_ckpt_directory = os.path.dirname(latest_ckpt_file)
            self.mdl.load_weights(os.path.join(self.latest_ckpt_directory, '.ckpt'))

        self.setOptimizer()
        self.mdl.compile(optimizer=self.optim,
                        loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # set callbacks
        self.setModelCallbacks()
        # tensor board summary visualization callback
        self.setTensorboardSummaries()

        # training
        self.history = self.mdl.fit(train_data,
                                    epochs=self.config.max_epochs,
                                    validation_data=val_data,
                                    callbacks=[self.save_callback, self.summary_callback])

        self.mdl.summary()
        # self.save() TODO - Do we need this?

        print(20 * "-", "Done Training", 20 * "-")
        return

    def test(self):

        self.setGPUs()

        # data setup
        test_data, n_samples_test = self.dataLoading(validation=True)

        # network setup
        self.mdl_test = ResNet8(out_dim=self.config.output_dim, f=self.config.f)

        latest_ckpt_file = max([os.path.join(dp, f) for dp, dn, filenames in
                                os.walk(os.path.join(self.config.checkpoint_dir, 'TrainingLog')) for f in
                                filenames if '.ckpt' in f], key=os.path.getmtime)
        self.latest_ckpt_directory = os.path.dirname(latest_ckpt_file)
        self.mdl_test.load_weights(os.path.join(self.latest_ckpt_directory, '.ckpt'))
        self.setOptimizer()
        self.mdl_test.compile(optimizer=self.optim,
                         loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # evaluate
        results = self.mdl_test.evaluate(test_data)

        print(20 * "-", "Done Evaluating", 20 * "-")

        print('Test Loss: {:}\nTest Accuracy: {:}\nTest MSE: {:}'.format(*results))
        return
