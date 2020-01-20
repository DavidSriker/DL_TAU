from LearningHierarchy.LearningPipeline.Nets import ResNet8
from LearningHierarchy.LearningPipeline.DataUtilities import *
from datetime import datetime
import matplotlib.pyplot as plt

from tensorboard.plugins.hparams import api as hp


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

class DataMode:
    train = 0
    validation = 1
    test = 2


class TrajectoryLearner(object):
    def __init__(self, config):
        self.config = config
        self.data_modes = DataMode()
        return

    def dataLoading(self, mode):
        """
        mode=0 -> training
        mode=1 -> validation
        mode=2 -> test
        """
        if mode == 0:
            img_iter = ImagesIterator(directory=self.config.train_dir)
        elif mode == 1:
            img_iter = ImagesIterator(directory=self.config.val_dir)
        elif mode == 2:
            img_iter = ImagesIterator(directory=self.config.val_dir, batch_s=1)
        else:
            print("Wrong mode, should be either 0,1,2; please check!")
        data_iter = img_iter.generateBatches()
        return data_iter, img_iter.num_samples

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
        return path

    def saveTFLiteModel(self, saved_model_path):
        model_name = self.mdl.name
        lite_model_path = os.path.join(saved_model_path, "TFLITE_MODEL")
        if not os.path.exists(lite_model_path):
            os.mkdir(lite_model_path)
        lite_mdl = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        lite_mdl_converted = lite_mdl.convert()
        open(os.path.join(lite_model_path, model_name +
                          '_epoch_{:}_converted_model.tflite'.format(self.config.max_epochs)), "wb").write(lite_mdl_converted)
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
        train_data, n_samples_train = self.dataLoading(self.data_modes.train)
        val_data, n_samples_val = self.dataLoading(self.data_modes.validation)

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
        if self.config.tflite:
            p = self.save()
            self.saveTFLiteModel(p)

        print(20 * "-", "Done Training", 20 * "-")
        return

    def test(self):

        self.setGPUs()

        # data setup
        test_data, n_samples_test = self.dataLoading(self.data_modes.test)

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

    def testTFLite(self, tflite_path):

        # todo need to add logic to get the desired tflite path
        tflite_path = os.path.join(tflite_path, "res_net8_epoch_4", "TFLITE_MODEL", "res_net8_epoch_4_converted_model.tflite")
        self.setGPUs()
        # network setup
        self.tf_lite_mdl = tf.lite.Interpreter(model_path=tflite_path)
        self.tf_lite_mdl.allocate_tensors()
        in_details = self.tf_lite_mdl.get_input_details()
        out_details = self.tf_lite_mdl.get_output_details()

        # data setup
        test_data, n_samples_test = self.dataLoading(self.data_modes.test)

        # evaluate
        for img, gt in test_data:
            self.tf_lite_mdl.set_tensor(in_details[0]['index'], img.numpy())
            self.tf_lite_mdl.invoke()
            pred = self.tf_lite_mdl.get_tensor(out_details[0]['index'])
            diff = np.sum(np.abs(pred - gt.numpy()))
            print(diff)
        return
