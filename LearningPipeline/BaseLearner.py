from LearningPipeline.Nets import *
from LearningPipeline.DataUtilities import *
from datetime import datetime
import time
import matplotlib.pyplot as plt
import gc
import pickle


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
        self.model_name = config.net_name
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
            img_iter = ImagesIterator(directory=self.config.test_dir, batch_s=1)
        else:
            print("Wrong mode, should be either 0,1,2; please check!")
        data_iter = img_iter.generateBatches()
        if mode == 2:
            return data_iter, img_iter.num_samples, img_iter.data
        else:
            return data_iter, img_iter.num_samples

    def loss(self, y_true, y_pred):
        coordinate_loss = tf.keras.losses.MSE(y_true=y_true[:, :2], y_pred=y_pred[:, :2])
        velocity_loss = tf.keras.losses.MSE(y_true=y_true[:, 2], y_pred=y_pred[:, 2])
        loss = coordinate_loss + self.config.gamma * velocity_loss
        return loss

    def setMdl(self, mode="ResNet8", test_mode=False):
        self.model_name = mode
        if mode == "ResNet8":
            mdl = ResNet8(out_dim=self.config.output_dim, f=self.config.f)
        elif mode == "ResNet8b":
            mdl = ResNet8b(out_dim=self.config.output_dim, f=self.config.f)
        elif mode == "TCResNet8":
            mdl = TCResNet8(out_dim=self.config.output_dim, f=self.config.f)
        else:
            print("Wrong mode, should be either ResNet8, ResNet8b, TCResNet; please check!")
            return
        return mdl

    def setOptimizer(self, mode="Adam"):
        if mode == "Adam":
            self.optim = tf.keras.optimizers.Adam(learning_rate=self.config.lr_adam,
                                                  beta_1=0.7, beta_2=0.9,
                                                  amsgrad=True)
        elif mode == "SGD":
            self.optim = tf.keras.optimizers.SGD(learning_rate=self.config.lr_sgd,
                                                 momentum=0.2, nesterov=True)
        elif mode == "Adagrad":
            self.optim = tf.keras.optimizers.Adagrad(learning_rate=self.config.lr_adagrad)
        elif mode == "Adadelta":
            self.optim = tf.keras.optimizers.Adadelta(learning_rate=self.config.lr_adadelta,
                                                      rho=0.7)
        else:
            print("Wrong mode, should be either Adam, SGD, Adagrad, Adadelta; please check!")
        return

    def saveHistory(self, saved_model_path, history_obj):
        history_dir = os.path.join(saved_model_path, "Hisotry")
        if not os.path.exists(history_dir):
            os.mkdir(history_dir)

        with open(os.path.join(history_dir, "trainHistoryDict"), 'wb') as f:
            pickle.dump(history_obj.history, f)
        return

    def save(self, optim):
        name = self.model_name + '_epoch_{:}'.format(self.config.max_epochs) + "_" + optim
        path = os.path.join(self.config.checkpoint_dir, name)
        print(" [*] Saving checkpoint to %s..." % path)
        if not os.path.exists(path):
            os.mkdir(path)

        self.mdl.save(filepath=path, save_format='tf')  # does the exact same thing
        self.mdl.save_weights(filepath=path + "/")
        return path

    def saveTFLiteModel(self, saved_model_path, optim):
        lite_model_path = os.path.join(saved_model_path, "TFLITE_MODEL")
        if not os.path.exists(lite_model_path):
            os.mkdir(lite_model_path)

        lite_mdl = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        lite_mdl_converted = lite_mdl.convert()
        name = self.model_name + '_epoch_{:}'.format(self.config.max_epochs) + "_" + optim + ".tflite"
        open(os.path.join(lite_model_path, name), "wb").write(lite_mdl_converted)
        return

    def setGPUs(self):
        gpu_config = tf.config.experimental.list_physical_devices('GPU')
        if gpu_config:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpu_config:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    gpus = tf.config.experimental.list_physical_devices('GPU')
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        return

    def saveTestImagesWithAnnotations(self, data_dict, dest_path):
        for run in data_dict.keys():
            imgas_run_path = os.path.join(dest_path, run)
            if not os.path.exists(imgas_run_path):
                os.mkdir(imgas_run_path)
            for img, gt, path in zip(data_dict[run]["images"], data_dict[run]["gt"], data_dict[run]["paths"]):
                pred = self.mdl_test.call(img)
                # origin_name = path.split("/")[-1]
                origin_name = os.path.basename(path)
                img_path = os.path.join(imgas_run_path, origin_name)
                self.plotTestPred(img.numpy(), pred.numpy()[0], gt.numpy()[0], img_path)
        return

    def unnormalizeCoordinates(self, pred, gt):
        img_w, img_h = self.config.test_img_width, self.config.test_img_height
        # un-normalize predictions and corresponding GTs:
        pred = np.multiply(pred, [img_w / 2, img_h / 2, 1])
        pred[:2] = np.max(np.floor(pred[0] + img_w / 2), 0), np.max(np.floor(pred[1] + img_h / 2), 0)
        gt = np.multiply(gt, [img_w / 2, img_h / 2, 1])
        gt[:2] = np.max(np.floor(gt[0] + img_w / 2), 0), np.max(np.floor(gt[1] + img_h / 2), 0)
        return pred, gt

    def lossForPlots(self, pred, gt):
        coor_loss = np.mean(np.power(pred[:2] - gt[:2], 2))
        vel_loss = np.power(pred[2] - gt[2], 2)
        return 100 * [coor_loss, vel_loss, coor_loss + self.config.gamma * vel_loss]

    def plotTestPred(self, img, pred, gt, img_path):
        fig, ax = plt.subplots(1)
        plt.get_current_fig_manager().full_screen_toggle()
        ax.imshow(img.reshape(img.shape[1:]))
        pred, gt = self.unnormalizeCoordinates(pred, gt)
        ax.scatter(pred[0], pred[1], color='green')
        ax.scatter(gt[0], gt[1], color='red')

        ax.figure.savefig(img_path)
        plt.close("all")
        gc.collect()
        return

    def train(self, optim_mode="Adam", net_mode="ResNet8"):
        self.setGPUs()
        # data setup
        train_data, n_samples_train = self.dataLoading(self.data_modes.train)
        val_data, n_samples_val = self.dataLoading(self.data_modes.validation)

        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)

        # network setup
        self.mdl = self.setMdl(net_mode)
        self.setOptimizer(optim_mode)
        self.mdl.compile(optimizer=self.optim,
                        loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # training
        history = self.mdl.fit(train_data,
                               epochs=self.config.max_epochs,
                               validation_data=val_data)

        if self.config.save_model:
            p = self.save(optim_mode)
            self.saveHistory(p, history)
            if self.config.tflite:
                self.saveTFLiteModel(p, optim_mode)

        print(20 * "-", "Done Training", 20 * "-")
        return

    def formatVectorsForLoss(self, gt, pred):
        pred = np.array(pred)
        pred = pred.reshape(pred.shape[0], 3)
        gt = np.array(gt)
        gt = gt.reshape(gt.shape[0], 3)
        return gt, pred

    def test(self, optim_mode="Adam", net_mode="ResNet8"):

        self.setGPUs()

        # data setup
        test_data, n_samples_test, test_img_data = self.dataLoading(self.data_modes.test)

        # network setup
        self.mdl_test = self.setMdl(net_mode)
        mdl_path = os.path.join(self.config.checkpoint_dir, self.model_name +
                                                '_epoch_{:}'.format(self.config.max_epochs) + "_" + optim_mode)
        self.mdl_test.load_weights(mdl_path + "/")
        self.setOptimizer(optim_mode)
        self.mdl_test.compile(optimizer=self.optim,
                         loss=self.loss,
                         metrics=['accuracy', 'mse'])

        # evaluate
        time_s = time.time()
        results = self.mdl_test.evaluate(test_data)
        total_time = time.time() - time_s
        print(20 * "~~")
        print("TF evaluate took {:}:".format(total_time))
        print("the inference time is per image: {:.6f}".format(total_time / n_samples_test))
        print("the inference fps is: {:.6f}".format(n_samples_test / total_time))
        print(20 * "~~")

        # custom evaluate
        pred_vec = []
        gt_vec = []
        time_s = time.time()
        for img, gt in test_data:
            pred_vec.append(self.mdl_test.call(img).numpy())
            gt_vec.append(gt.numpy())
        total_time = time.time() - time_s
        gt_vec, pred_vec = self.formatVectorsForLoss(gt_vec, pred_vec)
        print(20 * "~~")
        print("Custom evaluate took {:}:".format(total_time))
        print("the inference time is per image: {:.6f}".format(total_time / n_samples_test))
        print("the inference fps is: {:.6f}".format(n_samples_test / total_time))
        print(20 * "~~")

        print(20 * "-", "Done Evaluating", 20 * "-")
        print('Test Loss: {:}\nTest Accuracy: {:}\nTest MSE: {:}'.format(*results))

        # exporting images with annotations
        if self.config.export_test_data:
            # predictions vs gt plot and save on image
            path_eval_imgs = os.path.join(mdl_path, 'EvalImags')
            if not os.path.exists(path_eval_imgs):
                os.mkdir(path_eval_imgs)
            unique_runs = list(set(re.findall("Run_\d+", "".join(test_img_data.images_path))))
            data_dict = {}
            for run in unique_runs:
                if run not in data_dict:
                    data_dict[run] = {"images": [], "gt": [], "paths": []}
                for idx, (img, gt) in enumerate(test_data):
                    if run in test_img_data.images_path[idx]:
                        data_dict[run]["images"].append(img)
                        data_dict[run]["gt"].append(gt)
                        data_dict[run]["paths"].append(test_img_data.images_path[idx])
                    if len(data_dict[run]["images"]) == self.config.num_test_img_save:
                        break
            self.saveTestImagesWithAnnotations(data_dict, path_eval_imgs)
        return

    def testTFLite(self, optim_mode):
        name = self.model_name + "_epoch_{:}".format(self.config.max_epochs) + "_" + optim_mode
        tflite_path = os.path.join(self.config.checkpoint_dir, name + "/", "TFLITE_MODEL", name + ".tflite")
        self.setGPUs()
        # network setup
        self.tf_lite_mdl = tf.lite.Interpreter(model_path=tflite_path)
        self.tf_lite_mdl.allocate_tensors()
        in_details = self.tf_lite_mdl.get_input_details()
        out_details = self.tf_lite_mdl.get_output_details()

        # data setup
        test_data, n_samples_test, test_img_data = self.dataLoading(self.data_modes.test)

        # custom evaluate
        pred_vec = []
        gt_vec = []
        time_s = time.time()
        for img, gt in test_data:
            self.tf_lite_mdl.set_tensor(in_details[0]['index'], img.numpy())
            self.tf_lite_mdl.invoke()
            pred_vec.append(self.tf_lite_mdl.get_tensor(out_details[0]['index']))
            gt_vec.append(gt.numpy())
        total_time = time.time() - time_s
        gt_vec, pred_vec = self.formatVectorsForLoss(gt_vec, pred_vec)

        print("TF-Lite custom evaluate took {:}:".format(total_time))
        print("the inference time is: {:.6f}".format(total_time / n_samples_test))
        print("the inference fps is per image: {:.6f}".format(n_samples_test / total_time))
        print("TFLITE Test Loss: {:}".format(np.mean(self.loss(gt_vec, pred_vec))))

        return

    def initDronesMode(self, optim_mode):
        name = self.model_name + "_epoch_{:}".format(self.config.max_epochs) + "_" + optim_mode

        # set the TF model
        self.drone_mdl = self.setMdl(mode=self.model_name)
        mdl_path = os.path.join(self.config.checkpoint_dir, name)
        self.drone_mdl.load_weights(mdl_path + "/")
        self.setOptimizer(optim_mode)
        self.drone_mdl.compile(optimizer=self.optim,
                              loss=self.loss,
                              metrics=['accuracy', 'mse'])

        # set the TF-Lite model
        tflite_path = os.path.join(self.config.checkpoint_dir, name + "/", "TFLITE_MODEL", name + ".tflite")
        # network setup
        self.tf_lite_drone_mdl = tf.lite.Interpreter(model_path=tflite_path)
        self.tf_lite_drone_mdl.allocate_tensors()
        return

    def droneInference(self, img, tf_lite_flag=False):
        if tf_lite_flag:
            in_details = self.tf_lite_drone_mdl.get_input_details()
            out_details = self.tf_lite_drone_mdl.get_output_details()
            self.tf_lite_drone_mdl.set_tensor(in_details[0]['index'], img.numpy())
            self.tf_lite_drone_mdl.invoke()
            return self.tf_lite_drone_mdl.get_tensor(out_details[0]['index'])
        else:
            return self.drone_mdl.call(img).numpy()
