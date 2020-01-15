import os
import glob
import re
import numpy as np
# import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import Iterator


def manyExtensionCreate(extenstions):
    """
    This function gets extension of files and return
    a regex str in order to enter glob.glob function
    Example:
    ['tif', 'jpg'] -> [tj][ip][fg]*
    """
    s = zip(*[list(ft) for ft in extenstions])
    s = list(map('[{0}]'.format, list(map(''.join, s))))
    return ''.join(s) + '*'


class DataExtractor():
    """
    This class will extract all the desired images paths and labels
    from the given directory and all its sub-directories.
    Its assume there is an a certain structure to the files in it.
    """
    files_type = ['jpg', 'png']
    files_type_re_s = manyExtensionCreate(files_type)

    def __init__(self, directory):
        self.images_path = []
        self.images_gt = []
        for sub_dir in sorted(os.listdir(directory)):
            sub_dir_option = os.path.join(directory, sub_dir)
            if os.path.isdir(sub_dir_option):
                self.getImagesAndLabels(sub_dir_option)
        self.num_samples = len(self.images_path)
        return

    def getImagesAndLabels(self, path):
        labels_f = os.path.join(path, 'labels.txt')
        gt_flag = os.path.isfile(labels_f)
        if gt_flag:
            gt = np.loadtxt(labels_f, usecols=(0, 1, 2), delimiter=';')
        else:
            print("No labels file was found at: {}".format(path))

        images_paths = os.path.join(path, '**', '*.{:}'.format(self.files_type_re_s))
        paths_list = glob.glob(images_paths, recursive=True)
        if paths_list:
            paths_splitting = list(map(os.path.split, paths_list))
            paths_splitting = list(zip(*paths_splitting))[1]
            self.images_path.extend(paths_list)
            if gt_flag:
                gt_idx = [int(s) for s in re.findall(r'\d+', ''.join(paths_splitting))]
                self.images_gt.extend(gt[gt_idx])
        return


class ImagesIterator(Iterator):
    def __init__(self, directory, new_img_dim=(300, 200, 3), shuffle=True, batch_s=32, seed=2020):
        self.data = DataExtractor(directory)
        self.num_samples = self.data.num_samples
        self.batch_s = batch_s
        self.img_dims = (new_img_dim[0], new_img_dim[1])
        super(ImagesIterator, self).__init__(self.data.num_samples, batch_s, shuffle, seed)
        return

    # not in use for the moment
    # def imagePreperation(self, img_path):
    #     img = self.loadImage(img_path)
    #     if img.shape != self.new_img_dim:
    #         # Linear interpolation for speed and performance
    #         img = cv2.resize(img, self.new_img_dim, interpolation=cv2.INTER_LINEAR)
    #     return cv2.cvtcolor(img, cv2.COLOR_BGR2RGB)
    #
    # def loadImage(self, img_path):
    #     return cv2.imread(img_path)

    def generateBatches(self, validation=False):
        seed = np.random.randint(0, 2 ** 31 - 1)
        inputs_queue = tf.data.Dataset.from_tensor_slices((self.data.images_path,
                                                    self.data.images_gt)).shuffle(not validation,
                                                                                    seed=seed)
        if not validation:
            transformed_data_iter = inputs_queue.map(self.transformData).batch(self.batch_s)
        else:
            transformed_data_iter = inputs_queue.map(self.transformData).batch(1)
        return transformed_data_iter

    def transformData(self, img_path, gt):
        gt = tf.cast(gt, dtype=tf.float32)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = self.preprocessImage(img)
        return img, gt

    def preprocessImage(self, img):
        img = tf.image.resize(img, self.img_dims)
        img = tf.cast(img, dtype=tf.float32)
        img = tf.divide(img, 255.0)
        return img
