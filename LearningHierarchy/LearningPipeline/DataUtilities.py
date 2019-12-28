import os
import glob
import re
import numpy as np
import cv2 # TODO
from tensorflow.keras.preprocessing.image import Iterator # TODO

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
        return

    def getImagesAndLabels(self, path):
        labels_f = os.path.join(path, 'labels.txt')
        gt_flag = os.path.isfile(labels_f)
        if gt_flag:
            gt = np.loadtxt(labels_f, usecols=(0, 1, 2), delimiter=';')
        else:
            print("No labels file was found at: {}".format(path))

        images_paths = os.path.join(path, '**', '*.{:}'.format(self.files_type_re_s))
        paths_list = glob.glob(images_paths, recursive = True)
        if paths_list:
            paths_splitting = list(map(os.path.split, paths_list))
            paths_splitting = list(zip(*paths_splitting))[1]
            self.images_path.extend(paths_list)
            if gt_flag:
                gt_idx = [int(s) for s in re.findall(r'\d+', ''.join(paths_splitting))]
                self.images_gt.extend(gt[gt_idx])
        return

    class ImagesIterator(Iterator):
        def __init__(self, directory, new_img_dim=(300, 200, 3), shuffle=True,
                        batch_s=32, seed=2020):
            data = DataExtractor(directory)
            num_samples = len(data.images_path)
            super(ImagesIterator, self).__init__(num_samples, batch_s,
                                                    shuffle, seed)

            return

        def imagePreperation(self, img_path):
            img = self.loadImage(img_path)
            if img.shape != self.new_img_dim:
                # Linear interpolation for speed and performance
                img = cv2.resize(img, self.new_img_dim,
                                    interpolation=cv2.INTER_LINEAR)
            return cv2.cvtcolor(img, cv2.COLOR_BGR2RGB)

        def loadImage(self, img_path):
            return cv2.imread(img_path)
