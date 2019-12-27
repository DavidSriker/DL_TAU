import os
import glob
import re
import numpy as np

def manyExtensionCreate(extenstions):
    """
    This function gets extension of files and return
    a regex str in order to enter glob.glob function

    example:
    ['tif', 'jpg'] -> [tj][ip][fg]*
    """
    s = zip(*[list(ft) for ft in extenstions])
    s = list(map('[{0}]'.format, list(map(''.join, s))))
    return ''.join(s) + '*'

class DataExtractor():
    files_type = ['jpg', 'png']
    files_type_re_s = manyExtensionCreate(files_type)
    def __init__(self, directory, new_img_dim=(224, 224, 3),
                    batch_s=32, seed=2020):

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
            gt = np.loadtxt(labels_f, cols=(0, 1, 2), delimitert=';')
        else:
            print("No labels file was found at: {}".format(path))

        img_path = os.path.join(path, '**', '*.{:}'.format(files_type_re_s))
        for img in glob.glob(img_path, recursive = True):
            h, t = os.path.split(img)
            file_idx = re.search('\d+', t).group()
            self.images_path.append(img)
            self.images_gt.append(gt[file_idx])
        return
