from LearningPipeline.CommonFlags import *
from LearningPipeline.BaseLearner import *
import sys


def transformData(img_path, dims):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return preprocessImage(img, dims)

def preprocessImage(img, dims):
    img = tf.image.resize(img, dims)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.divide(img, 255.0)
    return tf.reshape(img, [1, *img.shape])


optimizer_mode = "Adam"

# Utility main to load flags
try:
    argv = FLAGS(sys.argv)  # parse flags
except gflags.FlagsError:
    print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
    sys.exit(1)

trl = TrajectoryLearner(FLAGS)
trl.initDronesMode(optim_mode=optimizer_mode)

drone_test_folder = os.path.join(os.getcwd(), "DroneTest")
for f in os.listdir(drone_test_folder):
    if f.lower().endswith('.jpg'):
        im = transformData(os.path.join(drone_test_folder, f), (200, 300))
        tf_res = trl.droneInference(im)
        tf_lite_res = trl.droneInference(im, tf_lite_flag=True)
        print("TF Result: {:}".format(tf_res))
        print("TF-Lite Result: {:}".format(tf_lite_res))
