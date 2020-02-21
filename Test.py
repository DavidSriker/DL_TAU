from LearningPipeline.CommonFlags import *
from LearningPipeline.BaseLearner import *
import sys

optimizer_mode = ["Adam",
                  "SGD",
                  "Adagrad",
                  "Adadelta"]

# Utility main to load flags
try:
    argv = FLAGS(sys.argv)  # parse flags
except gflags.FlagsError:
    print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
    sys.exit(1)

trl = TrajectoryLearner(FLAGS)
if FLAGS.tflite:  # test the TF lite model
    trl.testTFLite(optimizer_mode[0])
else:  # test the regular TF model
    trl.test(optimizer_mode[0], net_mode=FLAGS.net_name)
