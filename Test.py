from LearningHierarchy.LearningPipeline.CommonFlags import *
from LearningHierarchy.LearningPipeline.BaseLearner import *
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
if FLAGS.tflite:
    trl.testTFLite(optimizer_mode[2])
else:
    trl.test(optimizer_mode[3], net_mode=FLAGS.net_name)