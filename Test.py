from LearningHierarchy.LearningPipeline.CommonFlags import *
from LearningHierarchy.LearningPipeline.BaseLearner import *

# Utility main to load flags
try:
    argv = FLAGS(sys.argv)  # parse flags
except gflags.FlagsError:
    print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
    sys.exit(1)

trl = TrajectoryLearner(FLAGS)
if FLAGS.tflite:
    trl.testTFLite(FLAGS.directory_pb_file)
else:
    trl.test()