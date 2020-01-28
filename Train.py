from LearningHierarchy.LearningPipeline.CommonFlags import *
from LearningHierarchy.LearningPipeline.BaseLearner import *
import sys

experiment_num = 1


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

if experiment_num == 0:
    trl = TrajectoryLearner(FLAGS)
    trl.train("Adam")
elif experiment_num == 1:
    for opt in optimizer_mode:
        trl = TrajectoryLearner(FLAGS)
        trl.train(opt)
elif experiment_num == 2:
    print("TODO")