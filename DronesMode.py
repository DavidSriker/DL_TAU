from LearningPipeline.CommonFlags import *
from LearningPipeline.BaseLearner import *
import sys

optimizer_mode = "Adam"

# Utility main to load flags
try:
    argv = FLAGS(sys.argv)  # parse flags
except gflags.FlagsError:
    print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
    sys.exit(1)

trl = TrajectoryLearner(FLAGS)
trl.initDronesMode(optim_mode=optimizer_mode)

# TODO - add the loop that calls the DroneInference with single tensor
