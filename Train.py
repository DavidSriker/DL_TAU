from LearningHierarchy.LearningPipeline.CommonFlags import *
from LearningHierarchy.LearningPipeline.BaseLearner import *
import sys

experiment_num = 2


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
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    gammas_str = ["0_0001", "0_001", "0_01", "0_1", "1", "10"]
    num_epochs = [20, 40, 60, 80, 100]
    for idx, g in enumerate(gammas):
        FLAGS.gamma=g
        FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, "gamma_exp_g_{:}".format(gammas_str[idx]))
        for e in num_epochs:
            FLAGS.max_epochs = e
            trl = TrajectoryLearner(FLAGS)
            trl.train()
