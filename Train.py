from LearningPipeline.CommonFlags import *
from LearningPipeline.BaseLearner import *
import sys

"""
Train main function. Choose experiment and optimizer to begin training.
experiment_num = 0: naive training of single module with specified optimizer.
experiment_num = 1: model choise is given by predefined FLAGS. training is done over all optimizer_mode options
experiment_num = 2: gamma experiment is set for list of epochs. the default optimizer (Adam) is used.
experiment_num = 3: run all stated models with default optimizer and with gamma=0.1
"""


experiment_num = 3


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
    trl.train(optim_mode="Adam", net_mode="ResNet8") #net_mode=FLAGS.net_name)
elif experiment_num == 1:
    for opt in optimizer_mode:
        trl = TrajectoryLearner(FLAGS)
        trl.train(optim_mode=opt, net_mode=FLAGS.net_name)
elif experiment_num == 2:
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    gammas_str = ["0_0001", "0_001", "0_01", "0_1", "1", "10"]
    num_epochs = [20, 40, 60, 80, 100]
    ckpt_dir = FLAGS.checkpoint_dir
    for idx, g in enumerate(gammas):
        FLAGS.gamma = g
        FLAGS.checkpoint_dir = os.path.join(ckpt_dir, "gamma_exp_g_{:}".format(gammas_str[idx]))
        for e in num_epochs:
            FLAGS.max_epochs = e
            trl = TrajectoryLearner(FLAGS)
            trl.train()
elif experiment_num == 3:
    model_names = ["TCResNet8", "ResNet8b", "ResNet8"]
    for mdl in model_names:
        FLAGS.net_name = mdl
        trl = TrajectoryLearner(FLAGS)
        trl.train(net_mode=mdl)
