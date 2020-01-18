import gflags
import os
import glob
import sys

FLAGS = gflags.FLAGS

# rel_path = os.path.normpath(os.getcwd() + os.sep + os.pardir + os.sep + os.pardir)
# rel_path = os.path.normpath(os.getcwd())
rel_path = os.getcwd()
# rel_path = sys.path[2]

# Train parameters
gflags.DEFINE_integer('img_width', 300, 'Target Image Width')
gflags.DEFINE_integer('img_height', 200, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam")
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_float("f", 1.0, "Model Width, float in [0,1]")
gflags.DEFINE_integer('output_dim', 3, "Number of output dimensionality")

gflags.DEFINE_float('gamma', 0.1, "Factor the velocity loss for weighted MSE")

gflags.DEFINE_string('train_dir',
                     os.path.join(rel_path, "Data", "Datasets", "SimulationTrainingData", "Training"),
                     'Folder containing training experiments')
gflags.DEFINE_string('val_dir', os.path.join(rel_path, "Data", "Datasets", "ValidationRealData", "RealData"),
                     'Folder containing validation experiments')
gflags.DEFINE_string('checkpoint_dir', os.path.join(rel_path, "LearningHierarchy", "LearningPipeline", "Checkpoint"),
                     "Directory name to save checkpoints and logs.")

# Log parameters
# gflags.DEFINE_integer("max_epochs", 100, "Maximum number of training epochs")
gflags.DEFINE_integer("max_epochs", 4, "Maximum number of training epochs")

gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
# gflags.DEFINE_integer("save_latest_freq", 100, "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
gflags.DEFINE_integer("save_latest_period", 1, "Save the latest model every several epochs"
                                        " (overwrites the previous latest model)")

# Testing parameters
# gflags.DEFINE_string('test_dir', "../../data/validation_sim2real/beauty", 'Folder containing'
#                      ' testing experiments')
# gflags.DEFINE_string('output_dir', "./tests/test_0", 'Folder containing'
#                      ' testing experiments')

directory_pb_file = os.path.join(rel_path, "LearningHierarchy", "LearningPipeline", "Checkpoint")
# latest_pb_file = max(glob.glob(os.path.join(directory_pb_file, '*')), key=os.path.getmtime)
# # latest_pb_file = max([os.path.join(directory_pb_file, d) for d in os.listdir(directory_pb_file)], key=os.path.getmtime)
# gflags.DEFINE_string("pb_file", os.path.join(latest_pb_file, "saved_model.pb"),
#                      "Checkpoint file")

gflags.DEFINE_integer('test_img_width', 300, 'Target Image Width')
gflags.DEFINE_integer('test_img_height', 200, 'Target Image Height')

gflags.DEFINE_bool('test_phase', False, 'Whether to restore a trained model and test')
