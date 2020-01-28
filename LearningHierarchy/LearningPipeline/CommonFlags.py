import gflags
import os

FLAGS = gflags.FLAGS
rel_path = os.getcwd()

# Train parameters
gflags.DEFINE_integer('img_width', 300, 'Target Image Width')
gflags.DEFINE_integer('img_height', 200, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_float("f", 1.0, "Model Width, float in [0,1]")
gflags.DEFINE_integer('output_dim', 3, "Number of output dimensionality")
gflags.DEFINE_float('gamma', 0.1, "Factor the velocity loss for weighted MSE")

# Train Optimizer Params
gflags.DEFINE_float("lr_adam", 0.000001, "Learning rate of for adam")
gflags.DEFINE_float("lr_sgd", 0.0001, "Learning rate of for sgd")
gflags.DEFINE_float("lr_adagrad", 0.0001, "Learning rate of for adagrad")
gflags.DEFINE_float("lr_adadelta", 0.0001, "Learning rate of for adadelta")

# Directories
gflags.DEFINE_string('train_dir',
                     os.path.join(rel_path, "Data", "Datasets", "SimulationTrainingData", "Training"),
                     'Folder containing training experiments')
gflags.DEFINE_string('val_dir', os.path.join(rel_path, "Data", "Datasets", "ValidationRealData", "RealData"),
                     'Folder containing validation experiments')
gflags.DEFINE_string('checkpoint_dir', os.path.join(rel_path, "LearningHierarchy", "LearningPipeline", "Checkpoint"),
                     "Directory name to save checkpoints and logs.")
gflags.DEFINE_string('directory_pb_file', os.path.join(rel_path, "LearningHierarchy", "LearningPipeline", "Checkpoint"),
                     "Directory to the pb saved model file")

# Log parameters
# gflags.DEFINE_integer("max_epochs", 100, "Maximum number of training epochs")
gflags.DEFINE_integer("max_epochs", 2, "Maximum number of training epochs")

gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_integer("summary_freq_iter", 100, "Logging every log_freq iterations")
gflags.DEFINE_integer("summary_freq_epoch", 1, "Logging every log_freq epochs")
# gflags.DEFINE_integer("save_latest_freq", 100, "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
gflags.DEFINE_integer("save_latest_period", 1, "Save the latest model every several epochs"
                                        " (overwrites the previous latest model)")

# Testing parameters
# gflags.DEFINE_string('test_dir', "../../data/validation_sim2real/beauty", 'Folder containing'
#                      ' testing experiments')
# gflags.DEFINE_string('output_dir', "./tests/test_0", 'Folder containing'
#                      ' testing experiments')



gflags.DEFINE_integer('test_img_width', 300, 'Target Image Width')
gflags.DEFINE_integer('test_img_height', 200, 'Target Image Height')

gflags.DEFINE_bool('save_model', True, 'Whether to save the model')
gflags.DEFINE_bool('tflite', True, 'Whether to restore a trained model and test')
gflags.DEFINE_bool('export_test_data', True, 'Whether to export test images with annotations')
gflags.DEFINE_integer('num_test_img_save', 5, 'save only this number of test evaluation images for every Run### folder')
gflags.DEFINE_bool('test_img_save', False, 'Whether to save test evaluation images or not')
gflags.DEFINE_string('net_name', "ResNet8", 'fine tune optimizer')