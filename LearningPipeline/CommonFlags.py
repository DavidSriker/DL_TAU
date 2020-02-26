import gflags
import os

FLAGS = gflags.FLAGS
rel_path = os.getcwd()

# Directories
gflags.DEFINE_string('train_dir',
                     os.path.join(rel_path, "Data", "Training"),
                     'Folder containing training data')
gflags.DEFINE_string('val_dir', os.path.join(rel_path, "Data", "real_data"),
                     'Folder containing validation data')
gflags.DEFINE_string('test_dir', os.path.join(rel_path, "Data", "test_data"),
                     'Folder containing test data')
gflags.DEFINE_string('checkpoint_dir', os.path.join(rel_path, "LearningPipeline", "Checkpoint"),
                     "Directory name to save checkpoints and logs.")
gflags.DEFINE_integer("max_epochs", 300, "Maximum number of training epochs")

# Train parameters
gflags.DEFINE_integer('img_width', 200, 'Target Image Width')
gflags.DEFINE_integer('img_height', 300, 'Target Image Height')
gflags.DEFINE_integer('batch_size', 128, 'Batch size in training and evaluation')
gflags.DEFINE_float("f", 1.0, "Model Width, float in [0,1]")
gflags.DEFINE_integer('output_dim', 3, "Number of output dimensionality")
gflags.DEFINE_float('gamma', 0.1, "Factor the velocity loss for weighted MSE")
# Train Optimizer Params
gflags.DEFINE_float("lr_adam", 0.000001, "Learning rate of for adam")
gflags.DEFINE_float("lr_sgd", 0.0001, "Learning rate of for sgd")
gflags.DEFINE_float("lr_adagrad", 0.0001, "Learning rate of for adagrad")
gflags.DEFINE_float("lr_adadelta", 0.0001, "Learning rate of for adadelta")
# Network name
gflags.DEFINE_string('net_name', "ResNet8", 'fine tune optimizer')
# gflags.DEFINE_string('net_name', "ResNet8b", 'fine tune optimizer')
# gflags.DEFINE_string('net_name', "TCResNet8", 'fine tune optimizer')

# Testing parameters
gflags.DEFINE_integer('test_img_width', 200, 'Target Image Width')
gflags.DEFINE_integer('test_img_height', 300, 'Target Image Height')
gflags.DEFINE_bool('save_model', True, 'Whether to save the model')
gflags.DEFINE_bool('tflite', True, 'Whether to restore a trained model and test')
gflags.DEFINE_bool('export_test_data', True, 'Whether to export test images with annotations')
gflags.DEFINE_integer('num_test_img_save', 5, 'save only this number of test evaluation images for every Run### folder')
gflags.DEFINE_bool('test_img_save', False, 'Whether to save test evaluation images or not')
