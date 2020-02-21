import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pickle
import numpy as np

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

plt.close('all')
current_dir = os.getcwd()
images_dir = os.path.join(current_dir, "LatexPlots")
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

model_dirs = ["ResNet8_epoch_100_Adadelta", "ResNet8_epoch_100_Adagrad",
                "ResNet8_epoch_100_Adam", "ResNet8_epoch_100_SGD"]

# figure 1 validation rmse
plt.figure(figsize=(3.5, 3.2))
ax = plt.subplot(111)
for mdl in model_dirs:
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    ax.plot(np.sqrt(np.multiply(100,hist_dict['val_mse']))/100)
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.05,
                    box.width * 0.98, box.height * 0.95])
    plt.title('Original Net, Validation RMSE', fontsize=8, fontweight='bold')
    plt.ylabel('RMSE', fontsize=7, fontweight='bold')
    plt.xlabel('# Epoch', fontsize=7, fontweight='bold')

plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend(['Adadelta', 'Adagrad', 'Adam', 'SGD'],
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(images_dir, 'originalNetValiadationRMSE.eps'), format='eps')
plt.close()

# figure 2 validation loss
plt.figure(figsize=(3.5, 3.2))
ax = plt.subplot(111)
for mdl in model_dirs:
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    ax.plot(hist_dict['val_loss'])
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.05,
                    box.width * 0.98, box.height * 0.95])
    plt.title('Original Net, Validation Loss', fontsize=8, fontweight='bold')
    plt.ylabel('Loss', fontsize=7, fontweight='bold')
    plt.xlabel('# Epoch', fontsize=7, fontweight='bold')

plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend(['Adadelta', 'Adagrad', 'Adam', 'SGD'],
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(images_dir, 'originalNetValiadationLoss.eps'), format='eps')
plt.close()

# figure 3 training loss
plt.figure(figsize=(3.5, 3.2))
ax = plt.subplot(111)
for mdl in model_dirs:
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    ax.plot(hist_dict['loss'])
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.05,
                    box.width * 0.98, box.height * 0.95])
    plt.title('Original Net, Training Loss', fontsize=8, fontweight='bold')
    plt.ylabel('Loss', fontsize=7, fontweight='bold')
    plt.xlabel('# Epoch', fontsize=7, fontweight='bold')

plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend(['Adadelta', 'Adagrad', 'Adam', 'SGD'],
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(images_dir, 'originalNetTrainingLoss.eps'), format='eps')
plt.close()


# figure 4 TF TFlite timing per optimizer
width = 0.35  # the width of the bars
fig = plt.figure(figsize=(3.5, 3.2))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(True)

y_TF = [82.787810, 83.851351, 83.652581, 61.451183]
rects_TF = ax.bar(np.arange(4) - width / 2, y_TF, width, color=colors['darkred'])
y_TFLITE = [164.701725, 164.892162, 163.704565, 162.035138]
rects_TFLITE = ax.bar(np.arange(4) + width / 2, y_TFLITE, width, color=colors['dodgerblue'])
box = ax.get_position()
ax.set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.1,
                box.width * 0.95, box.height * 0.9])

ax.set_ylabel('Frames Per Second', fontsize=7, fontweight='bold')
ax.set_xlabel('Optimizer', fontsize=7, fontweight='bold')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(('Adam', 'SGD', 'Adagrad', 'Adadelta'), fontsize=6, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend((rects_TF[0], rects_TFLITE[0]), ('TF', 'TF-Lite'),
            loc='upper center', bbox_to_anchor=(0.5, -0.16),
            fancybox=True, shadow=True, ncol=2, fontsize=7)
plt.title('TensorFlow Version Effect\nFrames Per Second Vs. Optimizer', fontsize=8, fontweight='bold')
# plt.show()
plt.savefig(os.path.join(images_dir, 'FPSComparisonTF_TFLite.eps'), format='eps')
plt.close()


# figures 5 - 12 per optimizer the loss and rmse of train and validation

for mdl in model_dirs:
    mdl_name = mdl.split("_")[-1]
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    for t in ["loss", "rmse"]:
        plt.figure(figsize=(3.5, 3.2))
        ax = plt.subplot(111)
        if t == "loss":
            ax.plot(hist_dict['loss'])
            ax.plot(hist_dict['val_loss'])
        elif t == "rmse":
            ax.plot(np.sqrt(np.multiply(100,hist_dict['mse']))/100)
            ax.plot(np.sqrt(np.multiply(100,hist_dict['val_mse']))/100)

        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.06, box.y0 + box.height * 0.1,
                        box.width * 0.94, box.height * 0.9])
        if t == "loss":
            plt.title('{:}, Loss Vs. Epochs'.format(mdl_name), fontsize=8, fontweight='bold')
            plt.ylabel('Loss', fontsize=7, fontweight='bold')
            plt.xlabel('# Epoch', fontsize=7, fontweight='bold')
            plt.legend(['Training', 'Validation'],
                        loc='upper center', bbox_to_anchor=(0.5, -0.16),
                        fancybox=True, shadow=True, ncol=2, fontsize=7)
        elif t == "rmse":
            plt.title('{:}, RMSE Vs. Epochs'.format(mdl_name), fontsize=8, fontweight='bold')
            plt.ylabel('RMSE', fontsize=7, fontweight='bold')
            plt.xlabel('# Epoch', fontsize=7, fontweight='bold')
            plt.legend(['Training', 'Validation'],
                        loc='upper center', bbox_to_anchor=(0.5, -0.16),
                        fancybox=True, shadow=True, ncol=2, fontsize=7)

        plt.xticks(fontsize=8, fontweight='bold')
        plt.yticks(fontsize=8, fontweight='bold')
        plt.grid(True)
        # plt.show()
        plt.savefig(os.path.join(images_dir, '{:}_{:}.eps'.format(mdl_name, t)), format='eps')
        plt.close()


# figures 13 gammas experiments
gammas = {0.0001: [], 0.001: [], 0.01: [], 0.1: []}
gammas_name = ["0_0001", "0_001", "0_01", "0_1"]
exp_d = ["ResNet8_epoch_20_Adam", "ResNet8_epoch_40_Adam",
        "ResNet8_epoch_60_Adam", "ResNet8_epoch_80_Adam",
        "ResNet8_epoch_100_Adam"]
for n, g in zip(gammas_name, gammas.keys()):
    path = os.path.join(current_dir, "PlotsData", "gamma_exp_g_{:}".format(n))
    for d in exp_d:
        exp_path = os.path.join(path, d, "Hisotry")
        hist_dict = pickle.load(open(os.path.join(exp_path, "trainHistoryDict"), "rb"))
        rmse = np.sqrt(hist_dict['val_mse'])
        gammas[g].append(rmse[-1])

width = 0.2  # the width of the bars
fig = plt.figure(figsize=(3.5, 3.2))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(True)

rect_g_0_0001 = ax.bar(np.arange(5) - width * 1.5, gammas[0.0001], width, color=colors['darkred'])
rect_g_0_001 = ax.bar(np.arange(5) - width / 2, gammas[0.001], width, color=colors['dodgerblue'])
rect_g_0_01 = ax.bar(np.arange(5) + width / 2, gammas[0.01], width, color=colors['limegreen'])
rect_g_0_1 = ax.bar(np.arange(5) + width * 1.5, gammas[0.1], width, color=colors['rebeccapurple'])
box = ax.get_position()
ax.set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.12,
                box.width * 0.95, box.height * 0.88])

plt.title('${\gamma}$ Effect\nRMSE Vs. Epoch', fontsize=8, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=7, fontweight='bold')
ax.set_xlabel('# Epochs', fontsize=7, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=10)
ax.set_xticks(np.arange(5))
ax.set_xticklabels(('20', '40', '60', '80', '100'), fontsize=6, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend((rect_g_0_0001[0], rect_g_0_001[0], rect_g_0_01[0], rect_g_0_1[0]),
            ('${\gamma}$=1E-4', '${\gamma}$=1E-3', '${\gamma}$=1E-2', '${\gamma}$=1E-1'),
            loc='upper center', bbox_to_anchor=(0.45, -0.16),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
# plt.show()
plt.savefig(os.path.join(images_dir, 'gamma_exp.eps'), format='eps')
plt.close()

##
model_dirs = ["ResNet8_epoch_200_Adam", "ResNet8b_epoch_200_Adam", "TCResNet8_epoch_200_Adam"]
##
for mdl in model_dirs:
    mdl_name = mdl.split("_")[-1]
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    for t in ["loss", "rmse"]:
        plt.figure(figsize=(3.5, 3.2))
        ax = plt.subplot(111)
        if t == "loss":
            ax.plot(hist_dict['loss'])
            ax.plot(hist_dict['val_loss'])
        elif t == "rmse":
            ax.plot(np.sqrt(np.multiply(100,hist_dict['mse']))/100)
            ax.plot(np.sqrt(np.multiply(100,hist_dict['val_mse']))/100)

        box = ax.get_position()
        ax.set_position([box.x0 + box.width * 0.06, box.y0 + box.height * 0.1,
                        box.width * 0.94, box.height * 0.9])
        if t == "loss":
            plt.title('{:}, Loss Vs. Epochs'.format(mdl_name), fontsize=8, fontweight='bold')
            plt.ylabel('Loss', fontsize=7, fontweight='bold')
            plt.xlabel('# Epoch', fontsize=7, fontweight='bold')
            plt.legend(['Training', 'Validation'],
                        loc='upper center', bbox_to_anchor=(0.5, -0.16),
                        fancybox=True, shadow=True, ncol=2, fontsize=7)
        elif t == "rmse":
            plt.title('{:}, RMSE Vs. Epochs'.format(mdl_name), fontsize=8, fontweight='bold')
            plt.ylabel('RMSE', fontsize=7, fontweight='bold')
            plt.xlabel('# Epoch', fontsize=7, fontweight='bold')
            plt.legend(['Training', 'Validation'],
                        loc='upper center', bbox_to_anchor=(0.5, -0.16),
                        fancybox=True, shadow=True, ncol=2, fontsize=7)

        plt.xticks(fontsize=8, fontweight='bold')
        plt.yticks(fontsize=8, fontweight='bold')
        plt.grid(True)
        plt.savefig(os.path.join(images_dir, '{:}_{:}.eps'.format(mdl, t)), format='eps')
        plt.close()

# figure 15 validation rmse for different nets
plt.figure(figsize=(3.5, 3.2))
ax = plt.subplot(111)
for mdl in model_dirs_2:
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    ax.plot(np.sqrt(np.multiply(100,hist_dict['val_mse']))/100)
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.05,
                    box.width * 0.98, box.height * 0.95])
    plt.title('Different Nets, Validation RMSE', fontsize=8, fontweight='bold')
    plt.ylabel('RMSE', fontsize=7, fontweight='bold')
    plt.xlabel('# Epoch', fontsize=7, fontweight='bold')

plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend(['ResNet8', 'ResNet7', 'TC-ResNet8'],
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
plt.grid(True)
plt.savefig(os.path.join(images_dir, 'DifferentNetsValiadationRMSE.eps'), format='eps')
plt.close()

# figure 16 validation loss for different nets
plt.figure(figsize=(3.5, 3.2))
ax = plt.subplot(111)
for mdl in model_dirs_2:
    path = os.path.join(current_dir, "PlotsData", mdl, "Hisotry")
    hist_dict = pickle.load(open(os.path.join(path, "trainHistoryDict"), "rb"))

    ax.plot(hist_dict['val_loss'])
    box = ax.get_position()
    ax.set_position([box.x0 + box.width * 0.02, box.y0 + box.height * 0.05,
                    box.width * 0.98, box.height * 0.95])
    plt.title('Different Nets, Validation Loss', fontsize=8, fontweight='bold')
    plt.ylabel('Loss', fontsize=7, fontweight='bold')
    plt.xlabel('# Epoch', fontsize=7, fontweight='bold')

plt.xticks(fontsize=8, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend(['ResNet8', 'ResNet7', 'TC-ResNet8'],
            loc='upper center', bbox_to_anchor=(0.5, -0.2),
            fancybox=True, shadow=True, ncol=4, fontsize=7)
plt.grid(True)
# plt.show()
plt.savefig(os.path.join(images_dir, 'DifferentNetsValiadationLoss.eps'), format='eps')
plt.close()

# figure 17 TF TFlite timing per different nets
width = 0.35  # the width of the bars
fig = plt.figure(figsize=(3.5, 3.2))
ax = fig.add_subplot(111)
ax.set_axisbelow(True)
ax.grid(True)

y_TF = [81.896787, 108.975292, 87.859126]
rects_TF = ax.bar(np.arange(3) - width / 2, y_TF, width, color=colors['darkred'])
y_TFLITE = [164.634252, 205.332021, 340.987400]
rects_TFLITE = ax.bar(np.arange(3) + width / 2, y_TFLITE, width, color=colors['dodgerblue'])
box = ax.get_position()
ax.set_position([box.x0 + box.width * 0.05, box.y0 + box.height * 0.1,
                box.width * 0.95, box.height * 0.9])

ax.set_ylabel('Frames Per Second', fontsize=7, fontweight='bold')
ax.set_xlabel('Model', fontsize=7, fontweight='bold')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(('ResNet8', 'ResNet7', 'TC-ResNet8'), fontsize=6, fontweight='bold')
plt.yticks(fontsize=8, fontweight='bold')
plt.legend((rects_TF[0], rects_TFLITE[0]), ('TF', 'TF-Lite'),
            loc='upper center', bbox_to_anchor=(0.5, -0.16),
            fancybox=True, shadow=True, ncol=2, fontsize=7)
plt.title('TensorFlow Version Effect\nFrames Per Second Vs. Different Networks', fontsize=8, fontweight='bold')
# plt.show()
plt.savefig(os.path.join(images_dir, 'DifferentNetsFPSComparisonTF_TFLite.eps'), format='eps')
plt.savefig(os.path.join(images_dir, 'DifferentNetsFPSComparisonTF_TFLite.png'), format='png')
plt.close()
