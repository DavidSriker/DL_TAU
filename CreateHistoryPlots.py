import os
import matplotlib.pyplot as plt
import pickle

plt.close('all')
# optimizer_name = ['_SGD', '_Adam', '_Adagrad', '_Adadelta']
# parent_path = os.path.join('LearningHierarchy', 'LearningPipeline', 'Checkpoint')
# net_name = 'ResNet8_'
# epoch = 'epoch_100'
# optimizer_name = '_SGD'
# model_name = net_name+epoch+optimizer_name
# # path_load = os.path.join(os.getcwd(), parent_path, net_name+epoch+optimizer_name, 'Hisorty', 'trainHistoryDict')
# path_load = r'D:\Users\lidor\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\Checkpoint\ResNet8_epoch_100_SGD\Hisotry\trainHistoryDict'
# history = pickle.load(open(path_load, "rb"))
# # list all data in history
# print(history.keys())
#
#
# # summarize history for mse
# plt.figure()
# plt.plot(history['mse'])
# plt.plot(history['val_mse'])
# plt.title(model_name + ' - mse')
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # summarize history for accuracy
#
# plt.figure()
# plt.plot(history['accuracy'])
# plt.plot(history['val_accuracy'])
# plt.title(model_name + ' - accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# # summarize history for loss
# plt.figure()
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title(model_name + ' - loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# print('done')

path_load_adadelta = r'D:\Users\lidor\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\Checkpoint\ResNet8_epoch_100_Adadelta\Hisotry\trainHistoryDict'
history_adadelta = pickle.load(open(path_load_adadelta, "rb"))
path_load_adagrad = r'D:\Users\lidor\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\Checkpoint\ResNet8_epoch_100_Adagrad\Hisotry\trainHistoryDict'
history_adagrad = pickle.load(open(path_load_adagrad, "rb"))
path_load_adam = r'D:\Users\lidor\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\Checkpoint\ResNet8_epoch_100_Adam\Hisotry\trainHistoryDict'
history_adam = pickle.load(open(path_load_adam, "rb"))
path_load_sgd = r'D:\Users\lidor\PycharmProjects\deepdroneracing\Deep_Learning_TAU\LearningHierarchy\LearningPipeline\Checkpoint\ResNet8_epoch_100_SGD\Hisotry\trainHistoryDict'
history_sgd = pickle.load(open(path_load_sgd, "rb"))

net_name = 'ResNet8_'
epoch = 'epoch_100'

# Adadelta:
optimizer_name = '_Adadelta'
model_name = net_name+epoch+optimizer_name
# summarize history for mse
plt.figure()
plt.plot(history_adadelta['mse'])
plt.plot(history_adadelta['val_mse'])
plt.title(model_name + ' - mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for accuracy
plt.figure()
plt.plot(history_adadelta['accuracy'])
plt.plot(history_adadelta['val_accuracy'])
plt.title(model_name + ' - accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history_adadelta['loss'])
plt.plot(history_adadelta['val_loss'])
plt.title(model_name + ' - loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# Adagrad:
optimizer_name = '_Adagrad'
model_name = net_name+epoch+optimizer_name
# summarize history for mse
plt.figure()
plt.plot(history_adagrad['mse'])
plt.plot(history_adagrad['val_mse'])
plt.title(model_name + ' - mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for accuracy
plt.figure()
plt.plot(history_adagrad['accuracy'])
plt.plot(history_adagrad['val_accuracy'])
plt.title(model_name + ' - accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history_adagrad['loss'])
plt.plot(history_adagrad['val_loss'])
plt.title(model_name + ' - loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# Adam:
optimizer_name = '_Adam'
model_name = net_name+epoch+optimizer_name
# summarize history for mse
plt.figure()
plt.plot(history_adam['mse'])
plt.plot(history_adam['val_mse'])
plt.title(model_name + ' - mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for accuracy
plt.figure()
plt.plot(history_adam['accuracy'])
plt.plot(history_adam['val_accuracy'])
plt.title(model_name + ' - accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history_adam['loss'])
plt.plot(history_adam['val_loss'])
plt.title(model_name + ' - loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# SGD:
optimizer_name = '_SGD'
model_name = net_name+epoch+optimizer_name
# summarize history for mse
plt.figure()
plt.plot(history_sgd['mse'])
plt.plot(history_sgd['val_mse'])
plt.title(model_name + ' - mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for accuracy
plt.figure()
plt.plot(history_sgd['accuracy'])
plt.plot(history_sgd['val_accuracy'])
plt.title(model_name + ' - accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history_sgd['loss'])
plt.plot(history_sgd['val_loss'])
plt.title(model_name + ' - loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()



# summarize history for validation mse
plt.figure()
plt.plot(history_adadelta['val_mse'])
plt.plot(history_adagrad['val_mse'])
plt.plot(history_adam['val_mse'])
plt.plot(history_sgd['val_mse'])
plt.title('ResNet8 - validation - mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['val_adadelta', 'val_adagrad', 'val_adam', 'val_sgd'], loc='upper left')
plt.grid(True)
plt.show()

# summarize history for validation accuracy
plt.figure()
plt.plot(history_adadelta['val_accuracy'])
plt.plot(history_adagrad['val_accuracy'])
plt.plot(history_adam['val_accuracy'])
plt.plot(history_sgd['val_accuracy'])
plt.title('ResNet8 - validation - accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val_adadelta', 'val_adagrad', 'val_adam', 'val_sgd'], loc='upper left')
plt.grid(True)
plt.show()

# summarize history for validation loss
plt.figure()
plt.plot(history_adadelta['val_loss'])
plt.plot(history_adagrad['val_loss'])
plt.plot(history_adam['val_loss'])
plt.plot(history_sgd['val_loss'])
plt.title('ResNet8 - validation - loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val_adadelta', 'val_adagrad', 'val_adam', 'val_sgd'], loc='upper left')
plt.grid(True)
plt.show()
print('-'*20 + 'done' + '-'*20)




# print last value
print(history_adadelta['mse'][99])
print(history_adadelta['accuracy'][99])
print(history_adadelta['loss'][99])
print(history_adagrad['mse'][99])
print(history_adagrad['accuracy'][99])
print(history_adagrad['loss'][99])
print(history_adam['mse'][99])
print(history_adam['accuracy'][99])
print(history_adam['loss'][99])
print(history_sgd['mse'][99])
print(history_sgd['accuracy'][99])
print(history_sgd['loss'][99])

