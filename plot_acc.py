import matplotlib.pyplot as plt
import csv

FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/segnet_disc-2020-03-2016:43:32.238296/model_history_log.csv'

history_log_file = open(FILEPATH, 'r')
history = csv.reader(history_log_file)

# log files contain one row of epoch,accuracy,loss,val_accuracy,val_loss per epoch
epoch = []
accuracy = []
loss = []
val_accuracy = []
val_loss = []
with open(FILEPATH,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        epoch.append(row[0])
        accuracy.append(row[1])
        val_accuracy.append(row[3])

plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
