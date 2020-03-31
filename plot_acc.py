import matplotlib.pyplot as plt
import csv

FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/gen_segnet-2020-03-30-12:21:46.457167/model_history_log.csv'

history_log_file = open(FILEPATH, 'r')
history = csv.reader(history_log_file)

# log files contain one row of epoch,accuracy,loss,val_accuracy,val_loss per epoch
epoch = []
accuracy = []
loss = []
val_accuracy = []
val_loss = []
with open(FILEPATH,'r') as csvfile:
    plots = csv.DictReader(csvfile)
    for row in plots:
        epoch.append(int(row['epoch']))
        accuracy.append(float(row['accuracy']))
        loss.append(float(row['loss']))
        val_accuracy.append(float(row['val_accuracy']))
        val_loss.append(float(row['val_loss']))

plt.figure()

plt.subplot(211)
plt.title('Model Accuracy')
plt.plot(epoch, accuracy, marker='o')
plt.plot(epoch, val_accuracy, marker='o')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right')

plt.subplot(212)
plt.title('Model Loss')
plt.plot(epoch, loss, marker='o')
plt.plot(epoch, val_loss, marker='o')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')

plt.show()

# eval output is [test_loss, test_accuracy]? so both can be compared to graphs
# example: Metrics at 0 are [4.6877665519714355, 0.058624133467674255] (from job.185169.out)

