import matplotlib.pyplot as plt
import csv

# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/gen_segnet-2020-03-30-12:21:46.457167/model_history_log.csv'

# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-01/disc_segnet-2020-04-01-10:12:20.789132/model_history_log.csv'
# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-01/disc_segnet-2020-04-01-13:05:39.319831/model_history_log.csv'
# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-01/gan_segnet-2020-04-01-11:29:06.463303/model_history_log.csv'
# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-01/gan_segnet-2020-04-01-14:49:29.158507/model_history_log.csv'

FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-05/disc_stacked_segnet-2020-04-05-18:41:12.352539/model_history_log.csv'
# FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-05/gan_stacked_segnet-2020-04-05-18:41:12.353349/model_history_log.csv'

# history_log_file = open(FILEPATH, 'r')
# history = csv.reader(history_log_file)

# log files contain one row of epoch,accuracy,loss,val_accuracy,val_loss per epoch
# epoch,accuracy,auc_1,loss,sensitivity_at_specificity,specificity_at_sensitivity,val_accuracy,val_auc_1,val_loss,val_sensitivity_at_specificity,val_specificity_at_sensitivity
epoch = []
accuracy = []
auc_1 = []
auc_2 = []
loss = []
real_acc = []  # sensitivity_at_specificity
fake_acc = []  # specificity_at_sensitivity

val_accuracy = []
val_auc_1 = []
val_auc_2 = []
val_loss = []
val_real_acc = []  # sensitivity_at_specificity
val_fake_acc = []  # specificity_at_sensitivity

cum_epochs = []
discrim = True
gan = False
if discrim:
    with open(FILEPATH,'r') as csvfile:
        plots = csv.DictReader(csvfile)
        i = 0
        for row in plots:
            epoch.append(int(row['epoch']))
            accuracy.append(float(row['accuracy']))
            auc_1.append(float(row['auc_1']))
            loss.append(float(row['loss']))
            real_acc.append(float(row['sensitivity_at_specificity']))  # sensitivity_at_specificity
            fake_acc.append(float(row['specificity_at_sensitivity']))  # specificity_at_sensitivity
            val_accuracy.append(float(row['val_accuracy']))
            val_auc_1.append(float(row['val_auc_1']))
            val_loss.append(float(row['val_loss']))
            val_real_acc.append(float(row['val_sensitivity_at_specificity']))  # sensitivity_at_specificity
            val_fake_acc.append(float(row['val_specificity_at_sensitivity']))  # specificity_at_sensitivity
            cum_epochs.append(i)
            i += 1

    plt.figure()

    plt.subplot(211)
    plt.title('Discriminator Accuracy')
    plt.plot(cum_epochs, epoch, marker='o')
    plt.plot(cum_epochs, accuracy, marker='o')
    plt.plot(cum_epochs, val_accuracy, marker='o')
    plt.plot(cum_epochs, auc_1, marker='o')
    plt.plot(cum_epochs, real_acc, marker='o')
    plt.plot(cum_epochs, fake_acc, marker='o')
    plt.plot(cum_epochs, val_auc_1, marker='o')
    plt.plot(cum_epochs, val_real_acc, marker='o')
    plt.plot(cum_epochs, val_fake_acc, marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['epoch', 'Train Accuracy', 'Validation Accuracy', 'auc', 'real_acc', 'fake_acc', 'val_auc_1', 'val_real_acc', 'val_fake_acc'], loc='lower right')

    ax2 = plt.subplot(212)
    plt.title('Discriminator Loss')
    plt.plot(cum_epochs, loss, marker='o')
    plt.plot(cum_epochs, val_loss, marker='o')
    plt.ylabel('Loss')
    ax2.set_ylim([0, 5])
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')

if gan:
    FILEPATH = '/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/checkpoints/run-04-05/gan_stacked_segnet-2020-04-05-18:41:12.353349/model_history_log.csv'
    # epoch,accuracy,auc_2,loss,val_accuracy,val_auc_2,val_loss
    with open(FILEPATH, 'r') as csvfile:
        plots = csv.DictReader(csvfile)
        i = 0
        for row in plots:
            epoch.append(int(row['epoch']))
            accuracy.append(float(row['accuracy']))
            auc_2.append(float(row['auc_2']))
            loss.append(float(row['loss']))
            # real_acc.append(float(row['sensitivity_at_specificity']))  # sensitivity_at_specificity
            # fake_acc.append(float(row['specificity_at_sensitivity']))  # specificity_at_sensitivity
            val_accuracy.append(float(row['val_accuracy']))
            val_auc_2.append(float(row['val_auc_2']))
            val_loss.append(float(row['val_loss']))
            # val_real_acc.append(float(row['val_sensitivity_at_specificity']))  # sensitivity_at_specificity
            # val_fake_acc.append(float(row['val_specificity_at_sensitivity']))  # specificity_at_sensitivity
            cum_epochs.append(i)
            i += 1

    plt.figure()

    plt.subplot(211)
    plt.title('GAN Accuracy')
    plt.plot(cum_epochs, epoch, marker='o')
    plt.plot(cum_epochs, accuracy, marker='o')
    plt.plot(cum_epochs, val_accuracy, marker='o')
    plt.plot(cum_epochs, auc_2, marker='o')
    # plt.plot(cum_epochs, real_acc, marker='o')
    # plt.plot(cum_epochs, fake_acc, marker='o')
    plt.plot(cum_epochs, val_auc_2, marker='o')
    # plt.plot(cum_epochs, val_real_acc, marker='o')
    # plt.plot(cum_epochs, val_fake_acc, marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['epoch', 'Train Accuracy', 'Validation Accuracy', 'auc', 'val_auc'], loc='lower right')

    ax2 = plt.subplot(212)
    plt.title('GAN Loss')
    plt.plot(cum_epochs, loss, marker='o')
    plt.plot(cum_epochs, val_loss, marker='o')
    plt.ylabel('Loss')
    ax2.set_ylim([0, 300])
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')

plt.show()


# eval output is [test_loss, test_accuracy]? so both can be compared to graphs
# example: Metrics at 0 are [4.6877665519714355, 0.058624133467674255] (from job.185169.out)

