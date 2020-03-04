from keras_segmentation.pretrained import pspnet_101_cityscapes
from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.models.unet import unet_mini
from keras_segmentation.models.fcn import fcn_8
import tensorflow as tf
print("tensorflow version is ", tf.__version__)

data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
# data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/dataset1/"
# data_path = "/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/cityscape/prepped/"
print("data path is ", data_path)

# pret_model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset
# print("pret_model")
# # evaluating the pretrained model
# print(pret_model.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
#                                        annotations_dir=data_path + "annotations_prepped_test/"))


fcn8 = fcn_8(20, input_height=713, input_width=713)  # change to vgg_unet?  # n_classes changed from 19 to 20  # what is size of gtFine images?
print("model beginning training is ", fcn8.name)

fcn8.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    input_height=None,
    input_width=None,
    n_classes=None,
    verify_dataset=True,
    checkpoints_path="./checkpoints/fcn8",
    epochs=5,  # doesn't do anything now
    batch_size=4,  # default 2
    validate=True,
    val_images=data_path + "images_prepped_val",
    val_annotations=data_path + "annotations_prepped_val",
    val_batch_size=4,  # default 2
    auto_resume_checkpoint=False,
    load_weights=None,
    steps_per_epoch=512,
    val_steps_per_epoch=512,
    gen_use_multiprocessing=True,  # default False
    optimizer_name='adadelta',
    do_augment=False,
    history_csv="./checkpoints/fcn8/model_history_log.csv"
)

print("Evaluating ", fcn8.name)
# evaluating the model
print(fcn8.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                 annotations_dir=data_path + "annotations_prepped_test/"))


# psp_101 produces OOM error when training
psp_gtfine = pspnet(20, input_height=713, input_width=713)  # change to vgg_unet?  # n_classes changed from 19 to 20
print("model beginning training is ", psp_gtfine.name)

psp_gtfine.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    input_height=None,
    input_width=None,
    n_classes=None,
    verify_dataset=True,
    checkpoints_path="./checkpoints/psp_gtfine",
    epochs=5,  # doesn't do anything now
    batch_size=4,  # default 2
    validate=True,
    val_images=data_path + "images_prepped_val",
    val_annotations=data_path + "annotations_prepped_val",
    val_batch_size=4,  # default 2
    auto_resume_checkpoint=False,
    load_weights=None,
    steps_per_epoch=512,
    val_steps_per_epoch=512,
    gen_use_multiprocessing=True,  # default False
    optimizer_name='adadelta',
    do_augment=False,
    history_csv="./checkpoints/psp_gtfine/model_history_log.csv"
)

print("Evaluating ", psp_gtfine.name)
# evaluating the model
print(psp_gtfine.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                       annotations_dir=data_path + "annotations_prepped_test/"))


unet = unet_mini(20, input_height=713, input_width=713)  # change to vgg_unet?  # n_classes changed from 19 to 20
print("model beginning training is ", unet.name)

unet.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    input_height=None,
    input_width=None,
    n_classes=None,
    verify_dataset=True,
    checkpoints_path="./checkpoints/unet",
    epochs=5,  # doesn't do anything now
    batch_size=4,  # default 2
    validate=True,
    val_images=data_path + "images_prepped_val",
    val_annotations=data_path + "annotations_prepped_val",
    val_batch_size=4,  # default 2
    auto_resume_checkpoint=False,
    load_weights=None,
    steps_per_epoch=512,
    val_steps_per_epoch=512,
    gen_use_multiprocessing=True,  # default False
    optimizer_name='adadelta',
    do_augment=False,
    history_csv="./checkpoints/unet/model_history_log.csv"
)

print("Evaluating ", unet.name)
# evaluating the model
print(unet.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                 annotations_dir=data_path + "annotations_prepped_test/"))


'''
# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )
# import matplotlib.pyplot as plt
# plt.imshow(out)


# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)

from keras.callbacks import CSVLogger
csv_logger = CSVLogger("model_history_log.csv", append=True)
model.fit_generator(...,callbacks=[csv_logger])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()  # save to file instead
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
