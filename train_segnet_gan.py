from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from datetime import datetime
import os
import json
import keras
from keras_segmentation.data_utils.data_loader import image_segmentation_pairs_generator, image_flabels_generator
import tensorflow as tf

print("tensorflow version is ", tf.__version__)

data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
# data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/dataset1/"
# data_path = "/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/cityscape/prepped/"
print("data path is ", data_path)

print("loading gen_segnet")
# actual data is input_height=1024, input_width=2048, but using model defaults of input_height=416, input_width=608,
# encoder_level=3
gen_segnet = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20

n_classes = gen_segnet.n_classes
input_height = gen_segnet.input_height
input_width = gen_segnet.input_width
output_height = gen_segnet.output_height
output_width = gen_segnet.output_width
print("n_classes = ", n_classes)
print("input_height =", input_height)
print("input_width =", input_width)
print("output_height =", output_height)
print("output_width =", output_width)

print("creating disc_segnet")
# output_height and output_width needs to be assigned to variable (intead of using gen_segnet.input_height) since
# python is pass-by-object-reference
disc_segnet = gan_disc.discriminator(input_height=output_height, input_width=output_width)

time_begin = str(datetime.now()).replace(' ', '')
print("beginning at", time_begin)
checkpoints_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/segnet_disc-" + time_begin + "/"
os.mkdir(checkpoints_path)

train_images = data_path + "images_prepped_train/"
train_annotations = data_path + "annotations_prepped_train/"
verify_dataset = True
checkpoints_path = checkpoints_path
epochs = 5  # doesn't do anything now
batch_size = 4  # default 2
validate = True
val_images = data_path + "images_prepped_val"
val_annotations = data_path + "annotations_prepped_val"
val_batch_size = 4  # default 2
auto_resume_checkpoint = False
load_weights = None
steps_per_epoch = 512
val_steps_per_epoch = 512
gen_use_multiprocessing = True  # default False
optimizer_name = 'adadelta'
do_augment = False
history_csv = checkpoints_path + "model_history_log.csv"

if checkpoints_path is not None:
    with open(checkpoints_path + "_config.json", "w") as f:
        json.dump({
            "model_class": disc_segnet.model_name,
            "n_classes": n_classes,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": output_height,
            "output_width": output_width
        }, f)

# input_height and input_width not used in image_segmentation_pairs_generator
train_d_gen = image_segmentation_pairs_generator(train_images, train_annotations, batch_size, n_classes, input_height,
                                                 input_width, output_height, output_width, gen_segnet,
                                                 do_augment=do_augment)

val_d_gen = image_segmentation_pairs_generator(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                               input_width, output_height, output_width, gen_segnet,
                                               do_augment=do_augment)

# create 3 callbacks to log
checkpoints_path_save = checkpoints_path + "-{epoch: 02d}-{val_loss: .2f}.hdf5"
csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                         verbose=1, save_best_only=False,
                                                         save_weights_only=False, mode='auto', period=1)
early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                     mode='auto', baseline=None, restore_best_weights=False)

# disc_segnet.summary()

gen_segnet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

disc_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# disc_segnet.fit_generator(train_d_gen,
#                           steps_per_epoch=steps_per_epoch,  # eliminated rest of variables before
#                           validation_data=val_d_gen,
#                           validation_steps=val_steps_per_epoch,
#                           epochs=1000,
#                           use_multiprocessing=gen_use_multiprocessing,
#                           callbacks=[csv_logger, save_chckpts, early_stop])

gan = gan_disc.make_gan(gen_segnet, disc_segnet)

gan.summary()

gan.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

train_gan_gen = image_flabels_generator(train_images, train_annotations, batch_size, n_classes, input_height,
                                        input_width, output_height, output_width, do_augment=False)

gan.fit_generator(train_d_gen,
                  steps_per_epoch=steps_per_epoch,  # eliminated rest of variables before
                  validation_data=val_d_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=1000,
                  use_multiprocessing=gen_use_multiprocessing,
                  callbacks=[csv_logger, save_chckpts, early_stop])

