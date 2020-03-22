from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from datetime import datetime
import os
import json
import keras
from keras_segmentation.data_utils.data_loader import image_segmentation_pairs_generator, image_flabels_generator
import tensorflow as tf

print("tensorflow version is ", tf.__version__)


def train_gen(g_model, checkpoints_path, weights_path=None,
              data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    os.mkdir(checkpoints_path)

    g_model.train(
        train_images=data_path + "images_prepped_train/",
        train_annotations=data_path + "annotations_prepped_train/",
        input_height=None,
        input_width=None,
        n_classes=None,
        verify_dataset=True,
        checkpoints_path=checkpoints_path,
        epochs=5,  # doesn't do anything now
        batch_size=4,  # default 2
        validate=True,
        val_images=data_path + "images_prepped_val",
        val_annotations=data_path + "annotations_prepped_val",
        val_batch_size=4,  # default 2
        auto_resume_checkpoint=False,
        load_weights=weights_path,
        steps_per_epoch=512,
        val_steps_per_epoch=512,
        gen_use_multiprocessing=True,  # default False
        optimizer_name='adadelta',
        do_augment=False,
        history_csv=checkpoints_path + "model_history_log.csv"
    )

    print("finished training at", datetime.now())

    print("Evaluating ", g_model.model_name)
    print(g_model.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                        annotations_dir=data_path + "annotations_prepped_test/"))


def train_disc(g_model, d_model, checkpoints_path, weights_path=None,
               data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    os.mkdir(checkpoints_path)

    n_classes = g_model.n_classes
    input_height = g_model.input_height
    input_width = g_model.input_width
    output_height = g_model.output_height
    output_width = g_model.output_width
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
                "model_class": d_model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    # create data generators to feed data into discriminator: images + ground truth segs and images + generated segs
    # input_height and input_width not used in image_segmentation_pairs_generator
    train_d_gen = image_segmentation_pairs_generator(train_images, train_annotations, batch_size, n_classes,
                                                     input_height, input_width,
                                                     output_height, output_width, g_model,
                                                     do_augment=do_augment)

    val_d_gen = image_segmentation_pairs_generator(val_images, val_annotations, val_batch_size, n_classes,
                                                   input_height, input_width,
                                                   output_height, output_width, g_model,
                                                   do_augment=do_augment)

    # create 3 callbacks to log
    checkpoints_path_save = checkpoints_path + "-{epoch: 02d}-{val_loss: .2f}.hdf5"
    csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=False, mode='auto', period=1)
    early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                         mode='auto', baseline=None, restore_best_weights=False)

    # train discriminator
    d_model.fit_generator(train_d_gen,
                          steps_per_epoch=steps_per_epoch,  # eliminated rest of variables before
                          validation_data=val_d_gen,
                          validation_steps=val_steps_per_epoch,
                          epochs=1000,
                          use_multiprocessing=gen_use_multiprocessing,
                          callbacks=[csv_logger, save_chckpts, early_stop])


def train_gan(gan_model, g_model, d_model, checkpoints_path, weights_path=None,
              data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    os.mkdir(checkpoints_path)

    n_classes = g_model.n_classes
    input_height = g_model.input_height
    input_width = g_model.input_width
    output_height = g_model.output_height
    output_width = g_model.output_width
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
                "model_class": d_model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    # create data generators to feed into gan: images and FAKE label
    train_gan_gen = image_flabels_generator(train_images, train_annotations, batch_size, n_classes, input_height,
                                            input_width, output_height, output_width, do_augment=False)

    val_gan_gen = image_flabels_generator(val_images, val_annotations, val_batch_size, n_classes, input_height,
                                          input_width, output_height, output_width, do_augment=False)

    # create 3 callbacks to log
    checkpoints_path_save = checkpoints_path + "-{epoch: 02d}-{val_loss: .2f}.hdf5"
    csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=False, mode='auto', period=1)
    early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                         mode='auto', baseline=None, restore_best_weights=False)

    # train GAN
    gan_model.fit_generator(train_gan_gen,
                            steps_per_epoch=steps_per_epoch,  # eliminated rest of variables before
                            validation_data=val_gan_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=1000,
                            use_multiprocessing=gen_use_multiprocessing,
                            callbacks=[csv_logger, save_chckpts, early_stop])


# ------------ Train all ----------------

data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
checkpoints_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/"

gen_segnet = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning generator training at", time_begin)
gen_checkpoints_path = checkpoints_path + "gen_segnet-" + time_begin + "/"
train_gen(gen_segnet, gen_checkpoints_path)
print("saved at" + gen_checkpoints_path)

disc_segnet = gan_disc.discriminator(gen_segnet)
disc_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning discriminator training at", time_begin)
disc_checkpoints_path = checkpoints_path + "disc_segnet-" + time_begin + "/"
train_disc(disc_segnet, disc_checkpoints_path)
print("saved at" + disc_checkpoints_path)

gan_segnet = gan_disc.make_gan(gen_segnet, disc_segnet)
gan_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning discriminator training at", time_begin)
gan_checkpoints_path = checkpoints_path + "gan_segnet-" + time_begin + "/"
train_gan(gan_segnet, gen_segnet, disc_segnet, gan_checkpoints_path)
print("saved at" + gan_checkpoints_path)


