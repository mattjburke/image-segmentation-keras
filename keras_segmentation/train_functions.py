from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from datetime import datetime
import os
import json
import keras
from keras_segmentation.models.gan_disc import make_gan
from keras_segmentation.data_utils.data_loader import image_flabels_generator, image_segmentation_pairs_dataset, \
    image_segmentation_pairs_generator, image_segmentation_generator
import tensorflow as tf
# from keras.models import load_model, load_weights
from keras_segmentation.models.model_utils import add_input_dims

print("tensorflow version is ", tf.__version__)


def train_gen(g_model=None, checkpoints_path=None, load_g_model_path=None,
              data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    if (checkpoints_path is None) or (g_model is None):
        print("train_gen() needs a g_model and checkpoints_path")
    assert checkpoints_path is not None
    assert g_model is not None

    os.mkdir(checkpoints_path)

    g_model.train(
        train_images=data_path + "images_prepped_train/",
        train_annotations=data_path + "annotations_prepped_train/",
        input_height=None,
        input_width=None,
        n_classes=None,
        verify_dataset=True,
        checkpoints_path=checkpoints_path,
        epochs=1000,  # doesn't do anything now
        batch_size=25,
        validate=True,
        val_images=data_path + "images_prepped_val",
        val_annotations=data_path + "annotations_prepped_val",
        val_batch_size=25,  # default 2
        auto_resume_checkpoint=False,
        load_weights=load_g_model_path,  # uses model.load_weights(load_weights)
        steps_per_epoch=119,  # there are 1525 test images, 2975 train, and 500 val
        val_steps_per_epoch=20,
        gen_use_multiprocessing=True,  # default False
        optimizer_name='adadelta',
        do_augment=False,
        history_csv=checkpoints_path + "model_history_log.csv"
    )

    print("finished generator training at", datetime.now())

    # runs, but eval function seems off since mostly outputs 0s
    # print("Evaluating ", g_model.model_name)
    # print(g_model.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
    #                                     annotations_dir=data_path + "annotations_prepped_test/"))


# g_model is passed to create training dataset
def train_disc(g_model=None, d_model=None, checkpoints_path=None, epochs=2, reg_or_stacked="stacked",
               data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    if (checkpoints_path is None) or (d_model is None) or (g_model is None):
        print("train_disc() needs a d_model, g_model, and checkpoints_path")
    assert checkpoints_path is not None
    assert d_model is not None
    assert g_model is not None

    os.mkdir(checkpoints_path)

    # need to compile again to set as trainable
    d_model.trainable = True
    d_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # for layer in d_model.layers:
    #     print(layer.name, layer.trainable)


    train_images = data_path + "images_prepped_train/"
    train_annotations = data_path + "annotations_prepped_train/"
    val_images = data_path + "images_prepped_val"
    val_annotations = data_path + "annotations_prepped_val"

    do_augment = False
    history_csv = checkpoints_path + "model_history_log.csv"

    batch_size = 5
    val_batch_size = 5
    steps_per_epoch = 595
    val_steps_per_epoch = 100
    gen_use_multiprocessing = True  # messes up generator?

    with open(checkpoints_path + "_config.json", "w") as f:
        json.dump({
            "model_class": "discriminator",  # basic keras models do not have model_name
            "n_classes": g_model.n_classes,  # loaded model is a basic model, doesn't have n_classes; load just weights
            "g_input_height": g_model.input_height,
            "g_input_width": g_model.input_width,
            "g_output_height": g_model.output_height,
            "g_output_width": g_model.output_width
        }, f)

    # this step takes too much memory??
    # print("creating datasets X_train, Y_train")
    # # create and preprocess training dataset all at once instead of using training generators
    # X_train, Y_train = image_segmentation_pairs_dataset(train_images, train_annotations, g_model, do_augment=do_augment)
    # print("creating datasets X_val, Y_val")
    # X_val, Y_val = image_segmentation_pairs_dataset(val_images, val_annotations, g_model, do_augment=do_augment)

    train_d_gen = image_segmentation_pairs_generator(train_images, train_annotations, batch_size,
                                                     g_model, reg_or_stacked=reg_or_stacked, do_augment=do_augment)

    val_d_gen = image_segmentation_pairs_generator(val_images, val_annotations, val_batch_size,
                                                   g_model, reg_or_stacked=reg_or_stacked, do_augment=do_augment)

    # create 3 callbacks to log
    checkpoints_path_save = checkpoints_path + "e{epoch:02d}vl{val_loss:.2f}.hdf5"
    csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=True, mode='auto', period=1)

    if epochs == "early_stop":
        early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                             mode='auto', baseline=None, restore_best_weights=False)

        print("training discriminator with early stopping")

        d_model.fit_generator(train_d_gen,
                              workers=0,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_d_gen,
                              validation_steps=val_steps_per_epoch,
                              epochs=1000,
                              use_multiprocessing=gen_use_multiprocessing,
                              callbacks=[csv_logger, save_chckpts, early_stop])
    else:

        print("training discriminator for epochs =", epochs)

        d_model.fit_generator(train_d_gen,
                              workers=0,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_d_gen,
                              validation_steps=val_steps_per_epoch,
                              epochs=epochs,
                              use_multiprocessing=gen_use_multiprocessing,
                              callbacks=[csv_logger, save_chckpts])

    print("finished discriminator training at", datetime.now())


# g_model is passed to use input_height, output_height, etc to create data loaders and save to log
def train_gan(checkpoints_path=None, gan_model=None, g_model=None, epochs=2, num_gen_layers=5,
              data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    if (checkpoints_path is None) or (gan_model is None) or (g_model is None):
        print("train_gan() needs a gan_model, g_model, and checkpoints_path")
    assert checkpoints_path is not None
    assert gan_model is not None
    assert g_model is not None

    os.mkdir(checkpoints_path)

    # d_model.trainable = False
    # print("------ before setting trainable ----------------------------")
    # for layer in gan_model.layers:
    #     print(layer.name, layer.trainable)

    for layer in gan_model.layers:
        if layer.name == 'discriminator':
            layer.trainable = False

    # need to compile after changing trainable
    gan_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # print("------- after setting trainable ----------------------------")
    # for layer in gan_model.layers:
    #     print(layer.name, layer.trainable)


    input_height = g_model.input_height
    input_width = g_model.input_width
    # input_height = gan_model.input_height  # added with add_input_dims(model)
    # input_width = gan_model.input_width

    train_images = data_path + "images_prepped_train/"
    train_annotations = data_path + "annotations_prepped_train/"
    val_images = data_path + "images_prepped_val"
    val_annotations = data_path + "annotations_prepped_val"

    do_augment = False
    history_csv = checkpoints_path + "model_history_log.csv"

    batch_size = 5
    val_batch_size = 5
    steps_per_epoch = 595
    val_steps_per_epoch = 100
    gen_use_multiprocessing = True

    with open(checkpoints_path + "_config.json", "w") as f:
        json.dump({
            "model_class": "GAN",
            "n_classes": g_model.n_classes,
            "input_height": g_model.input_height,
            "input_width": g_model.input_width,
            "gen_output_height": g_model.output_height,
            "gen_output_width": g_model.output_width
        }, f)

    print("creating gan generators")
    # create data generators to feed into gan: images and FAKE label
    train_gan_gen = image_flabels_generator(train_images, train_annotations, batch_size,
                                            input_height, input_width, do_augment=do_augment)

    val_gan_gen = image_flabels_generator(val_images, val_annotations, val_batch_size,
                                          input_height, input_width, do_augment=do_augment)

    # create 3 callbacks to log
    checkpoints_path_save = checkpoints_path + "-{epoch:02d}-{val_loss:.2f}.hdf5"
    csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=True, mode='auto', period=1)

    if epochs == "early_stop":
        early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                             mode='auto', baseline=None, restore_best_weights=False)

        print("training gan with early stopping")
        # train GAN
        gan_model.fit_generator(train_gan_gen,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_gan_gen,
                                validation_steps=val_steps_per_epoch,
                                epochs=1000,
                                use_multiprocessing=gen_use_multiprocessing,
                                callbacks=[csv_logger, save_chckpts, early_stop])

    else:
        print("training gan for epochs =", epochs)
        # train GAN
        gan_model.fit_generator(train_gan_gen,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=val_gan_gen,
                                validation_steps=val_steps_per_epoch,
                                epochs=epochs,
                                use_multiprocessing=gen_use_multiprocessing,
                                callbacks=[csv_logger, save_chckpts])

    print("finished gan training at", datetime.now())


def eval_gen(gen_model, data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    test_images = data_path + "images_prepped_test/"
    test_annotations = data_path + "annotations_prepped_test/"

    n_classes = gen_model.n_classes
    input_height = gen_model.input_height
    input_width = gen_model.input_width
    output_height = gen_model.output_height
    output_width = gen_model.output_width
    batch_size = 5  # what size can be handled in memory? bigger = faster
    do_augment = False
    # history_csv = checkpoints_path + "model_history_log.csv"

    test_data_gen = image_segmentation_generator(
        test_images, test_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment)

    # print(gen_model.metrics_names)
    # there are 1525 test images, 2975 train, and 500 val
    return gen_model.evaluate_generator(test_data_gen, steps=305, use_multiprocessing=True, verbose=1)



    # from https://github.com/keras-team/keras/blob/master/keras/engine/network.py
    # class Network(Layer):
    """A Network is a directed acyclic graph of layers.
    It is the topological form of a "model". A Model
    is simply a Network with added training routines.
    # Properties
        name
        inputs
        outputs
        layers
        input_spec (list of class instances)
            each entry describes one required input:
                - ndim
                - dtype
        trainable (boolean)
        dtype
        input_shape
        output_shape
        weights (list of variables)
        trainable_weights (list of variables)
        non_trainable_weights (list of variables)
        losses
        updates
        state_updates
        stateful
    # Methods
        __call__
        summary
        get_layer
        get_weights
        set_weights
        get_config
        compute_output_shape
        save
        add_loss
        add_update
        get_losses_for
        get_updates_for
        to_json
        to_yaml
        reset_states
    # Class Methods
        from_config
    # Raises
        TypeError: if input tensors are not Keras tensors
            (tensors returned by `Input`).
    """
