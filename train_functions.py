from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from datetime import datetime
import os
import json
import keras
from keras_segmentation.models.gan_disc import make_gan
from keras_segmentation.data_utils.data_loader import image_flabels_generator, image_segmentation_pairs_dataset, image_segmentation_generator
import tensorflow as tf
from keras.models import load_model, load_weights
from keras_segmentation.models.model_utils import add_input_dims

print("tensorflow version is ", tf.__version__)


def train_gen(g_model, checkpoints_path, load_g_model_path=None,
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
        load_weights=load_g_model_path,  # uses model.load_weights(load_weights)
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


def train_disc(g_model, d_model, checkpoints_path, load_g_weights_path=None, load_d_model_path=None,
               data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):
    os.mkdir(checkpoints_path)

    if load_g_weights_path is not None and len(load_g_weights_path) > 0:
        print("Loading weights from ", load_g_weights_path)
        g_model.load_weights(load_g_weights_path)

    # if load_g_model_path is not None:
    #     g_model = load_weights(load_g_model_path)

    # if load_d_model_path is not None:
    #     d_model = load_model(load_d_model_path)
    #     d_model = add_input_dims(d_model)

    train_images = data_path + "images_prepped_train/"
    train_annotations = data_path + "annotations_prepped_train/"
    val_images = data_path + "images_prepped_val"
    val_annotations = data_path + "annotations_prepped_val"

    do_augment = False
    history_csv = checkpoints_path + "model_history_log.csv"

    with open(checkpoints_path + "_config.json", "w") as f:
        json.dump({
            "model_class": "discriminator",  # basic keras models do not have model_name
            "n_classes": g_model.n_classes,  # loaded model is a basic model, doesn't have n_classes; load just weights
            "g_input_height": g_model.input_height,
            "g_input_width": g_model.input_width,
            "g_output_height": g_model.output_height,
            "g_output_width": g_model.output_width
        }, f)

    # create and preprocess training dataset all at once instead of using training generators
    X_train, Y_train = image_segmentation_pairs_dataset(train_images, train_annotations, g_model, do_augment=do_augment)
    X_val, Y_val = image_segmentation_pairs_dataset(val_images, val_annotations, g_model, do_augment=do_augment)

    # create 3 callbacks to log
    checkpoints_path_save = checkpoints_path + "e{epoch:02d}vl{val_loss:.2f}.hdf5"
    csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=True, mode='auto', period=1)
    early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                         mode='auto', baseline=None, restore_best_weights=False)

    d_model.fit(X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=1000,
                use_multiprocessing=False,  # Used for generator or keras.utils.Sequence input only
                callbacks=[csv_logger, save_chckpts, early_stop])


def train_gan(checkpoints_path, gan_model=None, g_model=None, d_model=None,
              load_g_model_path=None, load_d_model_path=None, load_gan_path=None,
              data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):

    os.mkdir(checkpoints_path)

    # if load_g_model_path is not None:
    #     g_model = load_model(load_g_model_path)  # gets rid of model.input_height and other variables?
    #
    # if load_d_model_path is not None:
    #     d_model = load_model(load_d_model_path)
    #
    # if load_gan_path is not None:
    #     gan_model = load_model(load_gan_path)
    #
    # if gan_model is None:
    #     gan_model = make_gan(g_model, d_model)

    input_height = g_model.input_height
    input_width = g_model.input_width

    train_images = data_path + "images_prepped_train/"
    train_annotations = data_path + "annotations_prepped_train/"
    val_images = data_path + "images_prepped_val"
    val_annotations = data_path + "annotations_prepped_val"

    do_augment = False
    history_csv = checkpoints_path + "model_history_log.csv"

    batch_size = 4
    val_batch_size = 4
    steps_per_epoch = 512
    val_steps_per_epoch = 512
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
    early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                         mode='auto', baseline=None, restore_best_weights=False)

    # train GAN
    gan_model.fit_generator(train_gan_gen,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_gan_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=1000,
                            use_multiprocessing=gen_use_multiprocessing,
                            callbacks=[csv_logger, save_chckpts, early_stop])


def eval_gen(gen_model, data_path="/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"):

    test_images = data_path + "images_prepped_test/"
    test_annotations = data_path + "annotations_prepped_test/"

    n_classes = gen_model.n_classes
    input_height = gen_model.input_height
    input_width = gen_model.input_width
    output_height = gen_model.output_height
    output_width = gen_model.output_width
    batch_size = 4
    do_augment = False
    # history_csv = checkpoints_path + "model_history_log.csv"

    test_data_gen = image_segmentation_generator(
        test_images, test_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment)

    return gen_model.evaluate_generator(test_data_gen, use_multiprocessing=True, verbose=1)


# def alternate_training(gan_model, d_model):
    # train generator
    # create dataset for discriminator
    # train discriminator
    # update gan with disc weights
    # train gan
    # load generator section of gan weights into generator
    # test accuracy of generator for comparison
    # repeat past 3 steps until neither improve (early stop before x epochs?)

    # how to update weights of only specific layer (only update discrim)
    # how to extract certain layers to make new model (make gen from gan)
    # need to label layers?



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

