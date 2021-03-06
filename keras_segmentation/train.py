import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, image_segmentation_pairs_generator
import os
import glob
import six
import keras

# history_csv = "model_history_log.csv"

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint


def train(model,
          train_images,
          train_annotations,
          input_height=None,  # uses value from model if None
          input_width=None,  # uses value from model if None
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=3,  # max epochs, could be fewer if early stopping is triggered (if patience < epochs)
          batch_size=25,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=25,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=119,
          val_steps_per_epoch=20,
          gen_use_multiprocessing=False,
          optimizer_name='adadelta',
          do_augment=False,
          history_csv=None,
          train_gen=None
          ):

    from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
            assert verified

    if train_gen is None:
        train_gen = image_segmentation_generator(
            train_images, train_annotations,  batch_size,  n_classes,
            input_height, input_width, output_height, output_width, do_augment=do_augment )  # does keras allow training on arrays of different shapes?

    if train_gen is "discrim_input":
        train_gen = image_segmentation_pairs_generator(
            train_images, train_annotations,  batch_size,  n_classes,
            input_height, input_width, output_height, output_width, do_augment=do_augment )

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if history_csv is not None:
        csv_logger = keras.callbacks.callbacks.CSVLogger(history_csv, append=True)

    checkpoints_path_save = checkpoints_path +  "-{epoch: 02d}-{val_loss: .2f}.hdf5"
    save_chckpts = keras.callbacks.callbacks.ModelCheckpoint(checkpoints_path_save, monitor='val_loss',
                                                             verbose=1, save_best_only=False,
                                                             save_weights_only=True, mode='auto', period=1)
    early_stop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                                         mode='auto', baseline=None, restore_best_weights=False)

    model.summary()

    if not validate:
        model.fit_generator(train_gen, steps_per_epoch, epochs=1000, callbacks=[csv_logger, save_chckpts, early_stop])
    else:
        model.fit_generator(train_gen, steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs,
                            use_multiprocessing=gen_use_multiprocessing,
                            callbacks=[csv_logger, save_chckpts, early_stop])

