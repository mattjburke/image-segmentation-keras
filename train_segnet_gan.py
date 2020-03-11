from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from datetime import datetime
import os
import tensorflow as tf
print("tensorflow version is ", tf.__version__)

data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
# data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/dataset1/"
# data_path = "/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/cityscape/prepped/"
print("data path is ", data_path)

print("loading segnet")
# actual data is input_height=1024, input_width=2048, but using model defaults of input_height=416, input_width=608, encoder_level=3
gen_segnet = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20

# disc_seg = gan_disc.discriminator(input_height=416, input_width=608)
#
# image_segmentation_pairs_generator(images_path, segs_path, batch_size,
#                                  n_classes, input_height, input_width,
#                                  output_height, output_width, gen_model,
#                                  do_augment=False)
# train_images=data_path + "images_prepped_train/"
# train_annotations=data_path + "annotations_prepped_train/"
# batch_size=2
# n_classes=20
# input_height=416
# input_width=608
#
# n_classes = model.n_classes
# input_height = model.input_height
# input_width = model.input_width
# output_height = model.output_height
# output_width = model.output_width
#
#
# train_gen = image_segmentation_generator(
#             train_images, train_annotations,  batch_size,  n_classes,
#             input_height, input_width, output_height, output_width, do_augment=do_augment )

disc_segnet = gan_disc.discriminator(input_height=416, input_width=608)


time_begin = str(datetime.now()).replace(' ', '')
print("beginning at", time_begin)
checkpoints_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/segnet_disc-"+time_begin+"/"
os.mkdir(checkpoints_path)

disc_segnet.train(
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
    load_weights=None,
    steps_per_epoch=512,
    val_steps_per_epoch=512,
    gen_use_multiprocessing=True,  # default False
    optimizer_name='adadelta',
    do_augment=False,
    history_csv=checkpoints_path+"model_history_log.csv",
    train_gen="discrim_input"
)
