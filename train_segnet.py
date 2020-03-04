from keras_segmentation.pretrained import pspnet_101_cityscapes
from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.models.unet import unet_mini
from keras_segmentation.models.fcn import fcn_8
from keras_segmentation.models.segnet import segnet
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


segnet = segnet(20, input_height=713, input_width=713)  # change to vgg_unet?  # n_classes changed from 19 to 20  # what is size of gtFine images?
print("model beginning training is ", segnet.name)

segnet.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    input_height=None,
    input_width=None,
    n_classes=None,
    verify_dataset=True,
    checkpoints_path="./checkpoints/segnet",
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
    history_csv="./checkpoints/segnet/model_history_log.csv"
)

print("Evaluating ", segnet.name)
# evaluating the model
print(segnet.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                   annotations_dir=data_path + "annotations_prepped_test/"))
