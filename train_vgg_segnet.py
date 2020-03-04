from keras_segmentation.models.segnet import vgg_segnet
from datetime import datetime
import tensorflow as tf
print("tensorflow version is ", tf.__version__)

data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
# data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/dataset1/"
# data_path = "/Users/MatthewBurke/PycharmProjects/image-segmentation-keras/cityscape/prepped/"
print("data path is ", data_path)

print("loading vgg_segnet")
# actual data is input_height=1024, input_width=2048, but using model defaults
vgg_segnet = vgg_segnet(20) # n_classes changed from 19 to 20
print("model beginning training is ", vgg_segnet.model_name)
time_begin = str(datetime.now()).replace(' ', '')
print("beginning at", time_begin)
checkpoints_path = "./checkpoints/vgg_segnet-"+time_begin+"/"

vgg_segnet.train(
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
    history_csv=checkpoints_path+"model_history_log.csv"
)

print("finished training at", datetime.now())

print("Evaluating ", vgg_segnet.model_name)
# evaluating the model
print(vgg_segnet.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                             annotations_dir=data_path + "annotations_prepped_test/"))
