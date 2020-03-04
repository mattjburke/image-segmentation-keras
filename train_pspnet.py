from keras_segmentation.models.pspnet import pspnet
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

print("loading pspnet")
# psp_101 produces OOM error when training
# input_height=1024, input_width=2048 actual image dims, use defaults of input_height=384, input_width=576 instead
pspnet = pspnet(20)  # n_classes changed from 19 to 20
print("model beginning training is ", pspnet.model_name)

pspnet.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    input_height=None,
    input_width=None,
    n_classes=None,
    verify_dataset=True,
    checkpoints_path="./checkpoints/pspnet",
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
    history_csv="./checkpoints/pspnet/model_history_log.csv"
)

print("Evaluating ", pspnet.model_name)
# evaluating the model
print(pspnet.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                   annotations_dir=data_path + "annotations_prepped_test/"))
