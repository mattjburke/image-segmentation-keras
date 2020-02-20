from keras_segmentation.pretrained import pspnet_101_cityscapes

# python -m keras_segmentation verify_dataset
#  --images_path="dataset1/images_prepped_train/"
#  --segs_path="dataset1/annotations_prepped_train/"
#  --n_classes=50
#
# python -m keras_segmentation visualize_dataset
#  --images_path="dataset1/images_prepped_train/"
#  --segs_path="dataset1/annotations_prepped_train/"
#  --n_classes=50


pret_model = pspnet_101_cityscapes()  # load the pretrained model trained on Cityscapes dataset

'''
def pspnet_101_cityscapes():

    model_config = {
        "input_height": 713,
        "input_width": 713,
        "n_classes": 19,
        "model_class": "pspnet_101",
    }

    model_url = "https://www.dropbox.com/s/" \
                "c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1"
    latest_weights = keras.utils.get_file("pspnet101_cityscapes.h5", model_url)

    return model_from_checkpoint_path(model_config, latest_weights)

def pspnet_101(n_classes,  input_height=473, input_width=473):
    from ._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape)
    model.model_name = "pspnet_101"
    return model
'''

psp_gtfine = pspnet_101(19, 713, 713)

data_path = "./cityscape/prepped/"
psp_gtfine.train(
    train_images=data_path + "images_prepped_train/",
    train_annotations=data_path + "annotations_prepped_train/",
    checkpoints_path=data_path + "checkpoints/psp_gtfine", epochs=5
)

# want to plot validation loss while training

# out = model.predict_segmentation(
#     inp="dataset1/images_prepped_test/0016E5_07965.png",
#     out_fname="/tmp/out.png"
# )
# import matplotlib.pyplot as plt
# plt.imshow(out)

# evaluating the model
print(psp_gtfine.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                       annotations_dir=data_path + "annotations_prepped_test/"))

# evaluating the pretrained model
print(pret_model.evaluate_segmentation(inp_images_dir=data_path + "images_prepped_test/",
                                       annotations_dir=data_path + "annotations_prepped_test/"))
