from keras_segmentation.models.segnet import segnet
from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.models.fcn import fcn_8, fcn_32
from keras_segmentation.models.unet import unet_mini
import keras_segmentation.models.gan_disc as gan_disc
from keras_segmentation.train_functions import train_gen, train_disc, train_gan, eval_gen, eval_gen_mean_iou
from datetime import datetime
import keras
import os
import tensorflow as tf
print("finished imports")

pronto_data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
pronto_checkpoints_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/"
# gcolab_data_path = "/content/image-segmentation-keras/cityscape/My Drive/Datasets/cityscape/prepped/"
# gcolab_checkpoints_path = "/content/image-segmentation-keras/checkpoints/"

data_path = pronto_data_path
checkpoints_path = pronto_checkpoints_path
# data_path = gcolab_data_path
# checkpoints_path = gcolab_checkpoints_path


def get_path(name_string):
    time_begin = str(datetime.now()).replace(' ', '-')
    print("beginning " + name_string + " training at", time_begin)
    save_checkpoints_path = checkpoints_path + name_string + "-" + time_begin + "/"
    print(name_string + " checkpoints will be saved at " + save_checkpoints_path)
    return save_checkpoints_path


# could be useful?
def flatten_model(model_nested):
    def get_layers(layers):
        layers_flat = []
        for layer in layers:
            try:
                layers_flat.extend(get_layers(layer.layers))
            except AttributeError:
                layers_flat.append(layer)
        return layers_flat

    model_flat = keras.models.Sequential(
        get_layers(model_nested.layers)
    )
    return model_flat


def train_alternately(gen_model=None, gen_iou_model=None, d_model=None, gan_model=None, gen_model_name="unknown", reg_or_stacked="stacked", train_gen_first=False):
    if train_gen_first:
        gen_checkpoints_path = get_path("gen_" + reg_or_stacked + "_" + gen_model_name)
        train_gen(g_model=gen_segnet, checkpoints_path=gen_checkpoints_path, data_path=data_path)

    gen_model.summary()
    print("Metrics at 0 are", eval_gen(gen_model, data_path=data_path))
    iteration = 1
    disc_checkpoints_path = get_path("disc_" + reg_or_stacked + "_" + gen_model_name)
    os.mkdir(disc_checkpoints_path)
    gan_checkpoints_path = get_path("gan_" + reg_or_stacked + "_" + gen_model_name)
    os.mkdir(gan_checkpoints_path)
    gen_eval_path = get_path("gen_eval_" + reg_or_stacked + "_" + gen_model_name)
    os.mkdir(gen_eval_path)
    while iteration <= 5:
        print("beginning train_disc")
        train_disc(g_model=gen_model, d_model=d_model, reg_or_stacked=reg_or_stacked, checkpoints_path=disc_checkpoints_path,
                   epochs='early_stop', data_path=data_path)

        print("beginning train_gan")
        train_gan(gan_model=gan_model, g_model=gen_model, checkpoints_path=gan_checkpoints_path,
                  epochs=2, data_path=data_path)

        # print("transferring weights")
        for layer in gan_model.layers:
            if layer.name == 'generator':
                print("found generator layer")
                print("trainable:", layer.trainable is True)
                # gen_model.set_weights(layer.get_weights())
                # assert gen_model.get_weights() == layer.get_weights()  # fails
                # print("weights equal")

        # gen_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
        # gen_model.summary()
        print("Metrics of gen at", iteration, "are", eval_gen(gen_model, log_path=gen_eval_path, data_path=data_path))
        # print("Metrics of gen_iou at", iteration, "are", eval_gen_mean_iou(gen_iou_model, data_path=data_path))
        iteration += 1
        # implement stopping condition

# ------------------------- segnet -----------------------------------------
gen_segnet = segnet(20, input_height=256, input_width=512, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
gen_checkpoints_path = get_path("gen_segnet")
train_gen(gen_segnet, gen_checkpoints_path, data_path=data_path)
# gen_segnet.load_weights("/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/gen_segnet-2020-03-30-12:21:46.457167/- 10- 0.44.hdf5")

# Train my stacked input gan
disc_segnet_stacked = gan_disc.discriminator(gen_segnet)
fake_acc = tf.keras.metrics.SpecificityAtSensitivity(0.5)  # true negative rate == Fake accuracy
real_acc = tf.keras.metrics.SensitivityAtSpecificity(0.5)  # true positive rate == Real accuracy
disc_segnet_stacked.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', fake_acc, real_acc, tf.keras.metrics.AUC()])
print("len(disc_segnet_stacked.trainable_weights) = ", len(disc_segnet_stacked.trainable_weights))
print("len(disc_segnet_stacked._collected_trainable_weights) = ", len(disc_segnet_stacked._collected_trainable_weights))
disc_segnet_stacked.summary()
gan_segnet_stacked = gan_disc.make_gan(gen_segnet, disc_segnet_stacked)
gan_segnet_stacked.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
gen_segnet_iou = gan_disc.make_gen_iou(gen_segnet)
# gen_segnet_iou.summary()
# gen_segnet_iou.compile(loss='categorical_crossentropy', optimizer='adam',
#                        metrics=['accuracy', 'sparse_categorical_accuracy', tf.keras.metrics.MeanIoU(num_classes=20)])
train_alternately(gen_model=gen_segnet, d_model=disc_segnet_stacked, gan_model=gan_segnet_stacked,
                  gen_model_name="segnet", train_gen_first=False)

# Train a regular gan
# not sure why reinitializing speeds up
gen_segnet = segnet(20, input_height=128, input_width=256, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
# gen_segnet.summary()
gen_segnet.load_weights("/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/gen_segnet-2020-03-30-12:21:46.457167/- 10- 0.44.hdf5")
disc_segnet_reg = gan_disc.discriminator_reg(gen_segnet)
disc_segnet_reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
gan_segnet_reg = gan_disc.make_gan_reg(gen_segnet, disc_segnet_reg)
gan_segnet_reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])
train_alternately(gen_model=gen_segnet, d_model=disc_segnet_reg, gan_model=gan_segnet_reg,
                  gen_model_name="segnet", reg_or_stacked="reg", train_gen_first=False)


# --------------------- pspnet ---------------------------------------
gen_pspnet = pspnet(20, input_height=128, input_width=256)  # n_classes changed from 19 to 20
gen_pspnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
gen_checkpoints_path = get_path("gen_pspnet")
train_gen(gen_pspnet, gen_checkpoints_path, data_path=data_path)
# gen_pspnet.load_weights("")

# Train my stacked input gan
disc_pspnet_stacked = gan_disc.discriminator(gen_pspnet)
disc_pspnet_stacked.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan_pspnet_stacked = gan_disc.make_gan(gen_pspnet, disc_pspnet_stacked)
gan_pspnet_stacked.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_alternately(gen_model=gen_pspnet, d_model=disc_pspnet_stacked, gan_model=gan_pspnet_stacked,
                  gen_model_name="pspnet", train_gen_first=False)

# Train a regular gan
# gen_pspnet.load_weights("")
disc_pspnet_reg = gan_disc.discriminator_reg(gen_pspnet)
disc_pspnet_reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
gan_pspnet_reg = gan_disc.make_gan_reg(gen_pspnet, disc_pspnet_reg)
gan_pspnet_reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_alternately(gen_model=gen_pspnet, d_model=disc_pspnet_reg, gan_model=gan_pspnet_reg,
                  gen_model_name="pspnet", reg_or_stacked="reg", train_gen_first=False)


# --------------------- fcn ---------------------------------------
gen_fcn = fcn_8(20, input_height=128, input_width=256)  # n_classes changed from 19 to 20
gen_fcn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gen_checkpoints_path = get_path("gen_fcn_8")
train_gen(gen_fcn, gen_checkpoints_path, data_path=data_path)
# gen_fcn.load_weights("")

# Train my stacked input gan
disc_fcn_stacked = gan_disc.discriminator(gen_fcn)
disc_fcn_stacked.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_fcn_stacked = gan_disc.make_gan(gen_fcn, disc_fcn_stacked)
gan_fcn_stacked.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
train_alternately(gen_model=gen_fcn, d_model=disc_fcn_stacked, gan_model=gan_fcn_stacked,
                  gen_model_name="fcn_8", train_gen_first=False)

# Train a regular gan
# gen_fcn.load_weights("")
disc_fcn_reg = gan_disc.discriminator_reg(gen_fcn)
disc_fcn_reg.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_fcn_reg = gan_disc.make_gan_reg(gen_fcn, disc_fcn_reg)
gan_fcn_reg.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
train_alternately(gen_model=gen_fcn, d_model=disc_fcn_reg, gan_model=gan_fcn_reg,
                  gen_model_name="fcn_8", reg_or_stacked="reg", train_gen_first=False)


# # --------------------- unet ---------------------------------------
gen_unet = unet_mini(20, input_height=128, input_width=256)  # n_classes changed from 19 to 20
gen_unet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gen_checkpoints_path = get_path("gen_unet_mini")
train_gen(gen_unet, gen_checkpoints_path, data_path=data_path)
# gen_unet.load_weights("")

# Train my stacked input gan
disc_unet_stacked = gan_disc.discriminator(gen_unet)
disc_unet_stacked.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_unet_stacked = gan_disc.make_gan(gen_unet, disc_unet_stacked)
gan_unet_stacked.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
train_alternately(gen_model=gen_unet, d_model=disc_unet_stacked, gan_model=gan_unet_stacked,
                  gen_model_name="unet_mini", train_gen_first=False)

# Train a regular gan
# gen_unet.load_weights("")
disc_unet_reg = gan_disc.discriminator_reg(gen_unet)
disc_unet_reg.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_unet_reg = gan_disc.make_gan_reg(gen_unet, disc_unet_reg)
gan_unet_reg.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
train_alternately(gen_model=gen_unet, d_model=disc_unet_reg, gan_model=gan_unet_reg,
                  gen_model_name="unet_mini", reg_or_stacked="reg", train_gen_first=False)

