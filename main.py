from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from keras_segmentation.train_functions import train_gen, train_disc, train_gan, eval_gen
from datetime import datetime
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
    print("beginning" + name_string + "training at", time_begin)
    save_checkpoints_path = checkpoints_path + name_string + "-" + time_begin + "/"
    print(name_string + "checkpoints will be saved at" + save_checkpoints_path)
    return save_checkpoints_path

'''
Models follow 4 steps:
create model
compile model
create unique path to save checkpoints
train model
'''


# Generator model creation and training
gen_segnet = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gen_checkpoints_path = get_path("gen_segnet")
train_gen(gen_segnet, gen_checkpoints_path, data_path=data_path)


# Dicsriminator model creation and training
disc_segnet = gan_disc.discriminator(gen_segnet)
disc_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# just to test loading one previously saved model
# gen_segnet_load_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/gen_segnet-2020-03-21-21:50:30.495545/- 23- 0.82.hdf5"
disc_checkpoints_path = get_path("disc_segnet")
train_disc(gen_segnet, disc_segnet, disc_checkpoints_path, load_g_model_path=None, data_path=data_path)


# GAN creation and training
gan_segnet = gan_disc.make_gan(gen_segnet, disc_segnet)
gan_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_checkpoints_path = get_path("gan_segnet")
train_gan(gan_segnet, gen_segnet, disc_segnet, gan_checkpoints_path, data_path=data_path)


# load weights from GAN into generator to evaluate improvement
gen_weights = gen_segnet.get_weights()
gan_weights = gan_segnet.get_weights()
new_gen_weights = []
for layer in len(gen_weights):
    new_gen_weights.append(gan_weights(layer))

prev = eval_gen(gen_segnet)
print(prev)
gen_segnet.set_weights(new_gen_weights)
improved = eval_gen(gen_segnet)
print(improved)

# continue loop of disc training (from newly created dataset), gan training, and re-eval of gen
# NEED TO IMPLEMENT STILL

# train again just to see if history csv matches up or improves more
gen_checkpoints_path2 = get_path("gen_segnet_improved")
train_gen(gen_segnet, gen_checkpoints_path2, data_path=data_path)


# ------------ Normal GAN -----------------------------------

# make separate generator to compare using normal gan
gen_segnet_normal = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet_normal.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gen_segnet_normal.set_weights(gen_weights)  # no need to retrain new model, just use previously saved weights

# create and train normal discriminator
disc_segnet_reg = gan_disc.discriminator_reg(gen_segnet_normal)
disc_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
disc_reg_checkpoints_path = get_path("disc_segnet_reg")
train_disc(gen_segnet_normal, disc_segnet_reg, disc_reg_checkpoints_path, load_g_model_path=None, data_path=data_path)

# create and train normal gan
gan_segnet_reg = gan_disc.make_gan_reg(gen_segnet_normal, disc_segnet_reg)
gan_segnet_reg.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
gan_reg_path = get_path("gan_reg")
train_gan(gan_segnet_reg, gan_segnet_reg, disc_segnet, gan_reg_path, data_path=data_path)

# load weights from normal GAN into generator to evaluate improvement

# continue loop of disc training (from newly created datset), gan training, and re-eval of gen
# compare improvement from normal gan to improvement from my stacked input gan


