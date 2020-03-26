from keras_segmentation.models.segnet import segnet
import keras_segmentation.models.gan_disc as gan_disc
from .train_functions import train_gen, train_disc, train_gan, eval_gen
from datetime import datetime

pronto_data_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/cityscape/prepped/"
pronto_checkpoints_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/"
# gcolab_data_path = "/content/image-segmentation-keras/cityscape/My Drive/Datasets/cityscape/prepped/"
# gcolab_checkpoints_path = "/content/image-segmentation-keras/checkpoints/"

data_path = pronto_data_path
checkpoints_path = pronto_checkpoints_path
# data_path = gcolab_data_path
# checkpoints_path = gcolab_checkpoints_path


# Generator model creation and training
gen_segnet = segnet(20, input_height=416, input_width=608, encoder_level=3)  # n_classes changed from 19 to 20
gen_segnet.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning generator training at", time_begin)
gen_checkpoints_path = checkpoints_path + "gen_segnet-" + time_begin + "/"
train_gen(gen_segnet, gen_checkpoints_path, data_path=data_path)
print("saved at" + gen_checkpoints_path)


# Dicsriminator model creation and training
disc_segnet = gan_disc.discriminator(gen_segnet)
disc_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# just to test loading one previously saved model
# gen_segnet_load_path = "/work/LAS/jannesar-lab/mburke/image-segmentation-keras/checkpoints/gen_segnet-2020-03-21-21:50:30.495545/- 23- 0.82.hdf5"

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning discriminator training at", time_begin)
disc_checkpoints_path = checkpoints_path + "disc_segnet-" + time_begin + "/"
train_disc(gen_segnet, disc_segnet, disc_checkpoints_path, load_g_model_path=None, data_path=data_path)
print("saved at" + disc_checkpoints_path)


# GAN creation and training
gan_segnet = gan_disc.make_gan(gen_segnet, disc_segnet)
gan_segnet.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

time_begin = str(datetime.now()).replace(' ', '-')
print("beginning discriminator training at", time_begin)
gan_checkpoints_path = checkpoints_path + "gan_segnet-" + time_begin + "/"
train_gan(gan_segnet, gen_segnet, disc_segnet, gan_checkpoints_path, data_path=data_path)
print("saved at" + gan_checkpoints_path)

# load weights from GAN into generator to evaluate improvement
# continue loop of disc training (from newly created datset), gan training, and re-eval of gen

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

# train again just to see if history csv matches up or improves more
time_begin = str(datetime.now()).replace(' ', '-')
print("beginning generator training at", time_begin)
gen_checkpoints_path2 = checkpoints_path + "gen_segnet_improved-" + time_begin + "/"
train_gen(gen_segnet, gen_checkpoints_path2, data_path=data_path)
print("saved at" + gen_checkpoints_path2)

