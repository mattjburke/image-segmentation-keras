import keras
from keras.layers import *
import tensorflow as tf
import tensorflow_gan as tfgan
from .resnet50 import get_resnet50_encoder
# from tensorflow.models.official.vision.image_classification.resnet_model import resnet


def discriminator(pretrained_weights=None, input_height=224,  input_width=224):
    img_input, [f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height=input_height,  input_width=input_width, classes=2)
    x = AveragePooling2D((7, 7))(f5)
    model = keras.Model(img_input, x)
    model.add(Dense(256))
    model.add(Dense(32))
    model.add(Dense(1))
    return model


# define the combined generator and discriminator model, for updating the generator
def make_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    orig_img = g_model.input
    gen_seg_output = g_model.output

    # connect image output and label input from generator as inputs to discriminator
    stacked = tf.concatenate([orig_img, gen_seg_output], axis=3)
    gan_output = d_model(stacked)
    # define gan model as taking noise and label and outputting a classification
    model = keras.Model(orig_img, gan_output)
    # compile model
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# can I modify connection of generator and discriminator and still use this?
# gan_estimator = tfgan.estimator.GANEstimator(
#     generator_fn=unconditional_generator,
#     discriminator_fn=unconditional_discriminator,
#     generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
#     discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
#     params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
#     generator_optimizer=gen_opt,
#     discriminator_optimizer=tf.train.AdamOptimizer(discriminator_lr, 0.5),
#     get_eval_metric_ops_fn=get_eval_metric_ops_fn)
