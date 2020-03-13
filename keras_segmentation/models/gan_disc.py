import keras
from keras.layers import *
import tensorflow as tf
import tensorflow_gan as tfgan
from .resnet50 import get_resnet50_encoder
# from tensorflow.models.official.vision.image_classification.resnet_model import resnet


def discriminator(pretrained_weights=None, input_height=224,  input_width=224, model_name="resnet50_discrim"):
    img_input, [f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height=input_height,  input_width=input_width, input_chan=4, classes=2)
    x = AveragePooling2D((7, 7))(f5)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Dense(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.Model(img_input, x)
    model.model_name = model_name
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


# # see https://androidkt.com/create-gans-model-using-tensorflow-tfgan-library/
# def generator_fn(inputs, mode, keras_g_model):
#     is_training = mode == tf.estimator.ModeKeys.TRAIN
#     # model = generator()
#     return keras_g_model(inputs, is_training)
#
#
# def discriminator_fn(inputs, conditioning, mode, keras_d_model):
#     is_training = mode == tf.estimator.ModeKeys.TRAIN
#     # model = discriminator()
#     return keras_d_model(inputs, is_training)
#
#
# def gan():
#     # hyper param
#     model_dir = "../logs-2/"
#
#     batch_size = 64
#     num_epochs = 10
#     noise_dim = 64
#
#     # Run Configuration
#     run_config = tf.estimator.RunConfig(
#         model_dir=model_dir, save_summary_steps=100, save_checkpoints_steps=1000)
#
#     gan_estimator = tfgan.estimator.GANEstimator(
#         config=run_config,
#
#         generator_fn=generator_fn,
#         discriminator_fn=discriminator_fn,
#
#         generator_loss_fn=tfgan.losses.modified_generator_loss,
#         discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
#
#         generator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
#         discriminator_optimizer=tf.train.AdamOptimizer(0.0002, 0.5),
#         add_summaries=tfgan.estimator.SummaryType.IMAGES)
#
#     input_fn = train_input_fn(batch_size, num_epochs, noise_dim)
#     model = gan_estimator.train(input_fn, max_steps=None)
#
#     return model


# can I modify connection of generator and discriminator and still use this?
# test with regular discriminator first?
# gan_estimator = tfgan.estimator.GANEstimator(
#     generator_fn=unconditional_generator,
#     discriminator_fn=unconditional_discriminator,
#     generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
#     discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
#     params={'batch_size': train_batch_size, 'noise_dims': noise_dimensions},
#     generator_optimizer=gen_opt,
#     discriminator_optimizer=tf.train.AdamOptimizer(discriminator_lr, 0.5),
#     get_eval_metric_ops_fn=get_eval_metric_ops_fn)
