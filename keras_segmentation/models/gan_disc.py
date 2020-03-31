import keras
from keras.layers import *
import tensorflow as tf
from .resnet50 import get_resnet50_encoder
from .model_utils import add_input_dims


def discriminator(g_model, pretrained_weights=None, model_name="resnet50_discrim"):
    input_height = g_model.output_height
    input_width = g_model.output_width
    img_input, [f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height=input_height,  input_width=input_width,
                                                           input_chan=23, classes=2, pretrained=None)
    # x = AveragePooling2D((7, 7))(f5)
    x = Flatten()(f5)
    x = Dense(256)(x)
    x = Dense(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.Model(img_input, x)
    model = add_input_dims(model)
    model.model_name = model_name
    return model


def discriminator_reg(g_model, pretrained_weights=None, model_name="resnet50_discrim"):
    input_height = g_model.output_height
    input_width = g_model.output_width
    # input is 20 channels, one for each segmentation class
    img_input, [f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height=input_height,  input_width=input_width,
                                                           input_chan=20, classes=2, pretrained=None)
    # x = AveragePooling2D((7, 7))(f5)
    x = Flatten()(f5)
    x = Dense(256)(x)
    x = Dense(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.Model(img_input, x)
    model = add_input_dims(model)
    model.model_name = model_name
    return model


# define the combined generator and discriminator model, for updating the generator
def make_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    orig_img = g_model.input
    # shaped_o = Reshape([208, 304, 20])(g_model.output)  # should do nothing if same shape
    # This resizes in the same way as GT dataset (in dataloader)
    out_height = g_model.output_height  # 208 for segnet
    out_width = g_model.output_width  # 304 for segnet
    im_array_out = Lambda(lambda x: tf.compat.v1.image.resize(x, [out_height, out_width], align_corners=True))(orig_img)
    stacked = concatenate([im_array_out, g_model.output])
    gan_output = d_model(stacked)
    model = keras.Model(orig_img, gan_output)
    model = add_input_dims(model)  # use this or set to match g_model?
    return model


def make_gan_reg(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    orig_img = g_model.input
    # shaped_o = Reshape([208, 304, 20])(g_model.output)
    # This resizes in the same way as GT dataset (in dataloader)
    # im_array_out = Lambda(lambda x: tf.compat.v1.image.resize(x, [208, 304], align_corners=True))(orig_img)
    # stacked = concatenate([im_array_out, shaped_o])
    gan_output = d_model(g_model.output)
    model = keras.Model(orig_img, gan_output)
    model = add_input_dims(model)  # use this or set to match g_model?
    return model

