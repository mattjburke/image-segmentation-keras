import keras
from keras.layers import *
import tensorflow as tf
from .resnet50 import get_resnet50_encoder
from .basic_models import vanilla_encoder
from ..data_utils.data_loader import get_image_array
# from .model_utils import add_input_dims


def discriminator(g_model, pretrained_weights=None, model_name="resnet50_discrim"):
    # discriminator input dimensions should match generator output dimensions
    input_height = g_model.output_height
    input_width = g_model.output_width
    # input is 20 channels, one for each segmentation class and 3 for image rgb channels
    # img_input, [f1, f2, f3, f4, f5] = get_resnet50_encoder(input_height=input_height,  input_width=input_width,
    #                                                        input_chan=23, pretrained=None)

    img_input, levels = vanilla_encoder(input_height=input_height,  input_width=input_width, input_chan=23)
    # x = AveragePooling2D((7, 7))(f5)
    f4 = levels[0]
    x = Flatten()(f4)
    # x = Dense(256)(x)
    x = Dense(32)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = keras.Model(img_input, x)
    # model = add_input_dims(model)
    # model.model_name = model_name
    return model


def discriminator_reg(g_model, pretrained_weights=None, model_name="resnet50_discrim"):
    # discriminator input dimensions should match generator output dimensions
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
    # model = add_input_dims(model)
    # model.model_name = model_name
    return model


# define the combined generator and discriminator model, for updating the generator
def make_gan(g_model, d_model):
    orig_img = g_model.input
    out_height = g_model.output_height
    out_width = g_model.output_width
    # This resizes in the same way as GT dataset (in dataloader)? Yes, changed data_loader.image_segmentation_pairs_generator to match
    im_array_out = Lambda(lambda x: tf.compat.v1.image.resize(x, [out_height, out_width], align_corners=True), name='resize_input_img')(orig_img)
    # im_array_out = Lambda(lambda x: get_image_array(x, out_width, out_height), name='resize_input_img')(orig_img)
    gen_output = Lambda(lambda x: g_model(x), name='generator')(orig_img)
    stacked = concatenate([im_array_out, gen_output])
    d_model.trainable = False
    discrim_output = Lambda(lambda x: d_model(x), name='discriminator', trainable=False)(stacked)
    # gan_output = discrim_layer()(stacked)      # d_model(stacked)
    model = keras.Model(orig_img, discrim_output)
    # model = add_input_dims(model)  # use this or set to match g_model?
    return model


def make_gan_reg(g_model, d_model):
    orig_img = g_model.input
    gen_output = Lambda(lambda x: g_model(x), name='generator')(orig_img)
    d_model.trainable = False
    discrim_output = Lambda(lambda x: d_model(x), name='discriminator', trainable=False)(gen_output)
    model = keras.Model(orig_img, discrim_output)
    # model = add_input_dims(model)
    return model


def make_gen_iou(gen_model):
    gen_input = gen_model.get_input_at(0)  # bug causes multiple inbound nodes?
    gen_output = Lambda(lambda x: gen_model(x), name='generator')(gen_input)
    class_labels = Lambda(lambda x: tf.math.argmax(x, 2))(gen_output)  # axis reduced is 2
    new_gen = keras.Model(gen_input, class_labels)
    # new_gen.compile(loss='categorical_crossentropy', optimizer='adam',
    #                 metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.MeanIoU(num_classes=20)])
    new_gen.n_classes = gen_model.n_classes
    new_gen.input_height = gen_model.input_height
    new_gen.input_width = gen_model.input_width
    new_gen.output_height = gen_model.output_height
    new_gen.output_width = gen_model.output_width

    return new_gen

