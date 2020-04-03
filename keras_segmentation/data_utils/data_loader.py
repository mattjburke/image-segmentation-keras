import glob
import itertools
import os
import random
import six
import numpy as np
import cv2
import keras
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")
    def tqdm(iter):
        return iter


from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg

DATA_LOADER_SEED = 0
FAKE = 0
REAL = 1

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]


class DataLoaderError(Exception):
    pass


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png" , ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError("Segmentation file with filename {0} already exists and is ambiguous to resolve with path {1}. Please remove or rename the latter.".format(file_name, os.path.join(segs_path, dir_entry)))
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation found for image {0}.".format(image_full_path))

    return return_value


def get_image_array(image_input, width, height, imgNorm="sub_mean",
                  ordering='channels_first'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif  isinstance(image_input, six.string_types)  :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}".format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nClasses, width, height, no_reshape=True):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types) :
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_segmentation_array: Can't process input type {0}".format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))  # The source of the error!

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path, n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: {0} and segmentations path: {1}".format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print("The size of image {0} and its segmentation {1} doesn't match (possibly the files are corrupt).".format(im_fn, seg_fn))
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} violating range [0, {1}]. Found maximum pixel value {2}".format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            X.append(get_image_array(im, input_width,
                                   input_height, ordering=IMAGE_ORDERING))
            Y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)


def image_segmentation_pairs_generator(images_path, segs_path, batch_size, gen_model, do_augment=False):

    n_classes = gen_model.n_classes
    input_height = gen_model.input_height
    input_width = gen_model.input_width
    output_height = gen_model.output_height
    output_width = gen_model.output_width

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    # seq = keras.utils.Sequence()

    while True:
        X = []
        Y = []
        i = 0
        for pair in range(batch_size):
            # print("pair =", pair)
            im, seg = next(zipped)
            i += 1
            use_fake = i % 2  # use Math.rand() instead?

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)
            # print("im shape = ", im.shape)
            # print("seg shape = ", seg.shape)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            im_array_in = get_image_array(im, input_width, input_height, ordering=IMAGE_ORDERING)
            # im_array_out = get_image_array(im, output_width, output_height, ordering=IMAGE_ORDERING)
            # make sure it is resized the same way gan resizes it
            im_tensor_in = tf.convert_to_tensor(im_array_in)
            im_tensor_out = tf.compat.v1.image.resize(im_tensor_in, [output_height, output_width], align_corners=True)
            im_array_out = np.array(im_tensor_out)
            # print("im_array_out shape = ", im_array_out.shape)

            if use_fake == 1:
                seg_array = gen_model.predict([[im_array_in]])[0]
                # print("seg_array fake1 shape = ", seg_array.shape)
                # seg_array = get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
                # print("seg_array fake2 shape = ", seg_array.shape)
                Y.append(FAKE)
            else:
                seg_array = get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
                # print("seg_array real shape = ", seg_array.shape)
                Y.append(REAL)

            stacked = np.dstack((im_array_out, seg_array))  # stacks along 3rd axis
            X.append(stacked)

        yield np.array(X), np.array(Y)


def train_on_image_seg_batches(d_model, epochs, images_path, segs_path, batch_size,
                                       n_classes, input_height, input_width,
                                       output_height, output_width, gen_model,
                                       do_augment=False):
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    # seq = keras.utils.Sequence()

    use_fake = True
    # while True:
    for i in range(0, epochs):
        X = []
        Y = []
        use_fake = not use_fake
        print("epoch, use_fake =", i, use_fake)
        for pair in range(batch_size):
            # print("pair =", pair)
            im, seg = next(zipped)
            # i += 1
            # use_fake = i % 2  # use Math.rand() instead?

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)
            # print("im shape = ", im.shape)
            # print("seg shape = ", seg.shape)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            im_array_in = get_image_array(im, input_width, input_height, ordering=IMAGE_ORDERING)
            im_array_out = get_image_array(im, output_width, output_height, ordering=IMAGE_ORDERING)
            # print("im_array_out shape = ", im_array_out.shape)

            if use_fake:
                seg_array = gen_model.predict([[im_array_in]])
                seg_array = seg_array[0]
                # print("seg_array fake1 shape = ", seg_array.shape)
                # seg_array = get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
                # print("seg_array fake2 shape = ", seg_array.shape)
                Y.append(FAKE)
            else:
                seg_array = get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
                # print("seg_array real shape = ", seg_array.shape)
                Y.append(REAL)

            stacked = np.dstack((im_array_out, seg_array))  # stacks along 3rd axis
            X.append(stacked)

        # yield np.array(X), np.array(Y)
        X_batch = np.array(X)
        Y_batch = np.array(Y)
        loss_acc = d_model.train_on_batch(X_batch, Y_batch)

        # d_model.fit(X_batch, Y_batch,
        #             validation_data=(X_val, Y_val),
        #             epochs=epochs,
        #             batch_size=5, steps_per_epoch=595, validation_steps=100,  # there are 2975 train, 500 val
        #             use_multiprocessing=False,  # Used for generator or keras.utils.Sequence input only
        #             callbacks=[csv_logger, save_chckpts])


def image_segmentation_pairs_dataset(images_path, segs_path, gen_model, do_augment=False):

    n_classes = gen_model.n_classes
    g_input_height = gen_model.input_height
    g_input_width = gen_model.input_width
    g_output_height = gen_model.output_height
    g_output_width = gen_model.output_width

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)
    # seq = keras.utils.Sequence()

    # while True:
    X = []
    Y = []
    i = 0
    for pair in zipped:
        # print("pair =", pair)
        im, seg = next(zipped)
        i += 1
        use_fake = i % 2  # use Math.rand() instead?

        im = cv2.imread(im, 1)
        seg = cv2.imread(seg, 1)
        # print("im shape = ", im.shape)
        # print("seg shape = ", seg.shape)

        if do_augment:
            im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

        im_array_in = get_image_array(im, g_input_width, g_input_height, ordering=IMAGE_ORDERING)
        im_array_out = get_image_array(im, g_output_width, g_output_height, ordering=IMAGE_ORDERING)
        # print("im_array_out shape = ", im_array_out.shape)

        if use_fake == 1:
            seg_array = gen_model.predict([[im_array_in]])
            seg_array = seg_array[0]
            # print("seg_array fake1 shape = ", seg_array.shape)
            # seg_array = get_segmentation_array(seg, n_classes, output_width, output_height, no_reshape=True)
            # print("seg_array fake2 shape = ", seg_array.shape)
            Y.append(FAKE)
        else:
            seg_array = get_segmentation_array(seg, n_classes, g_output_width, g_output_height, no_reshape=True)
            # print("seg_array real shape = ", seg_array.shape)
            Y.append(REAL)

        stacked = np.dstack((im_array_out, seg_array))  # stacks along 3rd axis
        X.append(stacked)

        # yield np.array(X), np.array(Y)

    return X, Y


def image_flabels_generator(images_path, segs_path, batch_size, input_height, input_width, do_augment=False):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            # seg = cv2.imread(seg, 1)

            if do_augment:
                im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

            X.append(get_image_array(im, input_width, input_height, ordering=IMAGE_ORDERING))
            #Y.append(get_segmentation_array(seg, n_classes, output_width, output_height))
            # Y.append(FAKE)  # 0 means fake
            Y.append(REAL)  # We want the generator to be updated so that the discriminator thinks the generated images are REAL
            # labelling generated images as REAL for the gan phase of training is a way to do this.
            # The desired function of the gan is to produce images that are then classified as real, even though they are not

        yield np.array(X), np.array(Y)


# def input_fn(mode, params):
#     assert 'batch_size' in params
#     assert 'noise_dims' in params
#     bs = params['batch_size']
#     nd = params['noise_dims']
#     split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
#     shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
#     just_noise = (mode == tf.estimator.ModeKeys.PREDICT)
#
#     noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
#                 .map(lambda _: tf.random_normal([bs, nd])))
#
#     if just_noise:
#         return noise_ds
#
#     def _preprocess(element):
#         # Map [0, 255] to [-1, 1].
#         images = (tf.cast(element['image'], tf.float32) - 127.5) / 127.5
#         return images
#
#     images_ds = (tfds.load('mnist', split=split)
#                  .map(_preprocess)
#                  .cache()
#                  .repeat())
#     if shuffle:
#         images_ds = images_ds.shuffle(
#             buffer_size=10000, reshuffle_each_iteration=True)
#     images_ds = (images_ds.batch(bs, drop_remainder=True)
#                  .prefetch(tf.data.experimental.AUTOTUNE))
#
#     return tf.data.Dataset.zip((noise_ds, images_ds))

