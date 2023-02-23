import tensorflow as tf


def augment(image):
    h, w, c = image.shape # (#batch, h, w, 3)
    image = tf.image.random_crop(image, (h-2, w-2, c))
    image = tf.image.resize(image, (h, w))
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image

def augment_fn(image, label):
    image = augment(image)
    return image, label

def lookup_closure(x_train):
    def lookup_fn(z_train):
        i = tf.squeeze(z_train)
        image = tf.gather(x_train, i)
        return image, z_train
    return lookup_fn
