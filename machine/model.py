import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

from global_vars import *
import data

# Custom GateLayer to modify x_boxes and x_cls based on x_prob


class GateLayer(layers.Layer):
    def call(self, inputs):
        x_prob, x_boxes, x_cls = inputs
        gate = tf.where(x_prob > 0.5, tf.ones_like(
            x_prob), tf.zeros_like(x_prob))
        x_boxes = x_boxes * gate
        x_cls = x_cls * gate
        return x_boxes, x_cls


idx_p = [0]
idx_bb = [1, 2, 3, 4]


@tf.function
def loss_bb(y_true, y_pred):
    y_true = tf.gather(y_true, idx_bb, axis=-1)
    y_pred = tf.gather(y_pred, idx_bb, axis=-1)
    return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred))


@tf.function
def loss_p(y_true, y_pred):
    y_true = tf.gather(y_true, idx_p, axis=-1)
    y_pred = tf.gather(y_pred, idx_p, axis=-1)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))


@tf.function
def loss_cls(y_true, y_pred):
    y_true = tf.gather(y_true, data.idx_cls, axis=-1)
    y_pred = tf.gather(y_pred, data.idx_cls, axis=-1)
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))


@tf.function
def loss_func(y_true, y_pred):
    loss_bounding_boxes = loss_bb(y_true, y_pred)
    loss_presence = loss_p(y_true, y_pred)
    loss_classification = loss_cls(y_true, y_pred)
    w1 = 1
    w2 = 2
    w3 = 2
    total_loss = w1 * loss_bounding_boxes + w2 * \
        loss_presence + w3 * loss_classification
    return total_loss


def make_model():
    x_input = layers.Input(
        shape=(PROCESSED_IMG_WIDTH, PROCESSED_IMG_HEIGHT, 3))

    # Define the convolutional layers
    x = layers.Conv2D(32, kernel_size=3, padding='same',
                      activation='relu')(x_input)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)

    # Define the outputs
    x_prob = layers.Conv2D(1, kernel_size=3, padding='same',
                           activation='sigmoid', name='x_prob')(x)
    x_boxes = layers.Conv2D(
        4, kernel_size=3, padding='same', name='x_boxes')(x)
    x_cls = layers.Conv2D(data.NUM_CLASSES, kernel_size=3, padding='same',
                          activation='sigmoid', name='x_cls')(x)

    # Apply GateLayer
    gate_layer = GateLayer()
    x_boxes, x_cls = gate_layer([x_prob, x_boxes, x_cls])

    x = layers.Concatenate()([x_prob, x_boxes, x_cls])
    model = tf.keras.models.Model(x_input, x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.003)
    model.compile(optimizer=opt, loss=loss_func, metrics=['accuracy'])
    return model
