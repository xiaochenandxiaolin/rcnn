from tensorflow.keras import layers
import tensorflow as tf


class _ResBlock(tf.keras.Model):

    def __init__(self, filters, block,
                 downsampling=False, stride=1, **kwargs):
        super(_ResBlock, self).__init__(**kwargs)

        filters1 = filters
        filters3 = filters
        conv_name_base = 'res' + block + '_branch'
        bn_name_base = 'bn' + block + '_branch'

        self.downsampling = downsampling
        self.stride = stride
        self.out_channel = filters3

        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=(1, 1),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(bn2a, (3, 3), padding='same',
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(bn2b, (1, 1),
                                    kernel_initializer='he_normal',
                                    name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(filters3, (1, 1),
                                               strides=(stride, stride),
                                               kernel_initializer='he_normal',
                                               name=conv_name_base + '1')
        self.bn_shortcut = layers.BatchNormalization(name=bn_name_base + '1')

    def call(self, _inputs, training=False):
        x = self.conv2a(_inputs)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x, training=training)
        # x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(_inputs)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        x = tf.nn.relu(x)

        return x

class Grasp(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Grasp, self).__init__(**kwargs)
        
        # self._IMG = _IMG 
        self.grasp1a = _ResBlock([None,7, 7, 256], block='1a',      
                                 downsampling=True, stride=1)
        self.grasp1b = _ResBlock([None,7, 7, 256], block='1b',
                                 downsampling=True, stride=1)      
        self.grasp1c = _ResBlock([None,7, 7, 256], block='1b',
                                 downsampling=True, stride=1)

    def call(self, inputs, training=True):
        x = inputs
        grasp_x = self.grasp1a(x, training=training)
        grasp_x = self.grasp1b(x, training=training)
        grasp_x = self.grasp1c(x, training=training)
        return grasp_x
        

