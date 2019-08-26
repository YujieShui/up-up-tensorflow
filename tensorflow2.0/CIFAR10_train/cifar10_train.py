import tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x,y):
    # [0~255] => [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


batch_size = 128
(x, y), (x_val, y_val) = datasets.cifar10.load_data()

# x: (50000, 32, 32, 3)
# y: (50000, 1)
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10)
y_val = tf.one_hot(y_val, depth=10)
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())


train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(10000).map(preprocess).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batch_size)

sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)


# create own network
class MyDense(layers.Layer):
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x


class MyNetwork(keras.Model):

    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)


    def call(self, inputs, training=None, mask=None):
        '''

        :param inputs: [b, 32, 32,3]
        :param training:
        :param mask:
        :return:
        '''
        x = tf.reshape(inputs, [-1, 32*32*3])
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = self.fc5(x)

        return x


# train and evaluate
network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

network.evaluate(test_db)


# save and load weights
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')
network.evaluate(test_db)


