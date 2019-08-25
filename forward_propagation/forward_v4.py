'''
前向传播v1
1. numpy => tensor => datasets
2. 定义 [w1, b1, w2, b2, w3, b3]
3. 前向传播，计算loss

前向传播v2
1. 添加测试功能

前向传播v3
1. 使用 tf.Sequential 构建模型
2. 使用 tf.keras.metrics 指标来简化模型评过程
3. 将数据预处理过程封装成 preprocess(x,y)
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    """
    x is a simple image, not a batch
    """
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


# x: [60000,28,28], x_test:[10000,28,28]
# y: [60000], y_test:[10000]
(x, y), (x_text, y_text) = datasets.mnist.load_data()

# tensor => datasets
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(60000).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_text,y_text))
test_db = test_db.map(preprocess).shuffle(10000).batch(128)

# 使用 tf.Sequential 定义上面的这些参数
network = Sequential([layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
            ])

network.build(input_shape=(None, 28*28))
network.summary()

network.compile(optimizer=optimizers.Adam(lr=0.01),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

network.fit(train_db, epochs=10, validation_data=test_db, validation_freq=2)

network.evaluate(test_db)


sample = next(iter(ds_val))
x = sample[0]
y = sample[1] # one-hot
pred = network.predict(x) # [b, 10]
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)