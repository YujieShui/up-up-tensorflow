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
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
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

network.build(input_shape=[None, 28*28])
network.summary()

optimizer = optimizers.Adam(lr=0.01)

# Step1.Build a meter
acc_meter = metrics.Accuracy()
loss_meter = metrics.Mean()

for epoch in range(10): # iterate db for 10
    for step, (x, y) in enumerate(train_db, 1):
        # x:[128,28,28]
        # y:[128]

        # x:[128,28,28] => [128,784]
        x = tf.reshape(x, [-1,28*28])

        with tf.GradientTape() as tape:
            # x:[b,28*28]
            # h1 = x@w1+b1
            # [b,784]@[784,256]+[256] => [b,256] + [256] => []
            # h1 = tf.nn.relu(x@w1 + b1)
            # h2 = tf.nn.relu(h1@w2 + b2)
            # out = h2@w3 + b3
            #

            # [b,784] => [b,10]
            out = network(x)

            y_onehot = tf.one_hot(y, depth=10)

            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, out, from_logits=True))

            # Step2.Update data
            loss_meter.update_state(loss)

        # compute gradients
        # grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        grads = tape.gradient(loss, network.trainable_variables)
        optimizer.apply_gradients(zip(grads, network.trainable_variables))

        if step % 100 == 0:
            # Step3.Get Average data
            print(step, 'loss:', loss_meter.result().numpy())
            # Clear buffer
            loss_meter.reset_states()

    total_number = 0
    total_correct = 0
    for step, (x, y) in enumerate(test_db):
        acc_meter.reset_states()

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x,[-1,28*28])
        # [b, 784] => [b, 256] => [b, 128] => [b, 10]
        # h1 = tf.nn.relu(x@w1 + b1)
        # h2 = tf.nn.relu(h1@w2 + b2)
        # out = h2@w3 + b3

        out = network(x)

        # [b, 10] ~ R
        # [b, 10] ~ [0,1]
        prob = tf.nn.softmax(out, axis=1)
        # [b, 10] => [b]
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        # [b] int32
        # print(y.dtype, preb.dtype)
        # correct = tf.cast(tf.equal(y, pred), dtype=tf.int32)
        # correct = tf.reduce_sum(correct)
        # total_correct += int(correct)
        # total_number += x.shape[0]
        acc_meter.update_state(y, pred)

    # acc = total_correct / total_number
    print(step, 'Evaluate Acc:', acc_meter.result().numpy())



