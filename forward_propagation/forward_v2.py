'''
前向传播v1
1. numpy => tensor => datasets
2. 定义 [w1, b1, w2, b2, w3, b3]
3. 前向传播，计算loss

前向传播v2
1. 添加测试功能
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60000,28,28], x_test:[10000,28,28]
# y: [60000], y_test:[10000]
(x, y), (x_text, y_text) = datasets.mnist.load_data()

# convert to tensor
# x:[0~255] => [0~1]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

x_text = tf.convert_to_tensor(x_text, dtype=tf.float32) / 255.
y_text = tf.convert_to_tensor(y_text, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(y))
print(tf.reduce_min(y), tf.reduce_max(y))


# tensor => datasets
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_text,y_text)).batch(128)

train_iter = iter(train_db)
sample = next(train_iter)
print("batch:", sample[0].shape, sample[1].shape)



# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# w:[dim_in, dim_out]; b:[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

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
            h1 = tf.nn.relu(x@w1 + b1)
            h2 = tf.nn.relu(h1@w2 + b2)
            out = tf.nn.relu(h2@w3 + b3)

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # 均方误差 mse = mean(sum(y-out)^2)
            # loss:[b,10]
            loss = tf.square(y_onehot-out)
            # loss:[b,10] => scalar
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch+1, (epoch+1)*step, 'loss:', float(loss))

        # Q: 对模型进行测试，测试的是什么呢？
        # A: 需要在测试集上，使用每一个更新之后的 w 和 b，将输入映射为一个输出
        # 再将输出和 ground true 进行比较，就得到了测试结果

    total_number = 0
    total_correct = 0
    for step, (x, y) in enumerate(test_db):

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x,[-1,28*28])
        # [b, 784] => [b, 256] => [b, 128] => [b, 10]
        h1 = tf.nn.relu(x@w1 + b1)
        h2 = tf.nn.relu(h1@w2 + b2)
        out = h2@w3 + b3

        # [b, 10] ~ R
        # [b, 10] ~ [0,1]
        prob = tf.nn.softmax(out, axis=1)
        # [b, 10] => [b]
        preb = tf.argmax(prob, axis=1)
        preb = tf.cast(preb, dtype=tf.int32)
        # [b] int32
        # print(y.dtype, preb.dtype)
        correct = tf.cast(tf.equal(y, preb), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x.shape[0]

    acc = total_correct / total_number
    print("accuracy:", acc)


