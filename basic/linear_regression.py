# supervised learning

# y = ax + b linner model

import numpy as np

x_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([11000, 13100, 15100, 16200, 17200], dtype=np.float32)

# min-max scaling normalization
x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

# gradient descent update
# x^(t+1) = x^(t) - r*grad_f(x^(t))
# r: learning rate

# w
a, b = 0, 0

num_epoch = 10000
r = 5e-4

for e in range(num_epoch):
    # grad
    y_pred = a * x + b

    grad_a, grad_b = 2 * (y_pred - y).dot(x), 2 * (y_pred - y).sum()

    # update w
    a, b = a - r * grad_a, b - r * grad_b

print(a, b)


# use tf

import tensorflow as tf

X = tf.constant(x)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y)) # MSE
    
    grads = tape.gradient(loss, variables)
    # grads = [grad_a, grad_b]
    # grads_and_vars = [(grad_a, a), (grad_b, b)]
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)