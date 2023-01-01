import tensorflow as tf

# a float value
random_f = tf.random.uniform(shape=())
print('->', random_f)

# two zeros
zero_v = tf.zeros(shape=(2))
zero_int_v = tf.zeros(shape=(2), dtype=tf.int32)
print('->', zero_v)
print('->', zero_int_v)

# vector
a = tf.constant([[1., 2.], [3., 4.]])
b = tf.constant([[3., 4.], [5., 6.]])

print(a.shape)
print(a.dtype)
print(a.numpy())

# operation
c = tf.add(a, b)
print(c)

d = tf.matmul(a, b)
print(d)

# differential
x = tf.Variable(5.)

with tf.GradientTape() as tape:
    y = tf.square(x) # x^2

y_grad = tape.gradient(y, x)

print('y = x^2 =', y, ', y\' =', y_grad, ', x =', x.numpy())