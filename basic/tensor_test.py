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


# muti differential
# matrix x y
# L = |wx + b + y|^2
x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[10.], [20.]])
w = tf.Variable(initial_value=[[1.], [10.]])
b = tf.Variable(initial_value=3.)

with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(x, w) + b + y))

w_grad, b_grad = tape.gradient(L, [w, b])

print(L, w_grad, b_grad)


# reduce d sum
print('reduce d sum')
x = tf.constant([[1, 1, 1], [1, 1, 1]])
print('1.', x.numpy())
print('2.', tf.reduce_sum(x, 0).numpy())
print('3.', tf.reduce_sum(x, 1).numpy())
print('4.', tf.reduce_sum(x, 1, keepdims=True).numpy())