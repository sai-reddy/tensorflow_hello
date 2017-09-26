import tensorflow as tf

#initialize
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# create model
model = W * x + b

# define loss
squared_deltas = tf.square(model - y)
loss = tf.reduce_sum(squared_deltas)

# fix W and b
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

# initialize variables for session
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
sess.run([fixW, fixb])

# run session
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))