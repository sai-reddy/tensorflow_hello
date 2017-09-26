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

# define training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# initialize variables for session
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

# train the model
for i in range(10000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))