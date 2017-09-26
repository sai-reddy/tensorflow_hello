import tensorflow as tf

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

adder_node = node1 + node2
add_and_triple = adder_node * 3

sess = tf.Session()
print(sess.run(adder_node, {node1: 1, node2: 2}))
print(sess.run(adder_node, {node1: [1, 2], node2: [1, 2]}))
print(sess.run(add_and_triple, {node1: 1, node2: 2}))