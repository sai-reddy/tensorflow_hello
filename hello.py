import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(3.0, dtype=tf.float32)

print(node1, node2)

node3 = tf.add(node1, node2)
print("node3: ", node3)

sess = tf.Session()
print("sess.run(node3)", sess.run(node3))