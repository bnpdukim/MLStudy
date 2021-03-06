import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # for reproducibility

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# learning
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val)


# predict
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

# for step in range(2001):
#     cost_val, W_val, b_val, _ = \
#         sess.run([cost, W, b, train],
#             feed_dict={X: [1, 2, 3, 4, 5],
#                         Y: [2.1, 3.1, 4.1, 5.1, 6.1]}
#                  )
#     if step % 20 == 0:
#         print("step : ", step, ", cost : ", cost_val, ", weight : ", W_val, ", bias : ",b_val)
#
#
# print(sess.run(hypothesis, feed_dict={X: [5]}))
# print(sess.run(hypothesis, feed_dict={X: [2.5]}))
# print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))
#
plt.plot([1, 2, 3, 4, 5], [2.1, 3.1, 4.1, 5.1, 6.1])
plt.show()


# a = [3,4] shape 2  rank 1
# b = [[4,3],[1,2],[15,6]] shape 3,  2   rank 2
# c = [[[4,3],[1,2],[15,6]]] shape 1,3,2 rank 3
