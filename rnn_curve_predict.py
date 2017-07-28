import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.collections import LineCollection


def create_curve():
    # Create the curve which we want the RNN to learn
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    r = np.arange(0, .34, 0.001)
    n_points = len(r)
    theta = 45 * np.pi * r
    x_offset, y_offset = .5, .5
    y_curve_points = 1.4 * r * np.sin(theta) + y_offset
    x_curve_points = r * np.cos(theta) + x_offset
    curve = list(zip(x_curve_points, y_curve_points))
    collection = LineCollection([curve], colors='k')
    ax.add_collection(collection)
    return ax, n_points, x_curve_points, y_curve_points


def draw_pred_points(ax, batch_x, pred_points):
    circles = []
    for point_num in range(batch_x.shape[0]):
        circle = plt.Circle((batch_x[point_num], pred_points[point_num]), 0.005, color='#ff6666')
        ax.add_artist(circle)
        plt.pause(0.0000001)
        circles.append(circle)
    for circle in circles:
        circle.remove()


class RNN_curve_predictor():
    def __init__(self):
        # Hyper parameters
        self.init_learning_rate = 0.01
        self.batch_size = 1
        self.n_outputs = 1
        self.n_steps = 10
        self.n_hidden_units = 256
        self.decay_steps = 10000
        self.decay_rate = 0.5

    def prepare_training_data(self, x_curve_points, y_curve_points, n_points):
        # Shape curve points into training data
        training_data = np.array([x_curve_points.reshape([n_points, 1]), y_curve_points.reshape([n_points, 1])])
        training_data = np.transpose(training_data)[0]
        self.training_data = np.reshape(training_data, newshape=[-1, self.n_steps, 2])

    def build_network(self):
        self.xs = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.n_outputs], name="inputs")
        self.ys = tf.placeholder(tf.float32, [self.batch_size, self.n_steps, self.n_outputs], name="outputs")

        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_hidden_units)

        output_w = tf.get_variable(name='output_w',
                                   shape=[self.n_hidden_units, self.n_outputs],
                                   dtype=tf.float32)
        output_b = tf.get_variable(name='output_b',
                                   shape=[self.n_outputs],
                                   dtype=tf.float32)

        self.state = lstm_cell.zero_state(self.batch_size, tf.float32)
        with tf.name_scope('lstm'):
            lstm_output, self.lstm_state = tf.nn.dynamic_rnn(lstm_cell, self.xs,
                                                             initial_state=self.state,
                                                             dtype=tf.float32)

            lstm_output = tf.reshape(lstm_output, shape=[-1, self.n_hidden_units])
            self.pred_output = tf.matmul(lstm_output, output_w) + output_b

        # Calculate Loss
        flat_ys = tf.reshape(self.ys, shape=[-1, self.n_outputs])
        loss = tf.losses.mean_squared_error(self.pred_output, flat_ys)
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', self.loss)

        # Optimize
        self.gs = tf.contrib.framework.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(self.init_learning_rate,
                                             global_step=self.gs,
                                             decay_steps=self.decay_steps,
                                             decay_rate=self.decay_rate)
        tf.summary.scalar('learning rate', self.lr)
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss, global_step=self.gs)
        self.merged_summary = tf.summary.merge_all()


if __name__ == '__main__':

    train_iters = 100000
    draw_iters = 1000

    ax, n_points, x_curve_points, y_curve_points = create_curve()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RNN Curve Predictor')

    RNN_model = RNN_curve_predictor()
    RNN_model.prepare_training_data(x_curve_points, y_curve_points, n_points)
    RNN_model.build_network()

    saver = tf.train.Saver()

    # Init session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    board_writer = tf.summary.FileWriter(logdir='./logs', graph=sess.graph)

    saver.restore(sess, 'models/RNN_curve_predictor-300000')

    gs_np = 0
    lstm_state_np = None
    for train_iter in range(train_iters):

        # Prepare X inputs
        batch_num_start = train_iter % (n_points // RNN_model.n_steps)
        batch_num_end = batch_num_start + RNN_model.batch_size if batch_num_start + RNN_model.batch_size < \
                                                                  RNN_model.training_data.shape[0] else \
            RNN_model.training_data.shape[0]
        batch_x = RNN_model.training_data[batch_num_start:batch_num_end, :, 0, np.newaxis]
        batch_y = RNN_model.training_data[batch_num_start:batch_num_end, :, 1, np.newaxis]

        circle = plt.Circle((batch_x[0, 0, 0], batch_y[0, 0, 0]), 0.02, facecolor='#00ffff')
        ax.add_artist(circle)

        if train_iter == 0:
            feed_dict = {RNN_model.xs: batch_x,
                         RNN_model.ys: batch_y}
            _, loss_np, pred_np, \
            lr_np, gs_np, \
            lstm_state_np, summary_np = sess.run([RNN_model.train_op,
                                                  RNN_model.loss,
                                                  RNN_model.pred_output,
                                                  RNN_model.lr,
                                                  RNN_model.gs,
                                                  RNN_model.lstm_state,
                                                  RNN_model.merged_summary], feed_dict=feed_dict)
            board_writer.add_summary(summary_np, global_step=gs_np)
            print('Step: {:6}, lr: {:3.5f}, loss: {:3.5f}'.format(gs_np, lr_np, loss_np))

        else:
            feed_dict = {RNN_model.xs: batch_x,
                         RNN_model.ys: batch_y,
                         RNN_model.state: lstm_state_np}
            _, loss_np, pred_np, \
            lr_np, gs_np, \
            lstm_state_np, summary_np = sess.run([RNN_model.train_op,
                                                  RNN_model.loss,
                                                  RNN_model.pred_output,
                                                  RNN_model.lr,
                                                  RNN_model.gs,
                                                  RNN_model.lstm_state,
                                                  RNN_model.merged_summary], feed_dict=feed_dict)
            board_writer.add_summary(summary_np, global_step=gs_np)
            print('Step: {:6}, lr: {:3.5f}, loss: {:3.5f}'.format(gs_np, lr_np, loss_np))

        if (train_iter + 1) % draw_iters < 10:
            draw_pred_points(ax, batch_x[0], pred_np)
        circle.remove()

    saver.save(sess, './models/RNN_curve_predictor', global_step=gs_np)
    print('Model saved.')
