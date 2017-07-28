import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from naive_rnn import RNN_curve_predictor

if __name__ == '__main__':

    RNN_model = RNN_curve_predictor()
    RNN_model.n_steps = 1

    RNN_model.build_network()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess, 'models/RNN_curve_predictor-200000')

    r = np.arange(0, .34, 0.001)
    n_points = len(r)
    theta = 45 * np.pi * r
    x_offset, y_offset = .5, .5
    x_curve_points = r * np.cos(theta) + x_offset

    plt.ion()
    figure = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    init_lstm_state = tf.contrib.rnn.BasicLSTMCell(RNN_model.n_hidden_units).zero_state(RNN_model.batch_size,
                                                                                        tf.float32)
    for point_num in range(2 * n_points):

        current_x = np.array(x_curve_points[point_num % n_points])
        current_x = np.reshape(current_x, [1, 1, 1])

        if point_num == 0:
            feed_dict = {RNN_model.xs: current_x}
            pred_y, init_lstm_state_np = sess.run([RNN_model.pred_output,
                                                   RNN_model.lstm_state],
                                                  feed_dict=feed_dict)
        else:
            feed_dict = {RNN_model.xs: current_x,
                         RNN_model.state: init_lstm_state_np}
            pred_y, init_lstm_state_np = sess.run([RNN_model.pred_output,
                                                   RNN_model.lstm_state],
                                                  feed_dict=feed_dict)

        circle = plt.Circle((current_x, pred_y), 0.005, color='#ff6666')
        ax.add_artist(circle)
        plt.pause(0.0000001)
