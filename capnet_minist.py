import tensorflow as tf
import numpy as np

import subprocess
import os

from tensorflow.examples.tutorials.mnist import input_data


def cap_net(data_input, n_output, n_route=3, n_primary_cap=32, route_bias_size=(32 * 6 * 6, )):

    with tf.name_scope('Conv1') as conv1_scope:
        filter_1 = tf.get_variable(shape=(9, 9, 1, 256), dtype=tf.float32, name='conv1_kernel',
                                   initializer=tf.glorot_normal_initializer())
        tf.summary.histogram('Conv1_filter', filter_1)
        conv1 = tf.nn.conv2d(data_input,
                             filter=filter_1,
                             strides=[1, 1, 1, 1],
                             padding='VALID',
                             name='conv1')

    with tf.name_scope("Primary_Caps") as p_scope:
        # primary capsule
        filter_2 = tf.get_variable(name='Primary_kernel', shape=(9, 9, 256, 256),
                                   dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        tf.summary.histogram('primary_kernel_weights', filter_2)
        cap_output = tf.nn.conv2d(conv1,
                                  filter=filter_2,
                                  strides=[1, 2, 2, 1],
                                  padding='VALID')
        # shape (None, 32, 36, 8)
    caps_u = tf.reshape(cap_output, shape=(-1, 32, 36, 8))

    # dynamic routing
    digit_caps_list = []
    bias_size = [tf.shape(caps_u)[0], 1152]

    with tf.name_scope('Digit_Caps') as d_scope:
        for idx_output in range(n_output):
            digit_weights = tf.get_variable(name='{}_digit_weights'.format(idx_output),
                                            shape=(n_primary_cap, 1, 8, 16), dtype=tf.float32,
                                            initializer=tf.glorot_uniform_initializer(),
                                            trainable=True)
            tf.summary.histogram('{}_digit_caps_weight'.format(idx_output), digit_weights)
            digit_weights_tile = tf.tile(digit_weights, multiples=[1, 36, 1, 1])
            # shape (None, 32, 36, 8) * (32, 36, 8, 16) = None, 32, 36, 16
            digit_caps = tf.einsum('bijk,ijkm->bijm', caps_u, digit_weights_tile)
            digit_caps_input = tf.reshape(digit_caps, shape=(-1, 1152, 16))

            bias = tf.zeros(shape=bias_size, name='{}_bias'.format(idx_output))
            for i in range(n_route):

                # shape 1152, 1
                cap_c = tf.expand_dims(tf.nn.softmax(bias), -1)
                # shape None, 1, 16
                cap_s = tf.reduce_sum(tf.multiply(digit_caps_input, cap_c), axis=1, keepdims=True)
                # shape None, 1, 1
                cap_s_l2 = tf.reduce_sum(tf.square(cap_s), axis=2, keepdims=True)
                cap_v_1 = tf.divide(cap_s_l2, tf.add(tf.constant(1, dtype=tf.float32),
                                                     cap_s_l2))
                cap_v_2 = tf.divide(cap_v_1, tf.sqrt(cap_s_l2))
                # shape None, 1, 16
                cap_v = tf.multiply(cap_s, cap_v_2)
                # shape None, 1152
                delta_bias = tf.reduce_sum(tf.multiply(digit_caps_input, cap_v), axis=2)
                # shape None, 1152
                bias = tf.add(bias, delta_bias)

            tf.summary.scalar('{}_routing_bias'.format(idx_output), tf.argmax(bias[0]))
            digit_caps_list.append(cap_v)
    digit_caps = tf.concat(digit_caps_list, axis=1)
    return digit_caps


def loss_function(output, label, m_p=0.9, m_n=0.1, loss_lambda=0.5):
    m_p_tensor = tf.constant(m_p, dtype=tf.float32)
    m_n_tensor = tf.constant(m_n, dtype=tf.float32)
    lambda_tensor = tf.constant(loss_lambda, dtype=tf.float32)

    output_l2 = tf.sqrt(tf.reduce_sum(tf.square(output), axis=2))
    loss_first = tf.multiply(label, tf.square(tf.nn.relu(m_p_tensor - output_l2)))

    loss_second_1 = tf.multiply(lambda_tensor, tf.subtract(tf.constant(1, dtype=tf.float32), label))
    loss_second = tf.multiply(loss_second_1,
                              tf.square(tf.nn.relu(tf.subtract(output_l2, m_n_tensor))))

    loss = tf.reduce_mean(tf.reduce_sum(tf.add(loss_first, loss_second), axis=1))
    tf.summary.scalar('Total_loss', loss)
    return loss


def train():
    minist = input_data.read_data_sets('./data/minist', one_hot=True)

    n_iter = 10000
    batch_size = 128
    n_batch = minist.train.num_examples // batch_size
    init_lr = 0.001
    decay_rate = 0.9
    decay_iter = 1

    save_dir = 'cap_ckpt'

    input_hold = tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32, name='Data_input')
    label_hold = tf.placeholder(shape=(None, 10), dtype=tf.float32, name='Label_input')

    model_output = cap_net(input_hold, 10)

    loss = loss_function(model_output, label_hold)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=init_lr, global_step=global_step,
                                               decay_steps=decay_iter * n_batch,
                                               decay_rate=decay_rate, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)

    saver = tf.train.Saver(max_to_keep=5)
    if os.path.exists(save_dir):
        cmd = ['rm', '-rf', '{}/*'.format(save_dir)]
        subprocess.call(cmd)
    else:
        os.mkdir(save_dir)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('log', sess.graph)
        sess.run(init)

        min_loss = np.inf

        for idx_iter in range(n_iter):
            total_loss = 0
            total_acc = 0
            for idx_batch in range(n_batch):
                data, label = minist.train.next_batch(batch_size)

                _, batch_loss, batch_output, rs = sess.run(fetches=[train_op, loss, model_output, merged],
                                         feed_dict={input_hold: data.reshape(-1, 28, 28, 1),
                                                    label_hold: label})
                writer.add_summary(rs, idx_iter * n_batch + idx_batch)

                pred = np.argmax(np.sqrt(np.sum(np.square(batch_output), axis=2)), axis=-1)
                truth = np.argmax(label, axis=-1)
                batch_acc = np.sum(np.equal(pred, truth))

                if idx_batch == 0:
                    print('Ground truth: {}'.format(truth[0]))
                    print('Prediction: {}'.format(np.sqrt(np.sum(np.square(batch_output[0]), axis=1))))

                total_loss += batch_loss
                total_acc += batch_acc

                # batch_out = sess.run(model_output,
                #                      feed_dict={input_hold: data.reshape(-1, 28, 28, 1)})
                # print(batch_out.shape)

            mean_loss = total_loss / n_batch
            mean_acc = total_acc / (n_batch * batch_size) * 100

            print('Iter: {}, training loss: {}, training acc: {:.2f}%'.format(idx_iter, mean_loss, mean_acc))
            if mean_loss < min_loss:
                min_loss = mean_loss
                saver.save(sess, '{}/capnet.ckpt'.format(save_dir))

        writer.close()


def main():
    train()


if __name__ == '__main__':
    main()

