# coding: utf-8
from __future__ import print_function
import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
#np.set_printoptions(threshold=np.inf)

def main(_):
    csv_file_name = './data/period_trend.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=64, window_size=80)
    with tf.Session() as sess:
        data = reader.read_full()
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        data = sess.run(data)
        coord.request_stop()

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=1800, input_window_size=50, output_window_size=30,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    ar.train(input_fn=train_input_fn, steps=100)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=2500)))


			
    plt.figure(figsize=(15, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1))
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1))
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1))
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.savefig('predict_result.png')			
			

    print((predictions,))			


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
