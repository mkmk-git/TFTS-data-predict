# coding: utf-8
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader
np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser()


def main(_):
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 100) + x / 200. + noise
    plt.plot(x, y)
    plt.savefig('timeseries_y.jpg')

    data = {
        tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
        tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
    }

    reader = NumpyReader(data)

    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
        reader, batch_size=16, window_size=40)

    ar = tf.contrib.timeseries.ARRegressor(
        periodicities=200, input_window_size=30, output_window_size=10,
        num_features=1,
        loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

    ar.train(input_fn=train_input_fn, steps=6000)

    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
    evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

    (predictions,) = tuple(ar.predict(
        input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
            evaluation, steps=250)))



	
print(sess.run(predictions))
	
	
	

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
