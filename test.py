import os
import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope

# State your log directory where you can retrieve your model
log_dir = 'log'
dataset_dir = 'flowers/flower_photos/sunflowers'

# Get the latest checkpoint file
checkpoint_filename = tf.train.latest_checkpoint(log_dir)

image_size = 299


labels_file = 'flowers/labels.txt'
labels = open(labels_file, 'r')

# Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1]  # Remove newline
    labels_to_name[int(label)] = string_name


def main():
    with tf.Graph().as_default() as graph:
        # image processing graph
        testImage_str = tf.placeholder(tf.string)
        testImage = tf.image.decode_jpeg(testImage_str, channels=3)
        image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        images = tf.expand_dims(image, 0)

        # inception_resnet graph
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(images, num_classes=5, is_training=False)

        # Restoring saved session
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        session = tf.Session()
        saver.restore(session, checkpoint_filename)

        # Starting session
        with session.as_default() as sess:
            for image_file in os.listdir(dataset_dir):
                filepath = os.path.join(dataset_dir, image_file)
                testImage_string = tf.gfile.FastGFile(filepath, 'rb').read()
                end = sess.run(end_points, feed_dict={testImage_str: testImage_string})
                pred_prob = end['Predictions']
                label = labels_to_name[np.argmax(pred_prob)]
                print(pred_prob, label)

if __name__ == '__main__':
    main()
