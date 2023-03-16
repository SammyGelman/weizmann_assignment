import numpy as np
import tensorflow as tf

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def generate_datasets(circles_filename, rectangle_filename, percent_train=0.8):
    circles = np.load(circles_filename)
    rectangles = np.load(rectangle_filename)

    #set length of training data
    train_length = int(np.floor(
        len(circles[circles.files[0]]) * percent_train))

    #Load training data and labels
    training_circles_data = circles[circles.files[0]][0:train_length]
    training_circles_labels = circles[circles.files[1]][0:train_length]

    training_rectangle_data = rectangles[rectangles.files[0]][0:train_length]
    training_rectangle_labels = rectangles[rectangles.files[1]][0:train_length]

    #concatenate training data
    training_data = np.concatenate(
        (training_circles_data, training_rectangle_data))
    training_labels = np.concatenate(
        (training_circles_labels, training_rectangle_labels))

    #shuffle_data
    shuffled_training_data, shuffled_training_labels = unison_shuffled_copies(training_data, training_labels)
    #Load testing data and labels
    testing_circles_data = circles[circles.files[0]][train_length:]
    testing_circles_labels = circles[circles.files[1]][train_length:]

    testing_rectangle_data = rectangles[rectangles.files[0]][train_length:]
    testing_rectangle_labels = rectangles[rectangles.files[1]][train_length:]

    #concatenate testing data
    testing_data = np.concatenate(
        (training_circles_data, training_rectangle_data))
    testing_labels = np.concatenate(
        (training_circles_labels, training_rectangle_labels))

    #shuffle data
    shuffled_testing_data, shuffled_testing_labels = unison_shuffled_copies(testing_data, testing_labels)

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (training_data, training_labels))
    # test_dataset = tf.data.Dataset.from_tensor_slices(
    #     (testing_data, testing_labels))
    #
    # return train_dataset, test_dataset
    return shuffled_training_data, shuffled_training_labels, shuffled_testing_data, shuffled_testing_labels 
