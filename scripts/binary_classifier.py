import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from generate_datasets import generate_datasets
from classifier_plotting import plot_image, plot_value_array

# load dataset
circle_filename = "../data/circle_data.npz"
rectangle_filename = "../data/rectangle_data.npz"

training_data, training_labels, testing_data, testing_labels = generate_datasets(circle_filename, rectangle_filename)

#build model
#simple single layered model with output layer of length two, signiling the binary class estimate
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=training_data[0].shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

#complile the model with standard optimizer and crossentropy loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#fit the model to the training data
nepochs = 15
model.fit(training_data, training_labels, epochs=nepochs)

#evaluate model performance on test data
test_loss, test_acc = model.evaluate(testing_data, testing_labels, verbose=2)

print("\nTesting accuracy: ", test_acc)

#Use a softmax function to convert the final layer of the model into easily readable probabilites
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(testing_data)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], testing_labels, testing_data)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], testing_labels)
plt.tight_layout()
plt.show()


