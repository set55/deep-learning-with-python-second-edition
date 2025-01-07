# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print('train_images.ndim: ', train_images.ndim)
print('train_images.shape: ', train_images.shape)
print('train_images.dtype: ', train_images.dtype)

print('test images ndim: ', test_images.ndim)
print('test images shape: ', test_images.shape)
print('test images dtype: ', test_images.dtype)


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

print('train_images.ndim: ', train_images.ndim)
print('train_images.shape: ', train_images.shape)
print('train_images.dtype: ', train_images.dtype)

print('test images ndim: ', test_images.ndim)
print('test images shape: ', test_images.shape)
print('test images dtype: ', test_images.dtype)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5, batch_size=128)

