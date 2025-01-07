import tensorflow as tf
from tensorflow import keras
import numpy as np


# 3.6.1 Implementing custom layers
# Class Layer is the base class for all layers in Keras. A layer is a class implementing common neural networks operations, such as convolution, batch normalization, etc. These operations require managing weights, losses, updates, and inter-layer connectivity. All these are handled by Layer.
# def __call__(self, inputs):
#     if not self.built:
#          self.build(inputs.shape)
#          self.built = True
#     return self.call(inputs)
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
  
    def build(self, input_shape): # remember: when implementing your own layers, put the forward pass in the build() method
        input_dim = input_shape[-1]
        print('3.6.1@input_dim: ', input_dim)
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="random_normal")
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")
  
    def call(self, inputs): # remember: when implementing your own layers, put the forward pass in the call() method
        y = tf.matmul(inputs, self.W) + self.b
        print('3.6.1@y: ', y)
        if self.activation is not None:
            y = self.activation(y)
            print('3.6.1@activate y: ', y)
        return y
    


my_dense = SimpleDense(units=32, activation=tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = my_dense(input_tensor)
print(output_tensor.shape)


# 3.6.6 Monitoring loss and metrics on validation data
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])
  
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]
 
num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]
model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)

