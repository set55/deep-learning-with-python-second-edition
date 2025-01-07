import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


input_var = tf.Variable(initial_value=[[2.,3.],[3.,2.]])
a = tf.square(input_var)
with tf.GradientTape() as tape:
   result = tf.square(input_var)
gradient = tape.gradient(result, input_var)

print('gradient: ', gradient)


# Listing 3.11 Using GradientTape with constant tensor inputs
input_const = tf.constant(3.) 
with tf.GradientTape() as tape:
   tape.watch(input_const)
   result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
print('3.11 gradient: ', gradient)

# Listing 3.12 Using nested gradient tapes to compute second-order gradients
time = tf.Variable(0.) 
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position =  4.9 * time ** 2 
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)
print('3.12 acceleration: ', acceleration)




# Implement a linear classifier from scratch in TensorFlow
# Listing 3.13 Generating two classes of random points in a 2D plane
num_samples_per_class = 1000 
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)

# Listing 3.14 Stacking the two classes into an array with shape (2000, 2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
print('inputs: ', inputs)

# Listing 3.15 Generating the corresponding target label (0 and 1)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))
print('targets: ', targets)

# Listing 3.16 Plotting the two point classes (see figure 3.6)
# plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
# plt.show()

# Listing 3.17 Creating the linear classifier variables
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# Listing 3.18 The forward pass function
def model(inputs):
    return tf.matmul(inputs, W) + b

# Listing 3.19 The mean squared error loss function
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

# Listing 3.20 The training step function
learning_rate = 0.1 
  
def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

# Listing 3.21 The training loop
for step in range(40):
    loss = training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

predictions = model(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()


x = np.linspace(-1, 4, 100)
y = - W[0] /  W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
