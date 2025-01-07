import tensorflow as tf
import numpy as np

# Listing 3.1 All-ones or all-zeros tensors
x = tf.ones(shape=(2,1)) #Equivalent to np.ones(shape=(2, 1))
print('x: ', x)

x = tf.zeros(shape=(2,1)) #Equivalent to np.zeros(shape=(2, 1))
print('x: ', x)


# Listing 3.2 Random tensors
# Tensor of random values drawn from a normal distribution with mean 0 and standard deviation 1. Equivalent to np.random.normal(size=(3, 1), loc=0., scale=1.).
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
print('x: ', x)

# Tensor of random values drawn from a uniform distribution between 0 and 1. Equivalent to np.random.uniform(size=(3, 1), low=0., high=1.).
x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
print('x: ', x)

# Listing 3.3 NumPy arrays are assignable
x = np.ones(shape=(2, 2))
x[0, 0] = 0.
print('x: ', x)

# Listing 3.4 TensorFlow tensors are not assignable
x = tf.ones(shape=(2, 2))
# x[0, 0] = 0. #will raise an error
print('x: ', x)


# Listing 3.5 Creating a TensorFlow variable
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print('v: ', v)

# Listing 3.6 Assigning a value to a TensorFlow variable
v.assign(tf.ones((3, 1)))
print('v: ', v)

# Listing 3.7 Assigning a value to a subset of a TensorFlow variable
v[0, 0].assign(3.)
print('v: ', v)

# Listing 3.8 Using assign_add()
v.assign_add(tf.ones((3, 1)))
print('v: ', v)

# 3.5.2 Tensor operations: Doing math in TensorFlow
# Listing 3.9 A few basic math operations
a = tf.ones((2, 2))
b = tf.square(a)
c = tf.sqrt(a)
d = b + c
e = tf.matmul(a, b)
f = e*d

print('a: ', a)
print('b: ', b)
print('c: ', c)
print('d: ', d)
print('e: ', e)
print('f: ', f)


