import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

# choose the top 10,000 most frequently words in the dataset
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.imdb.load_data(num_words=10000)


# print('train_data.shape: ', train_data.shape, ', train_data.dtype: ', train_data.dtype, ', train_data: ', train_data)
# print('train_label.shape: ', train_label.shape, ', train_label.dtype: ', train_label.dtype, ', train_label: ', train_label) 

# max must <10000
# print('max: ', max([max(sequence) for sequence in train_data]))

# 
word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])


# print ('decoded_review: ', decoded_review)

def vectorize_sequences(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# print('x_train.shape: ', x_train.shape, ', x_train.dtype: ', x_train.dtype, ', x_train: ', x_train)

y_train = np.asarray(train_label).astype("float32")
y_test = np.asarray(test_label).astype("float32")

# print('train_label: ', train_label)
# print('y_train: ', y_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=4,
                    batch_size=512,
                    validation_data=(x_val, y_val))
# Listing 4.8 Plotting the training and validation loss
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")        
plt.plot(epochs, val_loss_values, "b", label="Validation loss")   
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Listing 4.9 Plotting the training and validation accuracy
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Using a trained model to generate predictions on new data
# print(model.predict(x_test))