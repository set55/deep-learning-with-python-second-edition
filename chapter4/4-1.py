import tensorflow as tf

(train_data, train_label), (test_data, test_label) = tf.keras.datasets.imdb.load_data(num_words=10000)


# print('train_data.shape: ', train_data.shape, ', train_data.dtype: ', train_data.dtype, ', train_data: ', train_data)
# print('train_label.shape: ', train_label.shape, ', train_label.dtype: ', train_label.dtype, ', train_label: ', train_label) 

print('max: ', max([max(sequence) for sequence in train_data]))

word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])


print ('decoded_review: ', decoded_review)