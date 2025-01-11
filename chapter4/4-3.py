import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.boston_housing.load_data()


# Normalize the data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# Build the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model

k = 4 
num_val_samples = len(train_data) // k
num_epochs = 100 
all_scores = [] 
# for i in range(k):
#     print(f"Processing fold #{i}")
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples],
#          train_data[(i + 1) * num_val_samples:]],
#         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_labels[:i * num_val_samples],
#          train_labels[(i + 1) * num_val_samples:]],
#         axis=0)
#     model = build_model()
#     model.fit(partial_train_data,
#               partial_train_targets,
#               epochs=num_epochs, 
#               batch_size=16, verbose=1)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
#     all_scores.append(val_mae)

# print('all_scores: ', all_scores)
# print('np.mean(all_scores): ', np.mean(all_scores))


num_epochs = 100
all_mae_histories = [] 
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_labels[:i * num_val_samples],
         train_labels[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data,
                        partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs,
                        batch_size=16,
                        verbose=1
                        )
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)
print('all_mae_histories: ', all_mae_histories)
print(len(all_mae_histories), len(all_mae_histories[0]))

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(len(average_mae_history))

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

truncated_mae_history = average_mae_history[10:]
print(len(truncated_mae_history))

plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()


model = build_model()
model.fit(train_data, train_labels,
          epochs=130, batch_size=16, verbose=1)
test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
print('test_mae_score: ', test_mae_score)


predictions = model.predict(test_data)

#compare the first prediction with the first label
print(predictions[0], test_labels[0])
