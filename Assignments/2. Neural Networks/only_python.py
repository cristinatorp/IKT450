import math
import random

all_data = []
with open("./data/ecoli.data", "r") as file:
    [all_data.append(line.split()) for line in file]

data = []
for row in all_data:
    for i in range(1, len(row) - 1):
        row[i] = float(row[i])          # Cast all strings to floats

    if row[-1] == "pp":
        row[-1] = 1                     # Classify "pp" as 1 for binary linearity
        data.append(row[1:])
    elif row[-1] == "im":
        row[-1] = 0                     # Classify "im" as 0 for binary linearity
        data.append(row[1:])

# [print(row) for row in data]

# Split data into 70% training and 30% testing
random.shuffle(data)
training_percent = round((len(data) / 100) * 70)
training_data = data[:training_percent]
testing_data = data[training_percent:]

weights = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]


def sigmoid(z):
    if z < -100:
        return 0
    if z > 100:
        return 0
    return 1.0 / math.exp(-z)


def first_layer(x, w):
    activation_1 = w[0]
    activation_1 += w[1] * x[0]
    activation_1 += w[2] * x[1]

    activation_2 = w[3]
    activation_2 += w[4] * x[2]
    activation_2 += w[5] * x[3]

    activation_3 = w[6]
    activation_3 += w[7] * x[4]
    activation_3 += w[8] * x[5]
    activation_3 += w[9] * x[6]

    return sigmoid(activation_1), sigmoid(activation_2), sigmoid(activation_3)


def second_layer(x, w):
    activation_4 = w[10]
    activation_4 += w[11] * x[0]
    activation_4 += w[12] * x[1]

    activation_5 = w[13]
    activation_5 += w[14] * x[1]
    activation_5 += w[15] * x[2]

    return sigmoid(activation_4), sigmoid(activation_5)


def third_layer(x, w):
    activation_6 = w[16]
    activation_6 += w[17] * x[0]
    activation_6 += w[18] * x[1]

    return sigmoid(activation_6) # Maybe linear step function instead?


def predict(x, w):
    input_layer = x
    layer_1 = first_layer(input_layer, w)
    layer_2 = second_layer(layer_1, w)
    layer_3 = third_layer(layer_2, w)
    return layer_3, layer_2, layer_1


# for row in training_data:
#     print(predict(row, weights)[0], row[-1])


def train_weights(train, learning_rate, epochs):
    for epoch in range(epochs):
        sum_error = 0.0
        last_error = 0.0

        for row in train:
            prediction, layer_2, layer_1 = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error

            # Layer 1
            weights[0] = weights[0] + learning_rate * error
            weights[3] = weights[3] + learning_rate * error
            weights[6] = weights[6] + learning_rate * error

            weights[1] = weights[1] + learning_rate * error * row[0]
            weights[2] = weights[2] + learning_rate * error * row[1]
            weights[4] = weights[4] + learning_rate * error * row[2]
            weights[5] = weights[5] + learning_rate * error * row[3]
            weights[7] = weights[7] + learning_rate * error * row[4]
            weights[8] = weights[8] + learning_rate * error * row[5]
            weights[9] = weights[9] + learning_rate * error * row[6]

            # Layer 2
            weights[10] = weights[10] + learning_rate * error
            weights[13] = weights[13] + learning_rate * error

            weights[11] = weights[11] + learning_rate * error * layer_1[0]
            weights[12] = weights[12] + learning_rate * error * layer_1[1]
            weights[14] = weights[14] + learning_rate * error * layer_1[1]
            weights[15] = weights[15] + learning_rate * error * layer_1[2]

            # Layer 3
            weights[16] = weights[16] + learning_rate * error

            weights[17] = weights[17] + learning_rate * error * layer_2[0]
            weights[18] = weights[18] + learning_rate * error * layer_2[1]

        if epoch % 100 == 0 or last_error != sum_error:
            print(f"Epoch {epoch}=>Error {sum_error}")

        last_error = sum_error

    return weights


learning_rate = 0.0001  # 0.00001
epochs = 1000   # 10000
train_weights = train_weights(training_data, learning_rate, epochs)
print(train_weights)



