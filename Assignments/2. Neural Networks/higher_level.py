import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
training_percent = round((len(data) / 100) * 70)
training_data = data[:training_percent]
testing_data = data[training_percent:]

X = torch.Tensor([i[0:7] for i in training_data])
Y = torch.Tensor([i[7] for i in training_data])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, 7)
        self.fc2 = nn.Linear(7, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

all_loss = []

for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()
    all_loss.append(loss.item())

plt.plot(all_loss)
plt.title(r"Total loss over 100 epochs with $\lambda = 0.001$")
plt.text(6, 0.18, r"$learning\ rate=0.001$")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


