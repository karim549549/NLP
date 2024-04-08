import pandas as pd
import torch

from LSTMModel import LSTMModel
from PreProcessing import Preprocessing
import torch.nn as nn
import torch.optim as optim


df = pd.read_csv("Amazon.csv")

preprocessor = Preprocessing(df)
df = preprocessor.dateTimeFormater(df, 'Date')
df = preprocessor.standardScaler()
df = preprocessor.outlierRemoval()

x_train, x_test, y_train, y_test = preprocessor.splitter('Open', 42, 0.3)
x_train = preprocessor.creatingSequences(x_train)
x_test = preprocessor.creatingSequences(x_test)
y_train = preprocessor.creatingSequences(y_train)
y_test = preprocessor.creatingSequences(y_test)


x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


input_size = x_train.shape[2]
hidden_size = 128
output_size = 1
num_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, num_layers)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


with torch.no_grad():
    model.eval()
    test_outputs = model(x_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')
