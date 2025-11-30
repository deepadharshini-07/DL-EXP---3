# DL- Developing a Recurrent Neural Network Model for Stock Prediction
# AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data..

# THEORY
A Recurrent Neural Network (RNN) is a type of deep learning model designed to handle sequential data, such as time series like stock prices. It processes previous inputs through loops, allowing it to capture temporal dependencies and patterns over time. When used for stock price prediction, the RNN analyzes historical price data to learn trends and make future price estimates. Its ability to remember information across sequences makes it suitable for modeling the dynamic and seasonal nature of stock markets. Overall, RNNs help improve forecast accuracy by leveraging past data to inform future predictions.

# DESIGN STEPS
Collect historical stock closing price data.

Preprocess the data (normalize and format into sequences).

Split the dataset into training and testing sets.

Define the RNN model architecture (input layer, hidden RNN/LSTM/GRU layers, output layer).

Compile the model with suitable loss function and optimizer.

Train the model on the training dataset.

Validate the model using the testing dataset.

Evaluate model performance using metrics like RMSE or MSE.

Predict future stock prices using the trained model.

Visualize actual vs. predicted stock prices for analysis.

# PROGRAM
# Name: Deepadharshini T
# Register Number: 212224230052

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv('trainset.csv')
df_test = pd.read_csv('testset.csv')
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)
scaler=MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences (data, seq_length):
  x = []
  y = []
  for i in range(len(data)-seq_length):
    x.append(data[i:i+seq_length])
    y.append(data[i+seq_length])
  return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences (scaled_train, seq_length)
x_test, y_test = create_sequences (scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor (y_test, dtype=torch.float32)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader (train_dataset, batch_size=64, shuffle=True)

class RNNModel (nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super (RNNModel, self). __init__()
    self.rnn = nn. RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn. Linear(hidden_size, output_size)
  def forward(self, x):
    out,_=self.rnn(x)
    out=self.fc (out[:,-1,:])
    return out

model=RNNModel()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
!pip install torchinfo

from torchinfo import summary
summary(model,input_size=(64,60,1))
criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=20):
  train_losses = []
  model.train()
  for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      outputs=model(x_batch)
      loss=criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    train_losses.append(total_loss/len(train_loader))
    print (f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/ len(train_loader):.4f}')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()

train_model (model, train_loader, criterion, optimizer)
model.eval()
with torch.no_grad():
  predicted = model(x_test_tensor.to(device)).cpu().numpy()
  actual=y_test_tensor.cpu().numpy()

predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()

print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```
# OUTPUT
# Training Loss Over Epochs Plot
<img width="840" height="648" alt="image" src="https://github.com/user-attachments/assets/7954acc1-fe35-441f-9fc7-6d04400f0d4e" />


# True Stock Price, Predicted Stock Price vs time
<img width="944" height="579" alt="image" src="https://github.com/user-attachments/assets/7c5ac3e6-583b-4932-9265-d31aa4520eb6" />

# prediction graph
<img width="365" height="87" alt="image" src="https://github.com/user-attachments/assets/706240d1-087d-4cbe-957f-d1a86d396dbc" />


# RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
