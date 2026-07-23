# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Yuvaram S

### Register Number: 212224230315

```python
"""
Exp-1: Simple Feedforward Neural Network for Regression
Learns the relationship between Input and Output from Exp-1.csv
Expected relationship: Output ≈ 4.5 × Input + 0.5
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
data = pd.read_csv('Exp-1.csv')
print(data.head())

x = data[['Input']]
y = data[['Output']]


# ---------------------------------------------------------
# 2. Scale features and target (separate scalers!)
# ---------------------------------------------------------
scale_x = MinMaxScaler()
scale_y = MinMaxScaler()

x_S = scale_x.fit_transform(x)
y_S = scale_y.fit_transform(y)


# ---------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    x_S, y_S, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 4. Convert to tensors
# ---------------------------------------------------------
X_train_tensor = torch.tensor(X_Train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_Train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_Test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_Test, dtype=torch.float32).view(-1, 1)


# ---------------------------------------------------------
# 5. Define the model
# ---------------------------------------------------------
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


model = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ---------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------
epochs = 1000
losses = []

for i in range(epochs):
    optimizer.zero_grad()
    pred = model(X_train_tensor)
    loss = criterion(pred, Y_train_tensor)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % 50 == 0:
        print(f"{i}/{epochs}: {loss.item()}")


# ---------------------------------------------------------
# 7. Plot training loss
# ---------------------------------------------------------
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Over Time")
plt.show()


# ---------------------------------------------------------
# 8. Predict on new input
# ---------------------------------------------------------
x11 = torch.Tensor([[16]])
ip = torch.FloatTensor(scale_x.transform(x11))
pred = model(ip)

pred_actual = scale_y.inverse_transform(pred.detach().numpy())
print(f"Predicted Output for Input=16: {pred_actual.item():.2f}")
```

### Dataset Information

<img width="930" height="425" alt="image" src="https://github.com/user-attachments/assets/f7cb3732-cfaa-463e-ab56-6a3136a6ee58" />


### OUTPUT
```
0/1000: 0.18436039984226227
50/1000: 0.0023445680271834135
100/1000: 0.00029114054632373154
150/1000: 6.463929457822815e-05
200/1000: 3.12847550958395e-05
250/1000: 2.6534193239058368e-05
300/1000: 2.3089980459189974e-05
350/1000: 2.0575376765918918e-05
400/1000: 1.8776599972625263e-05
450/1000: 1.742275526339654e-05
500/1000: 1.6443269487353973e-05
550/1000: 1.573728513903916e-05
600/1000: 1.5236218132486101e-05
650/1000: 1.4885816199239343e-05
700/1000: 1.464274919271702e-05
750/1000: 1.447871090931585e-05
800/1000: 1.437309128959896e-05
850/1000: 1.430635074939346e-05
900/1000: 1.4267251572164241e-05
950/1000: 1.4237691175367218e-05
```

### Training Loss Vs Iteration Plot

<img width="1045" height="707" alt="image" src="https://github.com/user-attachments/assets/14437170-2625-4917-9a2c-021f206797fa" />


### New Sample Data Prediction
Sample Input : 16

Sample Output : 72.39494323730469

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
