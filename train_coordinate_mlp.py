import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from kan import *

# Load and process the image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((64, 64))  # Resize for simplicity
    image_array = np.array(image).astype(np.float32) / 255.0
    return image_array

image_path = 'pykan_coordinate_mlp/img_schloss.jpg'
image_data = load_image(image_path)

# Prepare the data
height, width, _ = image_data.shape
X = np.array([[i, j] for i in range(height) for j in range(width)])  # Pixel locations
y = image_data.reshape(-1, 3)  # Corresponding pixel colors

# Split the data manually
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_kan_dataset = dict()
train_kan_dataset["train_input"] = X_train
train_kan_dataset["train_label"] = y_train
train_kan_dataset["test_input"] = X_test
train_kan_dataset["test_label"] = y_test

# Define MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize MLP and KAN
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

mlp = MLP(input_dim=input_dim, output_dim=output_dim)
kan = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)

# Define loss function and optimizers
criterion = nn.MSELoss()
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)
optimizer_kan = torch.optim.Adam(kan.parameters(), lr=0.001)

# Training loop for MLP
def train_model(model, optimizer, X_train, y_train, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

print("Training MLP...")
train_model(mlp, optimizer_mlp, X_train, y_train)

print("Training KAN...")
results_kan = kan.train(train_kan_dataset, opt="LBFGS", steps=1000, lamb=0.01, lamb_entropy=10.);
#kan.plot()

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        loss = criterion(predictions, y_test)
    return loss.item()

mlp_loss = evaluate_model(mlp, X_test, y_test)
#kan_loss = evaluate_model(kan, X_test, y_test)
kan_loss = results_kan['test_loss']

print(f'MLP Test Loss: {mlp_loss}') #0.018
print(f'KAN Test Loss: {kan_loss}') #0.14