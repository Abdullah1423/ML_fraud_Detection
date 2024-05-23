from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load your dataset
# Assuming you have a CSV file where each row represents a transaction and columns represent features
dataset = pd.read_csv("dataset.csv")

# Preprocess your data, handle missing values, encode categorical variables, etc.
# Assuming your target variable is named 'fraud' and all other columns are features
X = dataset.drop('fraud', axis=1)
y = dataset['fraud']

# Apply undersampling to balance the data
undersampler = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = undersampler.fit_resample(X, y)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert pandas DataFrame to numpy arrays
X_train_array = X_train.values
y_train_array = y_train.values
X_test_array = X_test.values
y_test_array = y_test.values

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32)

# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 100

# Initialize the model, loss function, and optimizer
model = FFNN(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    model.eval()
    y_pred = model(X_test_tensor)
    y_pred = (y_pred > 0.5).float()
    accuracy = accuracy_score(y_test_tensor, y_pred)
    precision = precision_score(y_test_tensor, y_pred)
    recall = recall_score(y_test_tensor, y_pred)
    f1 = f1_score(y_test_tensor, y_pred)
    conf_matrix = confusion_matrix(y_test_tensor, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
