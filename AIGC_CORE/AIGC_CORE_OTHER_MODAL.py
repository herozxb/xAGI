import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

#print(X.shape) # (300, 2)

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 1024)
        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, 64)
        self.fc10 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = self.fc10(x)
        return x

# Convert data to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
#print(X_tensor.shape) # torch.Size([300, 2])

# Initialize the autoencoder model
input_dim = X.shape[1]
#print(input_dim) # 2
latent_dim = 2  # Latent dimension
autoencoder = Generator(input_dim, input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(X_tensor)
    loss = criterion(outputs, X_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


X_2, _ = make_blobs(n_samples=300, centers=7, cluster_std=0.5, random_state=42)
X_tensor_2 = torch.tensor(X_2, dtype=torch.float32)

# Generate new data using the trained autoencoder
with torch.no_grad():
    generated_data = autoencoder(X_tensor_2).numpy()
    #generated_data = autoencoder(X_tensor).numpy()

# Plot original and generated data
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Training Data', alpha=0.5)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 2)
plt.scatter(X_2[:, 0], X_2[:, 1], c='green', label='Original Data', alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 3)
plt.scatter(generated_data[:, 0], generated_data[:, 1], c='red', label='Generated Data', alpha=0.5)
plt.title('Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
