import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

# Generate synthetic data
X, _ = make_blobs(n_samples=3000, centers=10, cluster_std=1.0, random_state=42)

print(X.shape) # (300, 2)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x, condition):
        encoded = self.encoder(x)
        if condition == 1:
            print("=============latent_space===============")
            print(encoded[:,0].shape)
            
            #encoded[:,0] = 0
            
        decoded = self.decoder(encoded)
        return decoded

# Convert data to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
print(X_tensor.shape) # torch.Size([300, 2])

# Initialize the autoencoder model
input_dim = X.shape[1]
print(input_dim) # 2
latent_dim = 2  # Latent dimension
autoencoder = Autoencoder(input_dim, latent_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(X_tensor,0)
    loss = criterion(outputs, X_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


X_2, _ = make_blobs(n_samples=3000, centers=17, cluster_std=0.5, random_state=42)
X_tensor_2 = torch.tensor(X_2, dtype=torch.float32)

# Generate new data using the trained autoencoder
with torch.no_grad():
    generated_data = autoencoder(X_tensor_2,1).numpy()
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
