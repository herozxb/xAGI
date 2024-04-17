import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data using make_blobs
X, _ = make_blobs(n_samples=3000, centers=10, cluster_std=1.0, random_state=42)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded
        
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
# Convert numpy array to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

# Initialize Autoencoder model
input_dim = X.shape[1]
latent_dim = 2  # Dimensionality of latent space
autoencoder = Autoencoder(input_dim, latent_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = autoencoder(X_tensor)
    loss = criterion(outputs, X_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print training progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        
# Generate new data using the trained autoencoder
with torch.no_grad():
    generated_data = autoencoder(X_tensor).numpy()        
        
# Generate new images using the trained decoder
with torch.no_grad():
    # Generate random latent vectors (e.g., from a normal distribution)
    num_images = 3
    random_latent = torch.randn(num_images, latent_dim)  # 10 random latent vectors
    random_latent[0][0] = 0
    random_latent[0][1] = 0
    random_latent[1][0] = 1
    random_latent[1][1] = 1
    random_latent[2][0] = -1
    random_latent[2][1] = -1
    print("====================size======================")
    print(random_latent)
    print(random_latent.shape)

    generated_images = autoencoder.decoder(random_latent)

    # Convert generated images to numpy array for visualization
    generated_images = generated_images.numpy()

'''
    # Plot the generated images
    plt.figure(figsize=(12, 6))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Generated Image {i+1}')
        plt.axis('off')

    plt.show()
    
'''

# Plot original and generated data
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Training Data', alpha=0.5)
plt.title('Training Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 2)
plt.scatter(generated_data[:, 0], generated_data[:, 1], c='green', label='AutoEncoder Data', alpha=0.5)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 3, 3)
plt.scatter(generated_images[:, 0], generated_images[:, 1], c='red', label='Latent Vector Generated Data', alpha=0.5)
plt.title('Generated Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
