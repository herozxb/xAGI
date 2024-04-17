import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data from two Gaussian distributions
def generate_data(num_samples, input_size):
    data1 = np.random.normal(loc=-2, scale=1, size=(num_samples // 2, input_size))
    data2 = np.random.normal(loc=2, scale=1, size=(num_samples // 2, input_size))
    data = np.concatenate((data1, data2), axis=0)
    return data

# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Output a probability
        )
    
    def forward(self, x):
        return self.net(x)

# Hyperparameters
input_size = 1
hidden_size = 32
output_size = 1
num_epochs = 10000
batch_size = 64
learning_rate = 0.001

# Generate synthetic data
data = generate_data(1000, input_size)

# Initialize networks and optimizers
generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary cross-entropy loss
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(len(data) // batch_size):
        # Generate real and fake data
        real_data = torch.tensor(data[i * batch_size: (i + 1) * batch_size], dtype=torch.float32)
        fake_data = generator(torch.randn(batch_size, input_size))
        
        # Train Discriminator
        dis_optimizer.zero_grad()
        dis_loss = criterion(discriminator(real_data), torch.ones(batch_size, 1)) + \
                   criterion(discriminator(fake_data.detach()), torch.zeros(batch_size, 1))
        dis_loss.backward()
        dis_optimizer.step()
        
        # Train Generator
        gen_optimizer.zero_grad()
        gen_loss = criterion(discriminator(fake_data), torch.ones(batch_size, 1))
        gen_loss.backward()
        gen_optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {dis_loss.item():.4f}")

# Generate synthetic data using the trained generator
num_samples = 1000
synthetic_data = generator(torch.randn(num_samples, input_size)).detach().numpy()

# Plot the synthetic data and original data
plt.hist(data, bins=30, alpha=0.5, label='Original Data')
plt.hist(synthetic_data, bins=30, alpha=0.5, label='Synthetic Data')
plt.legend()
plt.title('Generated Data vs. Original Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

