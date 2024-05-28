import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from config import num_procs as n
# Transform for the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 to fit the network's input size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Number of parts
# Shuffle the dataset and split it into n parts
num_samples = len(mnist_data)
part_size = num_samples // n
indices = torch.randperm(num_samples).tolist()
parts = [indices[i * part_size:(i + 1) * part_size] for i in range(n)]

# Function to split a part into train, validation, and test sets
def split_part(indices, train_frac=0.7, val_frac=0.2, test_frac=0.1):
    part_size = len(indices)
    train_size = int(train_frac * part_size)
    val_size = int(val_frac * part_size)
    test_size = part_size - train_size - val_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

# Create DataLoaders for each part
train_loaders = []
val_loaders = []
test_loaders = []

for part in parts:
    train_indices, val_indices, test_indices = split_part(part)
    
    train_subset = Subset(mnist_data, train_indices)
    val_subset = Subset(mnist_data, val_indices)
    test_subset = Subset(mnist_data, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)
    
    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

# Single test loader for the original test dataset
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False, num_workers=2)
