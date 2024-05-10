# Import modules
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the train data and test data.
train_data = datasets.FashionMNIST(root='../data/raw/', train=True, transform=ToTensor(), download=True)
test_data = datasets.FashionMNIST(root='../data/raw/', train=False, transform=ToTensor(), download=True)

# Create data loaders.
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

if __name__ == "__main__":
    for X, y in test_loader:
        print("Data Download Successfully!")
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
