import torch
from torch import nn
from src.data import train_loader
from models.model import NeuralNet


def train_model(dataloader, model, criterion, optimizer, epochs, device="cpu"):
    """
    Trains a model for a given number of epochs.

    Args:
        dataloader: the train dataloader.
        model: The model to train.
        criterion: The loss function to use.
        optimizer: The optimizer to use.
        epochs: The number of epochs to train for.
        device: The device to train the model on, "cpu" or "cuda".

    Returns:
        The trained model.
    """

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        # Set the model to training mode
        model.train()
        correct = 0
        size = len(dataloader.dataset)
        # Iterate over the training data
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            # Get the number of correct predictions
            correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        correct /= size
        # Print the training loss
        print(f"Training loss: {loss.item():.4f}")
        print(f"Training Accuracy: {100 * correct:.2f}%")

    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet().to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimize algorithm
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5

    # Train teh model
    train_model(train_loader, model, criterion, optimizer, epochs, device)
    # Save the state dict of model
    torch.save(model.state_dict(), "../artifacts/models/model.pth")
    print("Saved PyTorch Model State to model.pth")
