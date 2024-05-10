import torch
from torch import nn

from app.models.model import NeuralNet
from src.data import test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = test_loader
criterion = nn.CrossEntropyLoss()
model = NeuralNet().to(device)
model.load_state_dict(torch.load("../app/models/model.pth"))


def eval_test(dataloader, model, criterion):
    """
    Evaluates the performance of a trained model on a test dataset.

    Args:
        dataloader: The data loader for the test dataset.
        model: The trained model to be evaluated.
        criterion: The loss function to be used for evaluation.

    Returns:
        float: The average loss on the test dataset.
        float: The accuracy on the test dataset.
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            # Move data and labels to the same device as the model
            X, y = X.to(device), y.to(device)

            # Get model predictions
            outputs = model(X)

            # Calculate loss
            test_loss += criterion(outputs, y).item()

            # Count correct predictions
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    # Calculate average loss and accuracy
    test_loss /= num_batches
    correct /= size

    # Print results
    print(f"Test loss: {test_loss:.4f}")
    print(f"Train Accuracy: {100 * correct:.2f}%")

    return test_loss, correct


if __name__ == "__main__":
    dataloader = test_loader
    criterion = nn.CrossEntropyLoss()
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load("../app/models/model.pth"))
    eval_test(dataloader, model, criterion)
