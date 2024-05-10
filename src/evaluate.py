import torch
from torch import nn

from app.models.model import NeuralNet
from src.data import test_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
model = NeuralNet().to(device)
model.load_state_dict(torch.load("../app/models/model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


if __name__ == "__main__":
    model.eval()
    for i in range(10):
        x, y = test_data[i][0], test_data[i][1]
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
