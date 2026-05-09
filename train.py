import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("/tmp/data", train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = Net().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if i % 200 == 0:
                print(f"Epoch {epoch} step {i} loss {loss.item():.4f}")

    os.makedirs("/mnt/model", exist_ok=True)
    torch.save(model.state_dict(), "/mnt/model/mnist_cnn.pt")
    print("Saved model to /mnt/model/mnist_cnn.pt")

if __name__ == "__main__":
    main()
