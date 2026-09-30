from collections import Counter

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from TwoImageDataset import TwoImageDataset

dataset = TwoImageDataset('./Preprocessed/edges/train', './Preprocessed/texture/train', transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
]))

# dataset = datasets.ImageFolder('./Preprocessed/train', transform=transforms.Compose([
#     # transforms.Resize((400,400)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor()
# ]))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def train(filename):
    model = CNN()
    model.train()
    device = torch.device("cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    print(dataset.classes)
    print("Starting training")
    for epoch in range(epochs):
        sum_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {sum_loss/len(loader):.4f}")
    torch.save(model.state_dict(), filename)


def check_dataset():
    print(dataset.classes)
    print(dataset.class_to_idx)
    print(Counter(dataset.targets))