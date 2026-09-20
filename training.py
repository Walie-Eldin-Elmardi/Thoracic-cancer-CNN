from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

dataset = datasets.ImageFolder('./Dataset/train', transform=transforms.Compose([
    transforms.Resize((400,400)),
    transforms.ToTensor()
]))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 100 * 100, 128)
        self.fc2 = nn.Linear(128, 4)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 100 * 100)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(filename):
    model = CNN()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    print(dataset.classes)
    print("Starting training")
    for epoch in range(epochs):
        sum_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {sum_loss/len(loader):.4f}")
    torch.save(model.state_dict(), filename)

def eval(models, test):
    model = CNN()
    model.load_state_dict(torch.load(models))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])
    image = Image.open(test).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    print(f"Predicted class: {dataset.classes[pred]}, Confidence: {confidence:.4f}")