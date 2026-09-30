from training import CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

dataset = datasets.ImageFolder('./Preprocessed/valid', transform=transforms.Compose([
    # transforms.Resize((400,400)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
]))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

def eval(models, test):
    model = CNN()
    model.load_state_dict(torch.load(models))
    model.eval()
    transform = transforms.Compose([
        # transforms.Resize((400, 400)),
        transforms.Grayscale(num_output_channels=1),
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

def test(models):

    model = CNN()
    model.load_state_dict(torch.load(models))
    model.eval()

    correct = [0] * len(dataset.classes)
    total = [0] * len(dataset.classes)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)



            for label, prediction in zip(labels, predicted):
                total[label.item()] += 1

                if label == prediction:
                    correct[label.item()] += 1

    for i in range(len(dataset.classes)):
        accuracy = 100 * correct[i] / total[i]
        print(f"Class: {dataset.classes[i]}, Accuracy: {accuracy:.2f}%")