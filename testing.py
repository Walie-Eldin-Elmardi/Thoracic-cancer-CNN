from training import CNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from TwoImageDataset import TwoImageDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# dataset = datasets.ImageFolder('./Preprocessed/valid', transform=transforms.Compose([
#     # transforms.Resize((400,400)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor()
# ]))
dataset = TwoImageDataset('./Preprocessed/edges/train', './Preprocessed/texture/train', transform=transforms.Compose([
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



from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import torch

def test(models):
    model = CNN()
    model.load_state_dict(torch.load(models))
    model.eval()

    correct = [0] * len(dataset.classes)
    total = [0] * len(dataset.classes)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for label, prediction in zip(labels, predicted):
                total[label.item()] += 1
                if label == prediction:
                    correct[label.item()] += 1

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Total accuracy
    total_correct = sum(correct)
    total_samples = sum(total)
    total_accuracy = 100 * total_correct / total_samples
    print("=" * 60)
    print(f"Total Accuracy: {total_accuracy:.2f}% ({total_correct}/{total_samples})")
    print("=" * 60)

    # Per-class precision, recall, f1
    precision = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    print(f"\n{'Class':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 68)
    for i in range(len(dataset.classes)):
        acc = 100 * correct[i] / total[i] if total[i] > 0 else 0
        print(f"{dataset.classes[i]:<20}{acc:>9.2f}%  {precision[i]:>10.4f}  {recall[i]:>10.4f}  {f1[i]:>10.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(dataset.classes))))

    print("\n" + "=" * 60)
    print("Confusion Matrix (rows = True, cols = Predicted)")
    print("=" * 60)

    # Column header
    header = f"{'':<18}" + "".join(f"{c[:10]:>12}" for c in dataset.classes)
    print(header)
    print("-" * len(header))

    # Rows
    for i, row in enumerate(cm):
        row_str = f"{dataset.classes[i][:16]:<18}" + "".join(f"{v:>12}" for v in row)
        print(row_str)

    return cm