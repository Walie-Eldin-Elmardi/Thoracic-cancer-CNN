from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class TwoImageDataset(Dataset):

    def __init__(self, edge_dir, texture_dir, transform=None):

        self.edge_dir = edge_dir
        self.texture_dir = texture_dir
        self.transform = transform

        self.classes = sorted(os.listdir(edge_dir))
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }

        self.samples = []

        for cls in self.classes:

            edge_class_dir = os.path.join(edge_dir, cls)
            texture_class_dir = os.path.join(texture_dir, cls)

            for filename in os.listdir(edge_class_dir):

                edge_path = os.path.join(edge_class_dir, filename)
                texture_path = os.path.join(texture_class_dir, filename)

                if os.path.exists(texture_path):
                    self.samples.append(
                        (edge_path, texture_path,
                         self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        edge_path, texture_path, label = self.samples[index]

        edge = Image.open(edge_path).convert("L")
        texture = Image.open(texture_path).convert("L")

        if self.transform:
            edge = self.transform(edge)
            texture = self.transform(texture)

        # (1, H, W) + (1, H, W)
        # -> (2, H, W)
        image = torch.cat((edge, texture), dim=0)

        return image, label