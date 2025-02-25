import os
import json
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None):
        # Initialize dataset by providing the image directory and optional transformations
        self.image_dir = image_dir
        self.transform = transform
        # Get a list of all image files in the directory with .png extension
        self.image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]
        self.labels = []
        with open(labels_file, 'r') as file:
            for line in file:
                self.labels.append(line.strip())

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_files)

    def __getitem__(self, idx):
        # get caption
        caption = self.labels[idx]
        # Get the image file path by index
        img_name = os.path.join(self.image_dir, self.image_files[idx])

        # Open the image file
        image = Image.open(img_name)

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return the transformed image
        return image, caption


class CodeDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): Path to the folder containing .pth files.
        """
        self.folder_path = folder_path
        self.file_list = [
            os.path.join(folder_path, file)
            for file in sorted(os.listdir(folder_path))
            if file.endswith(".pth")
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the file to load.

        Returns:
            dict: A dictionary containing "code" and "y".
        """
        file_path = self.file_list[idx]
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        data["code"] = data["code"].astype(np.int32)
        return data


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, img_size=512, transform=None):
        self.samples = []
        self.targets = []
        self.paths = []
        self.transform = transform
        self.img_size = img_size
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
                    self.paths.append(sample_path)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
                self.paths.append(sample_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]