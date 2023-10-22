# Abstract class for the trainer
import os
import json

import numpy as np
from PIL import Image
import webdataset as wds

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.coco import CocoCaptions
from torchvision.datasets import ImageFolder


def custom_lamda(x):
    return x[:5]


class Trainer(object):
    """ Abstract class for each trainer """

    vit = None
    optim = None

    def __init__(self, args):
        """ Initialization of the Trainer """
        self.args = args
        self.writer = None if args.writer_log == "" else SummaryWriter(log_dir=args.writer_log)  # Tensorboard writer

    def get_data(self):
        """ class to load data """
        if self.args.data == "mnist":
            data_train = MNIST('./Dataset/mnist', download=False,
                               transform=transforms.Compose([transforms.Resize(self.args.img_size),
                                                             transforms.ToTensor(),
                                                             ]))
        elif self.args.data == "cifar10":
            data_train = CIFAR10('/Dataset/CIFAR10/', train=True, download=False,
                                 transform=transforms.Compose([
                                           transforms.Resize(self.args.img_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ]))
            data_test = CIFAR10('/Dataset/CIFAR10/', train=False, download=False,
                                transform=transforms.Compose([
                                    transforms.Resize(self.args.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

        elif self.args.data == "stl10":
            data_train = STL10('./Dataset/stl10', split="train+unlabeled",
                               transform=transforms.Compose([
                                   transforms.Resize(self.args.img_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
            data_test = STL10('./Dataset/stl10', split="test",
                              transform=transforms.Compose([
                                  transforms.Resize(self.args.img_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))

        elif self.args.data == "imagenet":
            # '/Dataset/ImageNet/train'
            t_train = transforms.Compose([transforms.Resize(self.args.img_size),
                                          transforms.RandomCrop((self.args.img_size, self.args.img_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          # transforms.Normalize(
                                          #    mean=[0.485, 0.456, 0.406],
                                          #    std=[0.229, 0.224, 0.225])
                                          ])
            t_test = transforms.Compose([transforms.Resize(self.args.img_size),
                                         transforms.CenterCrop((self.args.img_size, self.args.img_size)),
                                         transforms.ToTensor(),
                                         # transforms.Normalize(
                                         #     mean=[0.485, 0.456, 0.406],
                                         #     std=[0.229, 0.224, 0.225])
                                         ])

            try:
                data_train = ImageFolder(os.path.join(self.args.data_folder, "train"), transform=t_train)
                data_test = ImageFolder(os.path.join(self.args.data_folder, "val"), transform=t_test)
            except:
                data_train = ImageNetKaggle(self.args.data_folder, "train", transform=t_train)
                data_test = ImageNetKaggle(self.args.data_folder, "val", transform=t_test)

        elif self.args.data == "mscoco":
            data_train = CocoCaptions(root='/datasets_master/COCO/images/train2017/',
                                      annFile='/datasets_master/COCO/annotations/captions_train2017.json',
                                      transform=transforms.Compose([
                                                transforms.Resize(self.args.img_size),
                                                transforms.RandomCrop((self.args.img_size, self.args.img_size)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])

                                      ]),
                                      target_transform=custom_lamda)
            data_test = CocoCaptions(root='/datasets_master/COCO/images/val2017/',
                                     annFile='/datasets_master/COCO/annotations/captions_val2017.json',
                                     transform=transforms.Compose([
                                               transforms.Resize(self.args.img_size),
                                               transforms.CenterCrop((self.args.img_size, self.args.img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
                                     ]),
                                     target_transform=custom_lamda)

        elif self.args.data == "cc3m":
            def preprocess(sample):
                img, annotation = sample

                caption = annotation['caption']
                t_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(self.args.img_size),
                                              transforms.RandomCrop((self.args.img_size, self.args.img_size)),
                                              ])
                img = t_train(img)
                img = (2 * img) - 1
                return img, caption[:75]

            dataset = wds.WebDataset(None, resampled=True)
            dataset = dataset.shuffle(1000).decode("rgb").to_tuple("jpg", "json").map(preprocess).batched(self.args.bsize).with_epoch(10000)
            train_loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, pin_memory=True, num_workers=self.args.num_workers)
            train_loader = train_loader.unbatched().shuffle(1000).batched(self.args.bsize)

            return train_loader, None

        train_sampler = DistributedSampler(data_train, shuffle=True) if self.args.is_multi_gpus else None
        test_sampler = DistributedSampler(data_test, shuffle=True) if self.args.is_multi_gpus else None

        train_loader = DataLoader(data_train, batch_size=self.args.bsize,
                                  shuffle=False if self.args.is_multi_gpus else True,
                                  num_workers=self.args.num_workers, pin_memory=True,
                                  drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(data_test, batch_size=self.args.bsize,
                                 shuffle=False if self.args.is_multi_gpus else True,
                                 num_workers=self.args.num_workers, pin_memory=True,
                                 sampler=test_sampler)

        return train_loader, test_loader

    def get_network(self, archi, pretrained_file=None):
        pass

    def log_add_img(self, names, img, iter):
        """ Add an image in tensorboard"""
        if self.writer is None:
            return
        self.writer.add_image(tag=names, img_tensor=img, global_step=iter)

    def log_add_scalar(self, names, scalar, iter):
        """ Add scalar value in tensorboard """
        if self.writer is None:
            return
        if isinstance(scalar, dict):
            self.writer.add_scalars(main_tag=names, tag_scalar_dict=scalar, global_step=iter)
        else:
            self.writer.add_scalar(tag=names, scalar_value=scalar, global_step=iter)

    @staticmethod
    def get_optim(net, lr, mode="AdamW", **kwargs):
        """ Get the optimizer Adam or Adam with weight decay """
        if isinstance(net, list):
            params = []
            for n in net:
                params += list(n.parameters())
        else:
            params = net.parameters()

        if mode == "AdamW":
            return optim.AdamW(params, lr, weight_decay=1e-5, **kwargs)
        elif mode == "Adam":
            return optim.Adam(params, lr, **kwargs)
        return None

    @staticmethod
    def get_loss(mode="l1", **kwargs):
        """ return the loss """
        if mode == "l1":
            return nn.L1Loss()
        elif mode == "l2":
            return nn.MSELoss()
        elif mode == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        return None

    def train_one_epoch(self, epoch):
        return

    def fit(self):
        pass

    @torch.no_grad()
    def eval(self):
        pass

    def sample(self, nb_sample):
        pass

    @staticmethod
    def all_gather(obj, gpus, reduce="mean"):
        """ Gather the value obj from all GPUs and return the mean or the sum """
        tensor_list = [torch.zeros(1) for _ in range(gpus)]
        dist.all_gather_object(tensor_list, obj)
        obj = torch.FloatTensor(tensor_list)
        if reduce == "mean":
            obj = obj.mean()
        elif reduce == "sum":
            obj = obj.sum()
        elif reduce == "none":
            pass
        else:
            raise NameError("reduction not known")

        return obj

    def save_network(self, model, path, iter=None, optimizer=None, global_epoch=None):
        """ Save the state of the model, including the iteration,
            the optimizer state and the current epoch"""
        if self.args.is_multi_gpus:
            torch.save({'iter': iter,
                        'global_epoch': global_epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       path)
        else:
            torch.save({'iter': iter,
                        'global_epoch': global_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       path)


class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
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
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]
