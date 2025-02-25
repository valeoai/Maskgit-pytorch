import os
import glob
import webdataset as wds

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.stl10 import STL10
from torchvision.datasets.coco import CocoCaptions
from torchvision.datasets import ImageFolder

from Dataset.dataset import ImageNetKaggle
from Dataset.dataset import CodeDataset


def get_data(data, img_size, data_folder, bsize, num_workers, is_multi_gpus, seed):
    """ Class to load data """

    if data == "mnist":
        data_train = MNIST('./dataset_storage/mnist/', download=False,
                           transform=transforms.Compose([transforms.Resize(img_size),
                                                         transforms.ToTensor(),
                                                         ]))

    elif data == "cifar10":
        data_train = CIFAR10(data_folder, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

        data_test = CIFAR10(data_folder, train=False, download=False,
                            transform=transforms.Compose([
                                transforms.Resize(img_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

    elif data == "stl10":
        data_train = STL10('./Dataset/stl10', split="train+unlabeled",
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
        data_test = STL10('./Dataset/stl10', split="test",
                          transform=transforms.Compose([
                              transforms.Resize(img_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]))

    elif data == "imagenet":
        t_train = transforms.Compose([transforms.Resize(img_size),
                                      transforms.CenterCrop((img_size, img_size)),
                                      # transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                      ])

        t_test = transforms.Compose([transforms.Resize(img_size),
                                     transforms.CenterCrop((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                     ])

        try:
            data_train = ImageFolder(data_folder + "/train", transform=t_train)
            data_test = ImageFolder(data_folder + "val", transform=t_test)

        except:
            data_train = ImageNetKaggle(data_folder, split="train", img_size=img_size, transform=t_train)
            data_test = ImageNetKaggle(data_folder, "val", img_size=img_size, transform=t_test)

    elif data == "imagenet_feat":
        data_train = CodeDataset(data_folder + "Train")
        data_test = CodeDataset(data_folder + "Eval")

    elif data == "mscoco":
        data_test = CocoCaptions(root=os.path.join(data_folder, 'images/val2017/'),
                                 annFile=os.path.join(data_folder, 'annotations/captions_val2017.json'),
                                 transform=transforms.Compose([
                                     transforms.Resize(img_size),
                                     transforms.CenterCrop((img_size, img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[.5, .5, .5],
                                         std=[.5, .5, .5])
                                 ]),
                                 target_transform=lambda x: x[:5])
        test_sampler = DistributedSampler(data_test, shuffle=False, seed=seed) if is_multi_gpus else None
        test_loader = DataLoader(data_test, batch_size=bsize, shuffle=False,
                                 num_workers=num_workers, pin_memory=True,
                                 drop_last=False, sampler=test_sampler)

        return None, test_loader

    elif data == "webdata":
        # Segment_Anything
        sa_feat = glob.glob(os.path.join(data_folder, "sa_feat/*.tar")) # 307 shards, ~10_000img/shard
        # cc12m
        cc12m = glob.glob(os.path.join(data_folder, "cc12m_feat/*.tar")) # 1242 shards, ~650img/shard
        # DiffusionDB
        diffusiondb = glob.glob(os.path.join(data_folder, "diffusiondb_feat/*.tar")) # 2_000 shards, ~1_000img/shard
        # MidJourneyDB
        midjourneydb = glob.glob(os.path.join(data_folder, "midjourney_feat/*.tar")) # 191 shards, ~20_000img/shard

        urls = list(sa_feat) + list(cc12m) + list(diffusiondb) + list(midjourneydb)
        print(f"number of shard: {len(urls)}") # 16,884,356 ~number of images

        def preprocess(sample):
            keys = sample.keys()
            if 'png.txt' in keys:
                txt = sample['png.txt']
            else:
                txt = sample['txt']

            if 'vq_feat.npy' in keys:
                vq_feat = sample['vq_feat.npy']
            elif 'png.vq_feat.npy' in keys:
                vq_feat = sample['png.vq_feat.npy']
            else:
                vq_feat = sample['vq_feat']

            if 'txt_feat.npy' in keys:
                txt_feat = sample['txt_feat.npy']
            elif 'png.txt_feat.npy' in keys:
                txt_feat = sample['png.txt_feat.npy']
            else:
                txt_feat = sample['txt_feat']

            return txt, torch.LongTensor(vq_feat), torch.FloatTensor(txt_feat)

        dataset = (
            wds.WebDataset(urls, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(1_000) # buffer size for the shuffling, need to be higher than the bsize per node
            .decode("rgb")
            .map(preprocess) # from numpy to tensor
            .batched(bsize)  # batch_size
        )

        train_loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers) # iterable dataset
        train_loader = train_loader.unbatched().shuffle(1000).batched(bsize)
        train_loader = train_loader.with_epoch(16_884_356 // (16*8*4)) # simulates one epoch every 10_000 iterations roughly

        return train_loader, None
    else:
        data_train = None
        data_test = None

    train_sampler = DistributedSampler(data_train, shuffle=True, seed=seed) if is_multi_gpus else None
    test_sampler = DistributedSampler(data_test, shuffle=True, seed=seed) if is_multi_gpus else None

    train_loader = DataLoader(data_train, batch_size=bsize,
                              shuffle=False if is_multi_gpus else True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True, sampler=train_sampler)
    test_loader = DataLoader(data_test, batch_size=bsize,
                             shuffle=False if is_multi_gpus else True,
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, sampler=test_sampler)

    return train_loader, test_loader
